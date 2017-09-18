from __future__ import print_function
from mpi4py import MPI

from nn.nn import *

import time

STEP_START_ = 1

#TAG_LIST_ = [774, 67, 93, 203, 334, 482, 533]
TAG_LIST_ = [i*30 for i in range(50000)]

def backward_step_dist_version(activations, layer, output_grad=None, targets=None):
	'''
	this function is basically the same as the `backward_step_matrix_version` function
	the only difference is that for distributed version, we need to fetch the gradient
	layer by layer, one we gathered the gradient, we send it to the parameter server
	'''
	# each time we update activations again
	Y = activations.pop()

	# we only pass `targets` for output layer
	if targets is not None:
		output_grad = None

	if output_grad is None:
		input_grad = layer.get_input_grad(Y, targets)	
	else:
		input_grad = layer.get_input_grad(Y, output_grad)
	X = activations[-1]
	grads = layer.get_params_grad(X, output_grad)
	if layer.is_fc_layer:
		grads = np.array(grads).reshape((layer.get_shape[0]+1, layer.get_shape[1]))
	output_grad = input_grad
	return grads, output_grad, activations


class WorkerFC_NN(FC_NN):
	def __init__(self, comm, world_size, rank, **kwargs):
		self.comm = comm   # get MPI communicator object
		self.world_size = world_size # total number of processes
		self.rank = rank # rank of this Worker
		#self.status = MPI.Status()
		self.cur_step = 0
		self.next_step = 0 # we will fetch this one from parameter server
		self.fc_layer_counts = []

		self.batch_size = kwargs['batch_size']
		self.lr = kwargs['learning_rate']
		self.max_epochs = kwargs['max_epochs']

		# this one is going to be used to avoid fetch the weights for multiple times
		self._layer_cur_step = []

	def build_model(self, num_layers, layer_config):
		layers = []
		for i in range(num_layers):
		    if layer_config['layer'+str(i)]['type'] == 'activation':
		        if layer_config['layer'+str(i)]['name'] == 'sigmoid':
		            layers.append(SigmoidLayer())
		            layers[-1].layer_index = i
		        elif layer_config['layer'+str(i)]['name'] == 'softmax':
		            layers.append(SoftmaxOutputLayer())
		            layers[-1].layer_index = i
		    elif layer_config['layer'+str(i)]['type'] == 'fc':
				self.fc_layer_counts.append(i)
				# init the layer current step counters:
				self._layer_cur_step.append(0)
				layers.append(LinearLayer(layer_config['layer'+str(i)]['shape'][0], layer_config['layer'+str(i)]['shape'][1], init_mode="default"))
				layers[-1].layer_index = i
		self.module = layers

		return layers

	def train(self, training_set, validation_set):
		# the first step we need to do here is to sync fetch the inital worl_step from the parameter server
		# we still need to make sure the value we fetched from parameter server is 1
		self.sync_fetch_step()
		# do some sync check here
		assert(self.update_step())
		assert(self.cur_step == STEP_START_)

		# number of batches in one epoch
		
		num_batch_per_epoch = training_set.images.shape[0] / self.batch_size
		batch_idx = -1
		epoch_idx = 0
		epoch_avg_loss = 0

		first = True

		print("Worker {}: starting training".format(self.rank))
		# start the training process
		while True:
			# the worker shouldn't know the current global step
			# except received the message from parameter server
			self.async_fetch_step()

			# the only way every worker know which step they're currently on is to check the cur step variable
			updated = self.update_step()

			if (not updated) and (not first):
				# wait here unitl enter next step
				continue

			# the real start point of this iteration
			iter_start_time = time.time()
			first = False
			print("Rank of this node: {}, Current step: {}".format(self.rank, self.cur_step))

			# TODO(hwang): return layer request here and do weight before the forward step begins, rather than implement
			# the wait() in the fetch function
			self.async_fetch_weights()
			
			# start the normal training process
			X_batch, y_batch = training_set.next_batch(batch_size=self.batch_size)

			# manage batch index manually
			batch_idx += 1

			if batch_idx == num_batch_per_epoch - 1:
				batch_idx = 0
				epoch_avg_loss /= num_batch_per_epoch
				print("Average for epoch {} is {}".format(epoch_idx, epoch_avg_loss))
				epoch_idx += 1
				epoch_avg_loss = 0

			logits = forward_step(X_batch, self.module)  # Get the activations
			loss = self.module[-1].get_cost(logits[-1], y_batch)  # Get cost
			epoch_avg_loss += loss

			req_send_check = []
			for layer_idx, layer in enumerate(reversed(self.module)):
				# we make workers 1 and 2 quite slow to simulate the straggler
				if self.rank == 1 or self.rank == 2 or self.rank == 3:
					time.sleep(0.5)
				should_kill = False
				#if self.cur_step > 1:
				#	print("Entering {}th layer, worker {}".format(len(self.module)-1-layer_idx, self.rank))
				# use the probe to check killing signal
				# note that any worker has possibility to be killed
				for ite in range(10000):
					status = MPI.Status()
					#self.comm.Iprobe(MPI.ANY_SOURCE, 13, status)
					self.comm.Iprobe(0, 77, status)
					if status.Get_source() == 0:
						print("Worker {}: I'm the stragger, killing myself!".format(self.rank))
						tmp = self.comm.recv(source=0, tag=77)
						should_kill = True
						break
				if should_kill:
					break
				# gather gradient layer by layer
				if layer.get_name == "softmax_layer":
					# indicate that this layer is output layer
					grads, output_grad, logits = backward_step_dist_version(logits, layer=layer, output_grad=None, targets=y_batch)
				else:
					grads, output_grad, logits = backward_step_dist_version(logits, layer=layer, output_grad=output_grad, targets=None)
				# send the gradients after fetching the gradients
				
				if layer.is_fc_layer:
					mapped_layer_idx=len(self.module)-1-layer_idx
					if len(req_send_check) != 0:
						# if this layer is the first layer to send gradient, then we don't need to wait for anything
						# else we need to check that the previous gradient has been sent
						req_send_check[-1].wait()
					req_isend = self.comm.Isend([grads, MPI.DOUBLE], dest=0, tag=TAG_LIST_[self.cur_step]+mapped_layer_idx)
					req_send_check.append(req_isend)
			# on the end of a certain iteration
			print('Worker: {}, Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {}, Time Cost: {}'.format(self.rank,
                   epoch_idx, batch_idx * X_batch.shape[0], X_batch.shape[0]*num_batch_per_epoch, 
                   (100. * (batch_idx * X_batch.shape[0]) / (X_batch.shape[0]*num_batch_per_epoch)), loss, time.time()-iter_start_time))

	def sync_fetch_step(self):
		'''fetch the first step from the parameter server'''
		self.next_step = self.comm.recv(source=0, tag=10)

	def async_fetch_step(self):
		req = self.comm.irecv(source=0, tag=10)
		self.next_step = req.wait()

	def test_fetch(self):
		'''a test function, used to test async weight sending and receiving'''
		weights = np.zeros((20, 10), dtype=np.float64)
		req=self.comm.Irecv([weights, MPI.DOUBLE], source=0, tag=11)
		req.wait()

	def async_fetch_weights(self):
		request_layers = []
		# keep in mind that this list saved the mapped layer index
		layers_to_update = []
		for layer_idx, layer in enumerate(self.module):
			if layer.is_fc_layer:
				# do a weird layer index map here:
				layer_map_idx = self.fc_layer_counts.index(layer_idx)
				# Check if we have already fetched the weights for this
	    		# particular step. If so, don't fetch it.
				if self._layer_cur_step[layer_map_idx] < self.cur_step:
					layers_to_update.append(layer_map_idx)
					req = self.comm.Irecv([layer.recv_buf, MPI.DOUBLE], source=0, tag=11+layer_idx)
					request_layers.append(req)

		assert (len(layers_to_update) == len(request_layers))
		for req_idx, req_l in enumerate(request_layers):
			fc_layer_idx = self.fc_layer_counts[req_idx]
			req_l.wait() # got the weights
			weights = self.module[fc_layer_idx].recv_buf
			# we also need to update the layer cur step here:
			self._layer_cur_step[layers_to_update[req_idx]] = self.cur_step

			# assign fetched weights to weights of module containers
			assert (self.module[fc_layer_idx].W.shape == weights[0:self.module[fc_layer_idx].get_shape[0], :].shape)
			self.module[fc_layer_idx].W = weights[0:self.module[fc_layer_idx].get_shape[0], :]
			assert (self.module[fc_layer_idx].b.shape == weights[self.module[fc_layer_idx].get_shape[0], :].shape)
			self.module[fc_layer_idx].b = weights[self.module[fc_layer_idx].get_shape[0], :]

	def update_step(self):
		'''update local (global) step on worker'''
		changed = (self.cur_step != self.next_step)
		self.cur_step = self.next_step
		return changed

if __name__ == "__main__":
	# this is only a simple test case
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	world_size = comm.Get_size()
	worker_fc_nn = WorkerFC_NN(comm=comm, world_size=world_size, rank=rank)
	print("I am worker: {} in all {} workers".format(worker_fc_nn.rank, worker_fc_nn.world_size))