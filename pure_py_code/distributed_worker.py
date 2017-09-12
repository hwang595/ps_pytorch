from __future__ import print_function
from mpi4py import MPI

from nn.nn import *

STEP_START_ = 1

class WorkerFC_NN(FC_NN):
	def __init__(self, comm, world_size, rank):
		self.comm = comm   # get MPI communicator object
		self.world_size = world_size # total number of processes
		self.rank = rank # rank of this Worker
		#self.status = MPI.Status()
		self.cur_step = 0
		self.next_step = 0 # we will fetch this one from parameter server
		self.fc_layer_counts = []

	def build_model(self, num_layers, layer_config):
		layers = []
		for i in range(num_layers):
		    if layer_config['layer'+str(i)]['type'] == 'activation':
		        if layer_config['layer'+str(i)]['name'] == 'sigmoid':
		            layers.append(SigmoidLayer())
		        elif layer_config['layer'+str(i)]['name'] == 'softmax':
		            layers.append(SoftmaxOutputLayer())
		    elif layer_config['layer'+str(i)]['type'] == 'fc':
		    	self.fc_layer_counts.append(i)
		        layers.append(LinearLayer(layer_config['layer'+str(i)]['shape'][0], layer_config['layer'+str(i)]['shape'][1], init_mode="default"))
		self.module = layers
		return layers

	def train(self, training_set, validation_set):
		# the first step we need to do here is to sync fetch the inital worl_step from the parameter server
		# we still need to make sure the value we fetched from parameter server is 1
		self.sync_fetch_step()
		# do some sync check here
		assert(self.update_step())
		assert(self.cur_step == STEP_START_)

		first = True

		print("Worker {}: starting training".format(self.rank))
		# start the training process
		while True:
			# the worker shouldn't know the current global step
			# except received the message from parameter server
			self.async_fetch_step()
			updated = self.update_step()

			if (not updated) and (not first):
				# wait here unitl enter next step
				continue

			first = False
			print("Rank of this node: {}, Current step: {}".format(self.rank, self.cur_step))
			self.async_fetch_weights()
			#self.test_fetch()


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
		for layer_idx, layer in enumerate(self.module):
			if layer.is_fc_layer:
				req = self.comm.Irecv([layer.recv_buf, MPI.DOUBLE], source=0, tag=11+layer_idx)
				request_layers.append(req)
		for req_idx, req_l in enumerate(request_layers):
			fc_layer_idx = self.fc_layer_counts[req_idx]
			req_l.wait()
			weights = self.module[fc_layer_idx].recv_buf
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