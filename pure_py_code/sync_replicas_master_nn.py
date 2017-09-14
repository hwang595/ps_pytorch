from __future__ import print_function
import time

from mpi4py import MPI

from nn.nn import *

STEP_START_ = 1

MAX_NUM_ITERATIONS = 300

class GradientAccumulator(object):
	'''a simple class to implement gradient aggregator like the `Conditional Accumulators` in tensorflow'''
	def __init__(self, module, num_worker, fc_layer_num):
		# we will update this counter dynamically during the training process
		# the length of this counter should be number of fc layers in the network
		self.gradient_aggregate_counter = [0] * fc_layer_num

		# we used list to contain gradients of layers
		self.gradient_aggregator = []

		# TODO(hwang): we do not need to allocate so many space here since we need to aggregate the gradient
		# into each slot
		for layer_idx, layer in enumerate(module):
			if layer.is_fc_layer:
				tmp_aggregator = []
				for worker_idx in range(num_worker):
					tmp_aggregator.append(np.zeros((layer.get_shape[0]+1, layer.get_shape[1])))
				# initialize the gradient aggragator
				self.gradient_aggregator.append(tmp_aggregator)
	def meset_grad_counter(self):
		self.gradient_aggregate_counter = [0 for _ in self.gradient_aggregate_counter]


class SyncReplicasMaster_NN(FC_NN):
	def __init__(self, comm, world_size, num_grad_to_collect):
		'''master node here, no rank needed since the rank will always be 0 for master node'''
		self.comm = comm   # get MPI communicator object
		self.world_size = world_size # total number of processes
		self.num_grad_to_collect = num_grad_to_collect # how many grads we want to collect in a certain iteration
		self.cur_step = STEP_START_
		self.fc_layer_counts = []

		self._num_grad_to_collect = self.world_size - 1

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
				layers.append(LinearLayer(layer_config['layer'+str(i)]['shape'][0], layer_config['layer'+str(i)]['shape'][1]))
				layers[-1].layer_index = i
		self.module = layers
		self.grad_accumulator = GradientAccumulator(module=self.module, num_worker=self.world_size-1, fc_layer_num=len(self.fc_layer_counts))
		return layers

	def train(self, training_set, validation_set):
		# the first step we need to do here is to sync fetch the inital worl_step from the parameter server
		# we still need to make sure the value we fetched from parameter server is 1
		# please note that step is start from one here
		self.async_bcast_step()

		# fake test here:
		for i in range(1, MAX_NUM_ITERATIONS):
			enough_gradients_received = False

			print("Master node is entering step: {}".format(i))
			self.async_bcast_step()
			self.async_bcast_layer_weights()
			
			# set the gradient fetch step and gather the request
			gradient_fetch_requests=self.async_fetch_gradient_start()

			# wait for enough gradients to be aggregated:
			while not enough_gradients_received:
				status = MPI.Status()
				MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)

				if status.tag-12 in self.fc_layer_counts:
					fc_index = self.fc_layer_counts.index(status.tag-12)
					self.grad_accumulator.gradient_aggregate_counter[fc_index] += 1

				print(self.grad_accumulator.gradient_aggregate_counter)
				print('-----------------------------------------------------------')
				
				enough_gradients_received = True
				for j in self.grad_accumulator.gradient_aggregate_counter:
					enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)

			self.grad_accumulator.meset_grad_counter()
			self.cur_step += 1


	def async_bcast_step(self):
		req_list = []
		for i in range(self.world_size):
			if i != 0:
				req_list.append(self.comm.isend(self.cur_step, dest=i, tag=10))
		for i in range(len(req_list)):
			req_list[i].wait()

	def test_send(self):
		buf = np.random.randn(20, 10)
		for i in range(self.world_size):
			if i != 0:
				self.comm.Isend([buf, MPI.DOUBLE], dest=i, tag=11)

	def async_bcast_layer_weights(self):
		request_layers = []
		for layer_idx, layer in enumerate(self.module):
			request_workers = []
			for i in range(self.world_size):
				if i != 0:
					if layer.is_fc_layer:
						req = self.comm.Isend([layer.fetch_wrapped_layer, MPI.DOUBLE], dest=i, tag=11+layer_idx)
						request_workers.append(req)
			request_layers.append(request_workers)
		# TODO(hwang): check to see if these `wait` calls are necessary here
		for req_l in request_layers:
			for req_worker in req_l:
				req_worker.wait()

	def async_fetch_gradient_start(self):
		'''make gradient fetch requests and return the request list'''
		gradient_fetch_requests = [] # `graident_fetch_request` should have length of #fc_layer*num_grad_to_collect
		for layer_idx, layer in enumerate(self.module):
			if layer.is_fc_layer:
				for k in range(self._num_grad_to_collect):
					i = self.fc_layer_counts.index(layer_idx)
					# xxx need to be some specific buffer here
					req = self.comm.Irecv([self.grad_accumulator.gradient_aggregator[i][k], MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=12+layer_idx)
					gradient_fetch_requests.append(req)
		return gradient_fetch_requests