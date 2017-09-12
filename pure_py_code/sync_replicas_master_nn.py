from __future__ import print_function
import time

from mpi4py import MPI

from nn.nn import *

STEP_START_ = 1

MAX_NUM_ITERATIONS = 300

class GradientAccumulator(object):
	def __init__(self):
		pass

class SyncReplicasMaster_NN(FC_NN):
	def __init__(self, comm, world_size, num_grad_to_collect):
		'''master node here, no rank needed since the rank will always be 0 for master node'''
		self.comm = comm   # get MPI communicator object
		self.world_size = world_size # total number of processes
		self.num_grad_to_collect = num_grad_to_collect # how many grads we want to collect in a certain iteration
		self.cur_step = STEP_START_

	def build_model(self, num_layers, layer_config):
	    layers = []
	    for i in range(num_layers):
	        if layer_config['layer'+str(i)]['type'] == 'activation':
	            if layer_config['layer'+str(i)]['name'] == 'sigmoid':
	                layers.append(SigmoidLayer())
	            elif layer_config['layer'+str(i)]['name'] == 'softmax':
	                layers.append(SoftmaxOutputLayer())
	        elif layer_config['layer'+str(i)]['type'] == 'fc':
	            layers.append(LinearLayer(layer_config['layer'+str(i)]['shape'][0], layer_config['layer'+str(i)]['shape'][1]))
	    self.module = layers
	    return layers

	def train(self, training_set, validation_set):
		# the first step we need to do here is to sync fetch the inital worl_step from the parameter server
		# we still need to make sure the value we fetched from parameter server is 1
		# please note that step is start from one here
		self.async_bcast_step()

		# fake test here:
		for i in range(1, MAX_NUM_ITERATIONS):
			print("Master node is entering step: {}".format(i))
			self.async_bcast_step()
			self.async_bcast_layer_weights()
			#self.test_send()
			time.sleep(5)
			self.cur_step += 1


	def async_bcast_step(self):
		req_list = []
		for i in range(self.world_size):
			if i != 0:
				# TODO(hwang): try to generate specific tag for seperate step and weights
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
		for req_l in request_layers:
			for req_worker in req_l:
				req_worker.wait()