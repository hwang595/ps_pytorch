from __future__ import print_function
from mpi4py import MPI

from nn.nn import *

STEP_START_ = 1

class SyncReplicasMaster_NN(FC_NN):
	def __init__(self, comm, world_size, num_grad_to_collect):
		'''master node here, no rank needed since the rank will always be 0 for master node'''
		self.comm = comm   # get MPI communicator object
		self.world_size = world_size # total number of processes
		self.num_grad_to_collect = num_grad_to_collect # how many grads we want to collect in a certain iteration
		self.cur_step = STEP_START_

	def train(self, training_set, validation_set):
		# the first step we need to do here is to sync fetch the inital worl_step from the parameter server
		# we still need to make sure the value we fetched from parameter server is 1
		self.async_bcast_step()

	def async_bcast_step(self):
		req_list = []
		for i in range(self.world_size):
			req_list.append(self.comm.isend(self.cur_step, dest=i, tag=11))
		for i in range(len(req_list)):
			req_list[i].wait()