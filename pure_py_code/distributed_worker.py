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

	def train(self, training_set, validation_set):
		# the first step we need to do here is to sync fetch the inital worl_step from the parameter server
		# we still need to make sure the value we fetched from parameter server is 1
		self.sync_fetch_step()

	def sync_fetch_step(self):
		self.next_step = self.comm.recv(source=0, tag=11)

if __name__ == "__main__":
	# this is only a simple test case
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	world_size = comm.Get_size()
	worker_fc_nn = WorkerFC_NN(comm=comm, world_size=world_size, rank=rank)
	print("I am worker: {} in all {} workers".format(worker_fc_nn.rank, worker_fc_nn.world_size))