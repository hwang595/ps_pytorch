from __future__ import print_function
import time
import copy

from mpi4py import MPI
import numpy as np

from nn_ops import NN_Trainer

from model_ops.lenet import LeNet, LeNetSplit
from model_ops.resnet import *
from model_ops.resnet_split import *

import torch

STEP_START_ = 1

#MAX_NUM_ITERATIONS = 1000000
MAX_NUM_ITERATIONS = 100000


def update_params_dist_version(param, avg_grad, learning_rate):
	'''
	update the network layer by layer
	'''
	assert param.shape == avg_grad.shape
	param -= learning_rate * avg_grad
	return param

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

class GradientAccumulator(object):
	'''a simple class to implement gradient aggregator like the `Conditional Accumulators` in tensorflow'''
	def __init__(self, module, num_worker):
		# we will update this counter dynamically during the training process
		# the length of this counter should be number of fc layers in the network
		
		'''
		model_length = len(module.state_dict())
		self.gradient_aggregate_counter = [0] * model_length
		self.model_index_range = [i for i in range(model_length)]
		'''

		# we used list to contain gradients of layers
		self.gradient_aggregate_counter = []
		self.model_index_range = []
		self.gradient_aggregator = []


		# TODO(hwang): we do not need to allocate so many space here since we need
		# to aggregate the gradient into each slot
		# for similar reason to worker side, we temporarily change this to the version without batch norm
		'''
		for param_idx,(key_name, param) in enumerate(module.state_dict().items()):
			tmp_aggregator = []
			for worker_idx in range(num_worker):
				tmp_aggregator.append(np.zeros((param.size())))
			# initialize the gradient aggragator
			self.gradient_aggregator.append(tmp_aggregator)
		'''
		for param_idx, param in enumerate(module.parameters()):
			tmp_aggregator = []
			for worker_idx in range(num_worker):
				tmp_aggregator.append(np.zeros((param.size())))
			# initialize the gradient aggragator
			self.gradient_aggregator.append(tmp_aggregator)
			self.gradient_aggregate_counter.append(0)
			self.model_index_range.append(param_idx)

	def meset_everything(self):
		self._meset_grad_counter()
		self._meset_grad_aggregator()

	def _meset_grad_counter(self):
		self.gradient_aggregate_counter = [0 for _ in self.gradient_aggregate_counter]

	def _meset_grad_aggregator(self):
		'''reset the buffers in grad accumulator, not sure if this is necessary'''
		for i, tmp_aggregator in enumerate(self.gradient_aggregator):
			for j, buf in enumerate(tmp_aggregator):
				self.gradient_aggregator[i][j] = np.zeros(self.gradient_aggregator[i][j].shape)

class SyncReplicasMasterNormal_NN(NN_Trainer):
	def __init__(self, comm, **kwargs):
		'''master node here, no rank needed since the rank will always be 0 for master node'''
		self.comm = comm   # get MPI communicator object
		self.world_size = comm.Get_size() # total number of processes
		self.cur_step = STEP_START_
		self.lr = kwargs['learning_rate']
		self.momentum = kwargs['momentum']
		self.network_config = kwargs['network']
		self.comm_type = kwargs['comm_method']
		self._timeout_threshold = kwargs['timeout_threshold']

		self._num_grad_to_collect = self.world_size - 1
		# used to aggregate tmp gradients, the length is the same as # of fc layer 
		self._grad_aggregate_buffer = []
		self._model_shapes = []
		self._first_grad_received = False
		self._eval_freq = kwargs['eval_freq']
		self._train_dir = kwargs['train_dir']
		self._expected_grad_to_recv = kwargs['kill_threshold']
		

		############ will be deprecated soon #############################
		self._eval_batch_size = 1000

	def build_model(self):
		# build network
		if self.network_config == "LeNet":
			self.network=LeNetSplit()
		elif self.network_config == "ResNet18":
			self.network=ResNetSplit18(self._timeout_threshold)
		elif self.network_config == "ResNet34":
			self.network=ResNetSplit34()
		  
		# TODO(hwang): make sure this is useful
		self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.momentum)
		# assign a gradient accumulator to collect gradients from workers
		self.grad_accumulator = GradientAccumulator(module=self.network, num_worker=self.world_size-1)
		self.init_model_shapes()

	def train(self):
		# the first step we need to do here is to sync fetch the inital worl_step from the parameter server
		# we still need to make sure the value we fetched from parameter server is 1
		# please note that step is start from one here
		self.async_bcast_step()

		# fake test here:
		for i in range(1, MAX_NUM_ITERATIONS):
			# switch back to training mode
			self.network.train()
			self._first_grad_received = False
			enough_gradients_received = False

			print("Master node is entering step: {}".format(i))

			self.async_bcast_step()

			if self.comm_type == "Bcast":
				self.async_bcast_layer_weights_bcast()
			elif self.comm_type == "Async":
				self.async_bcast_layer_weights_async()
			
			# set the gradient fetch step and gather the request
			gradient_fetch_requests=self.async_fetch_gradient_start()

			# wait for enough gradients to be aggregated:
			while not enough_gradients_received:
				status = MPI.Status()
				MPI.Request.Waitany(requests=gradient_fetch_requests, status=status)

				if status.tag-88 in self.grad_accumulator.model_index_range:
					if not self._first_grad_received:
						self._first_grad_received=True
						grad_gather_start_time = time.time()

					layer_index = status.tag-88
					received_grad=self.grad_accumulator.gradient_aggregator[layer_index][status.source-1]
					
					# do gradient shape check here
					assert (received_grad.shape == self._model_shapes[layer_index])

					# aggregate the gradient
					############################ only for temp test ###################################
					if self.grad_accumulator.gradient_aggregate_counter[layer_index] <= self._expected_grad_to_recv:
						self.aggregate_gradient(gradient=received_grad, layer_idx=layer_index)
					#self.aggregate_gradient(gradient=received_grad, layer_idx=layer_index)

					self.grad_accumulator.gradient_aggregate_counter[layer_index] += 1
					
					#print(self.grad_accumulator.gradient_aggregate_counter)
					#print('---------------------------------------------------------------------')
				
				enough_gradients_received = True
				for j in self.grad_accumulator.gradient_aggregate_counter:
					enough_gradients_received = enough_gradients_received and (j >= self._num_grad_to_collect)

			grad_gather_duration = time.time()-grad_gather_start_time
			print("Master: gradient gather time: {:.4f}".format(grad_gather_duration))
			# average gradients and update the mode
			for i in range(len(self._grad_aggregate_buffer)):
				#self._grad_aggregate_buffer[i] /= self._num_grad_to_collect
				self._grad_aggregate_buffer[i] /= self._expected_grad_to_recv

			# update using SGD method
			tmp_module = []
			for param_idx, param in enumerate(self.network.parameters()):
				updated_model=update_params_dist_version(param=param.data.numpy(), avg_grad=self._grad_aggregate_buffer[param_idx], learning_rate=self.lr)
				tmp_module.append(updated_model)

			# update `state_dict` in pytorch modules
			print("Master start to update the model")
			self.model_update(tmp_module)

			# reset essential elements
			self.meset_grad_buffer()
			self.grad_accumulator.meset_everything()
			# save model for validation in a pre-specified frequency
			if self.cur_step%self._eval_freq == 0:
				self._save_model(file_path=self._generate_model_path())
			self.cur_step += 1

	def init_model_shapes(self):
		for param_idx, param in enumerate(self.network.parameters()):
			self._model_shapes.append(param.size())
			self._grad_aggregate_buffer.append(np.zeros(param.size()))

	def async_bcast_step(self):
		req_list = []
		for i in range(self.world_size):
			if i != 0:
				req_list.append(self.comm.isend(self.cur_step, dest=i, tag=10))
		for i in range(len(req_list)):
			req_list[i].wait()

	def async_bcast_layer_weights_async(self):
		request_layers = []
		for layer_idx, layer in enumerate(self.network.parameters()):
			request_workers = []
			layer_to_send = layer.data.numpy().astype(np.float64)
			for i in range(self.world_size):
				if i != 0:
					req = self.comm.Isend([layer_to_send, MPI.DOUBLE], dest=i, tag=11+layer_idx)
					request_workers.append(req)

			request_layers.append(request_workers)
		# TODO(hwang): check to see if these `wait` calls are necessary here
		for req_l in request_layers:
			for req_worker in req_l:
				req_worker.wait()

	def async_bcast_layer_weights_bcast(self):
		request_layers = []
		for layer_idx, layer in enumerate(self.network.parameters()):
			request_workers = []
			layer_to_send = layer.data.numpy().astype(np.float64)
			#for i in range(self.world_size):
			#	if i != 0:
			# try to see if collective communication is better here:
			self.comm.Bcast([layer_to_send, MPI.DOUBLE], root=0)
			#request_layers.append(request_workers)
		# TODO(hwang): check to see if these `wait` calls are necessary here
		#for req_l in request_layers:
		#	for req_worker in req_l:
		#		req_worker.wait()


	def async_fetch_gradient_start(self):
		'''make gradient fetch requests and return the request list'''
		gradient_fetch_requests = [] # `graident_fetch_request` should have length of #fc_layer*num_grad_to_collect
		for layer_idx, layer in enumerate(self.network.parameters()):
			for k in range(self._num_grad_to_collect):
				#print(88+layer_idx, layer_idx)
				req = self.comm.Irecv([self.grad_accumulator.gradient_aggregator[layer_idx][k], MPI.DOUBLE], source=MPI.ANY_SOURCE, tag=88+layer_idx)
				#req = self.comm.Irecv([self.grad_accumulator.gradient_aggregator[layer_idx][k], MPI.FLOAT], source=MPI.ANY_SOURCE, tag=88+layer_idx)
				gradient_fetch_requests.append(req)
		return gradient_fetch_requests

	def aggregate_gradient(self, gradient, layer_idx):
		'''keep in mind the gradient here is wrapped gradient, which means it contains `W` and `b`'''
		self._grad_aggregate_buffer[layer_idx] += gradient

	def model_update(self, tmp_module):
		"""write model fetched from parameter server to local model"""
		new_state_dict = {}
		model_counter_ = 0
		for param_idx,(key_name, param) in enumerate(self.network.state_dict().items()):
			# handle the case that `running_mean` and `running_var` contained in `BatchNorm` layer
			if "running_mean" in key_name or "running_var" in key_name:
				tmp_dict = {key_name : param}
			else:
				assert param.size() == tmp_module[model_counter_].shape
				tmp_dict = {key_name: torch.from_numpy(tmp_module[model_counter_])}
				model_counter_+=1
			new_state_dict.update(tmp_dict)
		self.network.load_state_dict(new_state_dict)

	def meset_grad_buffer(self):
		for i in range(len(self._grad_aggregate_buffer)):
			self._grad_aggregate_buffer[i] = np.zeros(self._grad_aggregate_buffer[i].shape)

	def _generate_model_path(self):
		return self._train_dir+"model_step_"+str(self.cur_step)

	def _save_model(self, file_path):
		with open(file_path, "wb") as f_:
			torch.save(self.network, f_)
		return

	def _evaluate_model(self, validation_loader):
		self.network.eval()
		prec1_counter_ = prec5_counter_ = batch_counter_ = 0
		# which indicate an epoch based validation is done
		while validation_loader.dataset.epochs_completed <= self._epoch_counter:
			eval_image_batch, eval_label_batch = validation_loader.next_batch(batch_size=self._eval_batch_size)
			X_batch, y_batch = Variable(eval_image_batch.float()), Variable(eval_label_batch.long())
			output = self.network(X_batch)
			prec1_tmp, prec5_tmp = accuracy(output.data, eval_label_batch.long(), topk=(1, 5))
			prec1_counter_ += prec1_tmp
			prec5_counter_ += prec5_tmp
			batch_counter_ += 1
		prec1 = prec1_counter_ / batch_counter_
		prec5 = prec5_counter_ / batch_counter_
		self._epoch_counter = validation_loader.dataset.epochs_completed
		print('Testset Performance: Cur Step:{} Prec@1: {} Prec@5: {}'.format(self.cur_step, prec1.numpy()[0], prec5.numpy()[0]))