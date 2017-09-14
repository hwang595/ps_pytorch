from __future__ import print_function

import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
np.random.seed(seed=1)
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections

from mpi4py import MPI

from mnist import mnist
from nn.nn import FC_NN
from distributed_worker import *
from sync_replicas_master_nn import *


if __name__ == "__main__":
    # this is only a simple test case
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # fetch dataset
    mnist_data = mnist.read_data_sets(train_dir='./data', reshape=True)
    kwargs = {'batch_size':128, 'learning_rate':0.1, 'max_epochs':100}
    '''
    layer_config = {'layer0':{'type':'fc', 'name':'fully_connected', 'shape':(mnist_data.train.images.shape[1], 10)},
                'layer1':{'type':'activation', 'name':'sigmoid'},
                'layer2':{'type':'fc', 'name':'fully_connected', 'shape':(10, 20)},
                'layer3':{'type':'activation', 'name':'sigmoid'},
                'layer4':{'type':'fc', 'name':'fully_connected', 'shape':(20, 20)},
                'layer5':{'type':'activation', 'name':'sigmoid'},
                'layer6':{'type':'fc', 'name':'fully_connected', 'shape':(20, mnist_data.train.labels.shape[1])},
                'layer7':{'type':'activation', 'name':'softmax'}}
    '''
    layer_config = {'layer0':{'type':'fc', 'name':'fully_connected', 'shape':(mnist_data.train.images.shape[1], 500)},
                'layer1':{'type':'activation', 'name':'sigmoid'},
                'layer2':{'type':'fc', 'name':'fully_connected', 'shape':(500, 500)},
                'layer3':{'type':'activation', 'name':'sigmoid'},
                'layer4':{'type':'fc', 'name':'fully_connected', 'shape':(500, 800)},
                'layer5':{'type':'activation', 'name':'sigmoid'},
                'layer6':{'type':'fc', 'name':'fully_connected', 'shape':(800, 800)},
                'layer7':{'type':'activation', 'name':'sigmoid'},
                'layer8':{'type':'fc', 'name':'fully_connected', 'shape':(800, 200)},
                'layer9':{'type':'activation', 'name':'sigmoid'},
                'layer10':{'type':'fc', 'name':'fully_connected', 'shape':(200, 100)},
                'layer11':{'type':'activation', 'name':'sigmoid'},
                'layer12':{'type':'fc', 'name':'fully_connected', 'shape':(100, mnist_data.train.labels.shape[1])},
                'layer13':{'type':'activation', 'name':'softmax'}}

    if rank == 0:
        master_fc_nn = SyncReplicasMaster_NN(comm=comm, world_size=world_size, num_grad_to_collect=world_size)
        master_fc_nn.build_model(num_layers=len(layer_config), layer_config=layer_config)
        print("I am the master: the world size is {}, cur step: {}".format(master_fc_nn.world_size, master_fc_nn.cur_step))
        master_fc_nn.train(training_set=None, validation_set=None)
        print("Done sending messages to workers!")
    else:
        worker_fc_nn = WorkerFC_NN(comm=comm, world_size=world_size, rank=rank, **kwargs)
        worker_fc_nn.build_model(num_layers=len(layer_config), layer_config=layer_config)
        print("I am worker: {} in all {} workers, next step: {}".format(worker_fc_nn.rank, worker_fc_nn.world_size-1, worker_fc_nn.next_step))
        worker_fc_nn.train(training_set=mnist_data.train, validation_set=mnist_data.validation)
        print("Now the next step is: {}".format(worker_fc_nn.next_step))
    '''
    mnist_data = mnist.read_data_sets(train_dir='./data', reshape=True)

    kwargs = {'batch_size':128, 'learning_rate':0.1, 'max_epochs':100}
    layer_config = {'layer0':{'type':'fc', 'name':'fully_connected', 'shape':(mnist_data.train.images.shape[1], 500)},
                    'layer1':{'type':'activation', 'name':'sigmoid'},
                    'layer2':{'type':'fc', 'name':'fully_connected', 'shape':(500, 500)},
                    'layer3':{'type':'activation', 'name':'sigmoid'},
                    'layer4':{'type':'fc', 'name':'fully_connected', 'shape':(500, 800)},
                    'layer5':{'type':'activation', 'name':'sigmoid'},
                    'layer6':{'type':'fc', 'name':'fully_connected', 'shape':(800, 800)},
                    'layer7':{'type':'activation', 'name':'sigmoid'},
                    'layer8':{'type':'fc', 'name':'fully_connected', 'shape':(800, 200)},
                    'layer9':{'type':'activation', 'name':'sigmoid'},
                    'layer10':{'type':'fc', 'name':'fully_connected', 'shape':(200, 100)},
                    'layer11':{'type':'activation', 'name':'sigmoid'},
                    'layer12':{'type':'fc', 'name':'fully_connected', 'shape':(100, mnist_data.train.labels.shape[1])},
                    'layer13':{'type':'activation', 'name':'softmax'}}

    fc_nn = FC_NN(**kwargs)
    fc_nn.build_model(num_layers=len(layer_config), layer_config=layer_config)
    fc_nn.print_model()  
    fc_nn.train(training_set=mnist_data.train, validation_set=mnist_data.validation, debug=True)
    '''
    exit()