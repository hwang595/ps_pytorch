from __future__ import print_function
'''code to run on single machine'''

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
    # fetch dataset
    mnist_data = mnist.read_data_sets(train_dir='./data', reshape=True)
    '''
    layer_config = {'layer0':{'type':'fc', 'name':'fully_connected', 'shape':(mnist_data.train.images.shape[1], 50)},
                'layer1':{'type':'activation', 'name':'sigmoid'},
                'layer2':{'type':'fc', 'name':'fully_connected', 'shape':(50, 100)},
                'layer3':{'type':'activation', 'name':'sigmoid'},
                'layer4':{'type':'fc', 'name':'fully_connected', 'shape':(100, 40)},
                'layer5':{'type':'activation', 'name':'sigmoid'},
                'layer6':{'type':'fc', 'name':'fully_connected', 'shape':(40, mnist_data.train.labels.shape[1])},
                'layer7':{'type':'activation', 'name':'softmax'}}
    '''
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
    
    exit()