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

    if rank == 0:
        master_fc_nn = SyncReplicasMaster_NN(comm=comm, world_size=world_size, num_grad_to_collect=world_size)
        print("I am the master: the world size is {}, cur step: {}".format(master_fc_nn.world_size, master_fc_nn.cur_step))
        master_fc_nn.train(training_set=None, validation_set=None)
    else:
        worker_fc_nn = WorkerFC_NN(comm=comm, world_size=world_size, rank=rank)
        print("I am worker: {} in all {} workers, next step: {}".format(worker_fc_nn.rank, worker_fc_nn.world_size-1, worker_fc_nn.next_step))
        worker_fc_nn.train(training_set=None, validation_set=None)
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


    # Create the minibatches
    batch_size = 128  # Approximately 25 samples per batch
    nb_of_batches = X_train.shape[0] / batch_size  # Number of batches
    # Create batches (X,Y) from the training set

    # TODO: this method can't permit each batch has the same length
    XT_batches = zip(
        np.array_split(X_train, nb_of_batches, axis=0),  # X samples
        np.array_split(T_train, nb_of_batches, axis=0))  # Y targets


    # Perform backpropagation
    # initalize some lists to store the cost for future analysis        
    minibatch_costs = []
    training_costs = []
    validation_costs = []

    max_epochs = 100  # Train for a maximum of 300 iterations
    learning_rate = 0.001  # Gradient descent learning rate

    # Train for the maximum number of iterations
    for iteration in range(max_epochs):
        for batch_idx, (X, T) in enumerate(XT_batches):  # For each minibatch sub-iteration
            activations = forward_step(X, layers)  # Get the activations
            minibatch_cost = layers[-1].get_cost(activations[-1], T)  # Get cost
            minibatch_costs.append(minibatch_cost)
            param_grads = backward_step(activations, T, layers)  # Get the gradients
            update_params(layers, param_grads, learning_rate)  # Update the parameters

            print('Train Epoch: {} [{}/{} ({:.0f}%)], Train_loss: {}'.format(
                iteration, batch_idx * X.shape[0], X_train.shape[0], (100. * (batch_idx * X.shape[0]) / X_train.shape[0]), np.mean(minibatch_costs)))
        # Get full training cost for future analysis (plots)
        activations = forward_step(X_train, layers)
        train_cost = layers[-1].get_cost(activations[-1], T_train)
        training_costs.append(train_cost)
        # Get full validation cost
        activations = forward_step(X_validation, layers)
        validation_cost = layers[-1].get_cost(activations[-1], T_validation)
        validation_costs.append(validation_cost)

        # perform early stop here:
        if len(validation_costs) > 3:
            # Stop training if the cost on the validation set doesn't decrease
            #  for 3 iterations
            if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3]:
                break
        if iteration > 10:
            break
    nb_of_iterations = iteration + 1  # The number of iterations that have been executed

    '''
    # Plot the minibatch, full training set, and validation costs
    minibatch_x_inds = np.linspace(0, nb_of_iterations, num=nb_of_iterations*nb_of_batches)
    iteration_x_inds = np.linspace(1, nb_of_iterations, num=nb_of_iterations)
    # Plot the cost over the iterations
    plt.plot(minibatch_x_inds, minibatch_costs, 'k-', linewidth=0.5, label='cost minibatches')
    plt.plot(iteration_x_inds, training_costs, 'r-', linewidth=2, label='cost full training set')
    plt.plot(iteration_x_inds, validation_costs, 'b-', linewidth=3, label='cost validation set')
    # Add labels to the plot
    plt.xlabel('iteration')
    plt.ylabel('$\\xi$', fontsize=15)
    plt.title('Decrease of cost over backprop iteration')
    plt.legend()
    x1,x2,y1,y2 = plt.axis()
    plt.axis((0,nb_of_iterations,0,2.5))
    plt.grid()
    plt.show()
    '''