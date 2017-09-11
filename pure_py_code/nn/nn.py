import itertools
import collections
import time

import numpy as np
from nn_layer import Layer, LinearLayer, SigmoidLayer, SoftmaxOutputLayer

NUM_LAYERS_SKIP = -1


# Define the forward propagation step as a method.
def forward_step(input_samples, layers):
    """
    Compute and return the forward activation of each layer in layers.
    Input:
        input_samples: A matrix of input samples (each row is an input vector)
        layers: A list of Layers
    Output:
        A list of activations where the activation at each index i+1 corresponds to
        the activation of layer i in layers. activations[0] contains the input samples.  
    """
    activations = [input_samples] # List of layer activations
    # Compute the forward activations for each layer starting from the first
    X = input_samples
    for layer in layers:
        Y = layer.get_output(X)  # Get the output of the current layer
        activations.append(Y)  # Store the output for future processing
        X = activations[-1]  # Set the current input as the activations of the previous layer
    return activations  # Return the activations of each layer


# Define the backward propagation step as a method
def backward_step(activations, targets, layers):
    """
    Perform the backpropagation step over all the layers and return the parameter gradients.
    Input:
        activations: A list of forward step activations where the activation at 
            each index i+1 corresponds to the activation of layer i in layers. 
            activations[0] contains the input samples. 
        targets: The output targets of the output layer.
        layers: A list of Layers corresponding that generated the outputs in activations.
    Output:
        A list of parameter gradients where the gradients at each index corresponds to
        the parameters gradients of the layer at the same index in layers. 
    """
    param_grads = collections.deque()  # List of parameter gradients for each layer
    output_grad = None  # The error gradient at the output of the current layer
    # Propagate the error backwards through all the layers.
    #  Use reversed to iterate backwards over the list of layers.
    for i, layer in enumerate(reversed(layers)):
        cur_layer_idx = len(layers) - i - 1
        if cur_layer_idx <= NUM_LAYERS_SKIP:
            # implement short circuit here
            if layer.is_fc_layer:
                grads = [0.0 for _ in range(layer.W.shape[0]*layer.W.shape[1]+layer.W.shape[1])]
        else:
            # normal gradient computation     
            Y = activations.pop()  # Get the activations of the last layer on the stack
            # Compute the error at the output layer.
            # The output layer error is calculated different then hidden layer error.
            if output_grad is None:
                input_grad = layer.get_input_grad(Y, targets)
            else:  # output_grad is not None (layer is not output layer)
                input_grad = layer.get_input_grad(Y, output_grad)
            # Get the input of this layer (activations of the previous layer)
            X = activations[-1]
            # Compute the layer parameter gradients used to update the parameters
            grads = layer.get_params_grad(X, output_grad)
        param_grads.appendleft(grads)
        # Compute gradient at output of previous layer (input of current layer):
        output_grad = input_grad
    return list(param_grads)  # Return the parameter gradients


# Define a method to update the parameters
def update_params(layers, param_grads, learning_rate):
    """
    Function to update the parameters of the given layers with the given gradients
    by gradient descent with the given learning rate.
    """
    for layer, layer_backprop_grads in zip(layers, param_grads):
        for param, grad in itertools.izip(layer.get_params_iter(), layer_backprop_grads):
            # The parameter returned by the iterator point to the memory space of
            #  the original layer and can thus be modified inplace.
            param -= learning_rate * grad  # Update each parameter


class FC_NN(object):
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.lr = kwargs['learning_rate']
        self.max_epochs = kwargs['max_epochs']

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

    def print_model(self):
        '''
        this function is used for printing out the configuration of the model to chek
        please make sure you always call this function after building the model
        '''
        print("Start showing conifguration of fc network:")
        print
        for i, layer in enumerate(self.module):
            if layer.is_fc_layer:
                print('{}th layer: {}, shape: {}*{}'.format(i, layer.get_name, layer.get_shape[0], layer.get_shape[1]))
            else:
                print('{}th layer: {}'.format(i, layer.get_name))
        print

    def train(self, training_set, validation_set, debug=False):
        num_batch_per_epoch = training_set.images.shape[0] / self.batch_size
        # start training process
        for i in range(self.max_epochs):
            epoch_start_time = time.time()
            avg_loss = 0
            for batch_idx in range(num_batch_per_epoch):
                iter_start_time = time.time()
                tmp_time_3 = time.time()
                X_batch, y_batch = training_set.next_batch(batch_size=self.batch_size)
                duration_get_data = time.time() - tmp_time_3

                tmp_time_0 = time.time()
                logits = forward_step(X_batch, self.module)  # Get the activations
                duration_forward = time.time()-tmp_time_0
                
                tmp_time_1 = time.time()
                minibatch_cost = self.module[-1].get_cost(logits[-1], y_batch)  # Get cost
                duration_getting_cost = time.time()-tmp_time_1

                time_tmp_2 = time.time()
                param_grads = backward_step(logits, y_batch, self.module)  # Get the gradients
                duration_backward = time.time()-time_tmp_2

                tmp_time_4 = time.time()
                update_params(self.module, param_grads, self.lr)  # Update the parameters
                duration_update_model = time.time() - tmp_time_4

                avg_loss += minibatch_cost

                if debug:
                    print("fetch data duration: {}".format(duration_get_data))
                    print
                    print("forward duration: {}".format(duration_forward))
                    print
                    print("get cost duration: {}".format(duration_getting_cost))
                    print
                    print("backward duration: {}".format(duration_backward))
                    print
                    print("update duration: {}".format(duration_update_model))
                    print
                # print something to check if the model is converging
                print('Train Epoch: {} [{}/{} ({:.0f}%)], Train Loss: {}, Time Cost: {}'.format(
                    i, batch_idx * X_batch.shape[0], X_batch.shape[0]*num_batch_per_epoch, 
                    (100. * (batch_idx * X_batch.shape[0]) / (X_batch.shape[0]*num_batch_per_epoch)), minibatch_cost, time.time()-iter_start_time))
            # on the end of one epoch we do some test here:
            print("Average loss for epoch {}: {}\tTime cost: {}".format(i, avg_loss/num_batch_per_epoch, time.time()-epoch_start_time))