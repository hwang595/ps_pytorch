import numpy as np

# Define the non-linear functions used
def logistic(z): 
    return 1 / (1 + np.exp(-z))

def logistic_deriv(y):  # Derivative of logistic function
    return np.multiply(y, (1 - y))
    
def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)