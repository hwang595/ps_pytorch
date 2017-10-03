"""
Since we need to initialize the dataset in parallel, pre-download data is necessary
This script will do the job for you
"""
import torch
#for tmp solution
from mnist import mnist
from datasets import MNISTDataset
from cifar10 import cifar10
from datasets import Cifar10Dataset


if __name__ == "__main__":
    # download data to directory ./mnist_data
    mnist_data = mnist.read_data_sets(train_dir='./mnist_data', reshape=True)
    # download data to directory ./cifar10_data
    cifar10_data = cifar10.read_data_sets(padding_size=0, reshape=True)
