"""
Since we need to initialize the dataset in parallel, pre-download data is necessary
This script will do the job for you
"""
import torch
from torchvision import datasets, transforms
#for tmp solution
from datasets import MNISTDataset, Cifar10Dataset


if __name__ == "__main__":
	'''
    # download data to directory ./mnist_data
    mnist_data = mnist.read_data_sets(train_dir='./mnist_data', reshape=True)
    # download data to directory ./cifar10_data
    cifar10_data = cifar10.read_data_sets(padding_size=0, reshape=True)
    '''
	training_set_mnist = datasets.MNIST('./mnist_data', train=True, download=True,
	           transform=transforms.Compose([
	               transforms.ToTensor(),
	               transforms.Normalize((0.1307,), (0.3081,))]))
	train_loader_mnist = torch.utils.data.DataLoader(training_set_mnist, batch_size=128, shuffle=True)
	test_loader_mnist = torch.utils.data.DataLoader(
	    datasets.MNIST('./mnist_data', train=False, transform=transforms.Compose([
	               transforms.ToTensor(),
	               transforms.Normalize((0.1307,), (0.3081,))
	           ])), batch_size=100, shuffle=True)
	trainset_cifar10 = datasets.CIFAR10(root='./cifar10_data', train=True,
	                                        download=True, transform=transforms.Compose([
	               transforms.ToTensor(),
	               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	           ]))
	train_loader_cifar10 = torch.utils.data.DataLoader(trainset_cifar10, batch_size=128,
	                                          shuffle=True)
	test_loader_cifar10 = torch.utils.data.DataLoader(
	    datasets.CIFAR10('./cifar10_data', train=False, transform=transforms.Compose([
	               transforms.ToTensor(),  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	           ])), batch_size=100, shuffle=True)
