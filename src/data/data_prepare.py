"""
Since we need to initialize the dataset in parallel, pre-download data is necessary
This script will do the job for you
"""
import torch
from torchvision import datasets, transforms


if __name__ == "__main__":
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
    # load training and test set here:
    training_set = datasets.CIFAR100(root='./cifar100_data', train=True,
                                            download=True, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
               ]))
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=128,
                                              shuffle=True)
    testset = datasets.CIFAR100(root='./cifar100_data', train=False,
                                           download=True, transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
               ]))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                             shuffle=False)