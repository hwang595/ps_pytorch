import torch
from torch import nn
import torch.nn.functional as F

# we use LeNet here for our simple case
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.ceriation = nn.CrossEntropyLoss()
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.fc2(x)
        #loss = self.ceriation(x, target)
        return x
    def name(self):
        return 'lenet'

class LeNetSplit(nn.Module):
    '''
    this is a module that we split the module and do backward process layer by layer
    please don't call this module for normal uses, this is a hack and run slower than
    the automatic chain rule version
    '''
    def __init__(self):
        super(LeNetSplit, self).__init__()
        self.layers0 = nn.ModuleList([
            nn.Conv2d(1, 20, 5, 1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(20, 50, 5, 1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            ])
        self.layers1 = nn.ModuleList([
            nn.Linear(4*4*50, 500),
            nn.Linear(500, 10),
            ])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        self.output = []
        self.input = []
        for layer in self.layers0:
            # detach from previous history
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # compute output
            x = layer(x)
            # add to list of outputs
            self.output.append(x)
        x = x.view(-1, 4*4*50)
        for layer in self.layers1:
            # detach from previous history
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # compute output
            x = layer(x)
            # add to list of outputs
            self.output.append(x)
        return x

    def backward(self, g):
        for i, output in reversed(list(enumerate(self.output))):
            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
            else:
                output.backward(self.input[i+1].grad.data)