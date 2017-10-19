'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385


Please Note that, this version is a hack, it's super hacky and quite hacky never call this one for normal use
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlockSplit(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockSplit, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x, input_list, output_list):
        '''
        the input_list and output_list here is similar to input/output in ResNet class
        '''
        # we skip the detach and append operation on the very first x here 
        # since that's done outside of this function
        x = self.conv1(x)
        output_list.append(x)

        x = Variable(x.data, requires_grad=True)
        input_list.append(x)
        x = self.bn1(x)
        output_list.append(x)

        x = Variable(x.data, requires_grad=True)
        input_list.append(x)
        x = nn.ReLU(x)
        output_list.append(x)

        x = Variable(x.data, requires_grad=True)
        input_list.append(x)
        x = self.conv2(x)
        output_list.append(x)

        x = Variable(x.data, requires_grad=True)
        input_list.append(x)
        x = self.bn2(x)
        output_list.append(x)

        # TODO(hwang): figure out if this part also need hack
        x += self.shortcut(x)

        x = Variable(x.data, requires_grad=True)
        input_list.append(x)
        x = nn.ReLU(x)
        output_list.append(x)
        return x, input_list, output_list   
        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out, input_list, output_list
        '''


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # we skip the detach operation on the very first x here since that's done outside of this function
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetSplit(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetSplit, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        layers_split = nn.ModuleList(layers)
        #return nn.Sequential(*layers)
        return layers_split

    def forward(self, x):
        # use these containers to save intermediate variables
        self.output = []
        self.input = []

        # start the forward process right here implement the following logic to every intermediate var:
        # detach from previous history
        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.conv1(x)
        # add to list of outputs
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.bn1(x)
        self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = nn.ReLU(x)
        self.output.append(x)

        for layer in self.layer1:
            # each `layer` here is either a `BasicBlockSplit` or `BottleneckSplit`
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # call the `.forward()` func in `BasicBlockSplit` or `BottleneckSplit` here
            x, self.input, self.output = layer(x, self.input, self.output)
            self.output.append(x)

        for layer in self.layer1:
            # each `layer` here is either a `BasicBlockSplit` or `BottleneckSplit`
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # call the `.forward()` func in `BasicBlockSplit` or `BottleneckSplit` here
            x, self.input, self.output = layer(x, self.input, self.output)
            self.output.append(x)

        for layer in self.layer2:
            # each `layer` here is either a `BasicBlockSplit` or `BottleneckSplit`
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # call the `.forward()` func in `BasicBlockSplit` or `BottleneckSplit` here
            x, self.input, self.output = layer(x, self.input, self.output)
            self.output.append(x)

        for layer in self.layer3:
            # each `layer` here is either a `BasicBlockSplit` or `BottleneckSplit`
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # call the `.forward()` func in `BasicBlockSplit` or `BottleneckSplit` here
            x, self.input, self.output = layer(x, self.input, self.output)
            self.output.append(x)

        for layer in self.layer4:
            # each `layer` here is either a `BasicBlockSplit` or `BottleneckSplit`
            x = Variable(x.data, requires_grad=True)
            self.input.append(x)
            # call the `.forward()` func in `BasicBlockSplit` or `BottleneckSplit` here
            x, self.input, self.output = layer(x, self.input, self.output)
            self.output.append(x)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = nn.AvgPool2d(x)
        self.output.append(x)

        x = x.view(x.size(0), -1)

        x = Variable(x.data, requires_grad=True)
        self.input.append(x)
        x = self.linear(x)
        self.output.append(x)
        return x
        '''
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        '''
    def backward(self, g):
        for i, output in reversed(list(enumerate(self.output))):
            if i == (len(self.output) - 1):
                # for last node, use g
                output.backward(g)
            else:
                output.backward(self.input[i+1].grad.data)