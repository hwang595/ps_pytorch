from model_ops.lenet import LeNet, LeNetSplit
from model_ops.resnet import *
from model_ops.resnet_split import *

def build_model(model_name):
    # build network
    if model_name == "LeNet":
        return LeNet()
    elif model_name == "ResNet18":
        return ResNet18()
    elif model_name == "ResNet34":
        return ResNet34()
    elif model_name == "ResNet50":
        return ResNet50()