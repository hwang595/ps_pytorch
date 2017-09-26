import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Function
import math
class _Linear(Function):

    # bias is an optional argument
    def forward(self, input, weight, bias=None):
        self.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    def backward(self, grad_output):
        input, weight, bias = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        print("backwarding......")
        if self.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if self.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias

def module_hook(module, grad_input, grad_out):
    print('module hook')
    print('grad_out', grad_out)

def variable_hook(grad):
    print('variable hook')
    print('grad', grad)
    return grad*.1

class Linear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.bias is None:
            return _Linear()(input, self.weight)
        else:
            return _Linear()(input, self.weight, self.bias)
linear = Linear(3,1)
linear.register_backward_hook(module_hook)
value = Variable(torch.FloatTensor([[1,2,3]]), requires_grad=True)

res = linear(value)
res.register_hook(variable_hook)

res.backward()