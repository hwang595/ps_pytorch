# pytorch_distributed_nn
implement distributed neural network and straggler kill in Pytorch

# AWS EC2 AMI for distributed Pythorch:
`ami-ebc83c93` (with pytorch/pytorch vision installed and configured with CUDA 7.5 and cuDNN 5)

# Enable `MPI` backend in pytorch distributed
The frist thing you need to to is to remove your current version of `pytorch`, and build it from source (we may not need this later, but for now they're not enable `MPI` automatically in the binary source).

To build pytorch from source, you can follow guidence here (https://github.com/pytorch/pytorch#from-source). But there are a few things you should be careful about.
