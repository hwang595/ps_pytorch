# pytorch_distributed_nn
implement distributed neural network and straggler kill in Pytorch

# AWS EC2 AMI for distributed Pythorch:
`ami-ebc83c93` (with pytorch/pytorch vision installed and configured with CUDA 8.0 and cuDNN 7)

# Enable `MPI` backend in pytorch distributed
The frist thing you need to to is to remove your current version of `pytorch`, and build it from source (we may not need this later, but for now they're not enable `MPI` automatically in the binary source).

To build pytorch from source, you can follow guidence here (https://github.com/pytorch/pytorch#from-source). But there are a few things you should be careful about.

1. make sure you're in your `conda env` when you run `python setup.py install` by
```
source /home/user_name/anaconda[2 or 3]/bin/activate ~/anaconda[2 or 3]
```
otherwise, pytorch will be built in your system lib directory rather than conda lib directory.

2. make sure you use CUDA (version >= 7.5), and have cuDNN (version >= 7.0) installed. I have a quick and easy way to do this in this github repo (https://github.com/hwang595/distributed-MXNet). I'm not sure why, even the version specified here is `CUDA 7.5`, but this method will still install `CUDA 8.0` for you. But this dosen't matter.

To make sure if your `MPI` is enabled, just run:
```
import torch
torch.distributed.init_process_group(backend='mpi')
```
