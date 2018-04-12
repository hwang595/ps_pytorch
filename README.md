# ps_pytorch
implement [parameter server](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf) with PyTorch and OpenMPI
## Contents

1. [Motivations](#motivations)
2. [System design](#system-design)
3. [Basic usages](#basic-usages)
4. [How to prepare datasets](#prepare-datasets)

## Motivations:
1. PyTorch provides easy-to-use APIs with dynamic computational graph
2. PyTorch dose not offer full distributed packages (there is some comm lib but not support operations with same flexiblity as OpenMPI)
3. [mpi4py](https://github.com/mpi4py/mpi4py) is a good Python bindings for any distributions of MPI (e.g. OpenMPI, MPICH, and etc)

## System Design:
1. parameter server node: This node serves both as master and parameter server in our system, i.e. it synchronize all workers to enter next iteration by broadcast global step to workers and also store the global model, which are keeping fetched by worker nodes at beginning of one iteration. For a user defined frequency, parameter server node will save the current model as checkpoint to shared file system (NFS in our system) for model evaluation.
2. workers mainly aim at sample data points (or mini-batch) in from local dataset (we don't pass data among nodes to maintain data locality), computing gradients, and send them back to parameter server.
3. evaluator read the checkpoints from the shared directory, and do model evaluation. Note that: there is only testset data saved on evaluator nodes.
4. gradient compression is implemented using high-speed compression tool [Blosc](https://github.com/Blosc/c-blosc) to mitigate communication overhead

![alt text](https://github.com/hwang595/ps_pytorch/blob/master/images/system_overview.jpg)

## Basic Usages
### Single Machine:
```
python single_machine.py --dataset=MNIST/Cifar10 --network=LeNet/Resnet --batch-size=${BATCH_SIZE}
```
### Cluster Setup:
#### Launching Instances:
The first thing you need do is to launch AWS EC2 instances. [This script](https://github.com/hwang595/ps_pytorch/tree/master/tools) helps you to launch EC2 instances automatically, but before running this script, you should follow [the instruction](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) to setup AWS CLI on your local machine.
After that, please edit this part in `pytorch_ec2.py`
``` python
cfg = Cfg({
    "name" : "PS_PYTORCH",      # Unique name for this specific configuration
    "key_name": "NameOfKeyFile",          # Necessary to ssh into created instances
    # Cluster topology
    "n_masters" : 1,                      # Should always be 1
    "n_workers" : 8,
    "num_replicas_to_aggregate" : "8",
    "method" : "spot",
    # Region speficiation
    "region" : "us-west-2",
    "availability_zone" : "us-west-2b",
    # Machine type - instance type configuration.
    "master_type" : "m4.2xlarge",
    "worker_type" : "m4.2xlarge",
    # please only use this AMI for pytorch
    "image_id": "ami-xxxxxxxx",
    # Launch specifications
    "spot_price" : "0.15",                 # Has to be a string
    # SSH configuration
    "ssh_username" : "ubuntu",            # For sshing. E.G: ssh ssh_username@hostname
    "path_to_keyfile" : "/dir/to/NameOfKeyFile.pem",

    # NFS configuration
    # To set up these values, go to Services > ElasticFileSystem > Create new filesystem, and follow the directions.
    #"nfs_ip_address" : "172.31.3.173",         # us-west-2c
    #"nfs_ip_address" : "172.31.35.0",          # us-west-2a
    "nfs_ip_address" : "172.31.14.225",          # us-west-2b
    "nfs_mount_point" : "/home/ubuntu/shared",       # NFS base dir
```
Then, launch EC2 instances by running
```
python ./tools/pytorch_ec2.py launch
```
After all launched instances are ready (this may take a while), getting private ips of instances by
```
python ./tools/pytorch_ec2.py get_hosts
```
this will write ips into a file named `hosts_address`, which looks like
```
172.31.16.226 (${PS_IP})
172.31.27.245
172.31.29.131
172.31.18.108
172.31.18.174
172.31.17.228
172.31.16.25
172.31.30.61
172.31.29.30
```
Running the following command will copy your keyfile to the parameter server (PS) and do some basic configurations
```
bash ./tool/local_script.sh ${PS_IP}
```
#### SSH related:
In PS setting, PS should be able to ssh to any compute node, [this part](https://github.com/hwang595/ps_pytorch/blob/master/tools/remote_script.sh#L8-L22) dose the job for you.

## Prepare Datasets
We currently support [MNIST](http://yann.lecun.com/exdb/mnist/) and [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) datasets. Download, split, and transform datasets by
```
bash ./src/data_prepare.sh
```

## future works:
(Please note that this project is still in early alpha version)
1. move APIs into PyTorch completely using its [built-in communication lib](http://pytorch.org/docs/master/distributed.html)
2. optimize the speedups and minize communication overhead
3. support async communication mode
4. wrap up more state-of-art deep models and dataset
