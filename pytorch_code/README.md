do remeber to transfer the `numpy.ndarray` to `numpy.float64` before push them to MPI send/receive buffer, otherwise there will be some data type transfer issues between numpy and MPI.

# To use it on single machine:
```
python single_machine.py --dataset=MNIST/Cifar10 --network=LeNet/Resnet --batch-size=${BATCH_SIZE}
```

# To use it on distributed cluster:
```
mpirun -n ${NUM_WORKERS} --hostfile=${HOST_DIR} python distributed_nn.py --dataset=MNIST/Cifar10 --network=LeNet/Resnet --batch-size=${BATCH_SIZE}
```

# Run the whole thing automatically
The first thing you need do is to launch AWS EC2 instances, you can do that using `tools/pytorch_ec2.py` by running the following command:
```
python pytorch_ec2.py launch
```
After the launch command are executed and all instances are initialized (this may cost several minutes), you need to fetch the host addresses:
```
python pytorch_ec2.py get_hosts
```
Then, copying essential configuration files and hosts files using the public address of master node (the first address in the `host` file):
```
sh local_script.sh ${MASTER_PUB_ADDR}
```
After that, launch to master node manually, running the remote script under `$HOME` dir:
```
sh remote_script.sh
```
This script will do the cluster setup and data preparation works for you.
