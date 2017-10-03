do remeber to transfer the `numpy.ndarray` to `numpy.float64` before push them to MPI send/receive buffer, otherwise there will be some data type transfer issues between numpy and MPI.

# To use it on single machine:
```
python single_machine.py --dataset=MNIST/Cifar10 --network=LeNet/Resnet --batch-size=${BATCH_SIZE}
```

# To use it on distributed cluster:
```
mpirun -n ${NUM_WORKERS} --hostfile=${HOST_DIR} python distributed_nn.py --dataset=MNIST/Cifar10 --network=LeNet/Resnet --batch-size=${BATCH_SIZE}
```
I will write script to make the whole process automated, stay tuned.
