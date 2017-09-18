# running distributed MPI (mpi4py) code:
firstly a `hosts` file need to be created, in format of

```
master_private_ip
worker1_private_ip
...
workern_private_ip
```
Then, we running the code, use the following command:

```
mpirun -n (num procs) --hostfile hosts(dir) python distributed_nn.py
```
