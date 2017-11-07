mpirun -n 3 --hostfile hosts_address \
python distributed_nn.py \
--lr=0.1 \
--network=ResNet18 \
--dataset=Cifar10 \
--batch-size=1024 \
--comm-type=Bcast \
--num-aggregate=5 \
--mode=normal \
--kill-threshold=6.8 \
--eval-freq=20 \
--train-dir=/home/ubuntu/MPI_shared/