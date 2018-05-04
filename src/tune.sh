tune_dir=~/ps_pytorch/src/tune/
max_tuning_step=100
mkdir ${tune_dir}

echo "Start parameter tuning ..."
for lr in 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5
do
  echo "Trial running for learning rate: ${lr}"
  mpirun -n 17 --hostfile hosts_address \
  python distributed_nn.py \
  --lr=${lr} \
  --momentum=0.9 \
  --network=ResNet18 \
  --dataset=Cifar10 \
  --batch-size=8 \
  --test-batch-size=200 \
  --comm-type=Bcast \
  --num-aggregate=16 \
  --eval-freq=200 \
  --epochs=10 \
  --max-steps=${max_tuning_step} \
  --enable-gpu= \
  --train-dir=/home/ubuntu/ > ${tune_dir}lr_${lr}

  cat ${tune_dir}lr_${lr} | grep Step:\ ${max_tuning_step} > ${tune_dir}lr_${lr}_processing
  bash ~/killall.sh
done

for lr in 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5
do
  echo "Logging out tunning results"
  python tiny_tuning_parser.py \
  --tuning-dir=${tune_dir}lr_${lr}_processing \
  --tuning-lr=${lr} \
  --num-workers=16
done
