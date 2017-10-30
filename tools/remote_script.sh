KEY_PEM_NAME=HongyiScript.pem
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`
cd ~/pytorch_distributed_nn/pytorch_code/
python data_prepare.py
cd ~

sudo bash -c "cat hosts >> /etc/hosts"
cp config ~/.ssh/

cd ~/.ssh
eval `ssh-agent -s`
ssh-add ${KEY_PEM_NAME}
ssh-keygen -t rsa -b 4096 -C "hongyiwang.hdu@gmail.com"

for i in $(seq 2 $DEEPLEARNING_WORKERS_COUNT);
  do
  scp -i ${KEY_PEM_NAME} id_rsa.pub deeplearning-worker${i}:~/.ssh
  ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'git clone https://github.com/hwang595/pytorch_distributed_nn.git; cd ~/.ssh; cat id_rsa.pub >> authorized_keys'
  scp -i ${KEY_PEM_NAME} -r ~/pytorch_distributed_nn/pytorch_code/mnist_data deeplearning-worker${i}:~/pytorch_distributed_nn/pytorch_code/
  scp -i ${KEY_PEM_NAME} -r ~/pytorch_distributed_nn/pytorch_code/cifar10_data deeplearning-worker${i}:~/pytorch_distributed_nn/pytorch_code/
  echo "Done writing public key to worker: deeplearning-worker${i}"
 done