KEY_PEM_NAME=HongyiScript.pem
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`
cd ~/ps_pytorch/src/
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
  ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'git clone https://github.com/hwang595/ps_pytorch.git; cd ~/.ssh; cat id_rsa.pub >> authorized_keys'
  scp -i ${KEY_PEM_NAME} -r ~/ps_pytorch/src/mnist_data deeplearning-worker${i}:~/ps_pytorch/src/
  scp -i ${KEY_PEM_NAME} -r ~/ps_pytorch/src/cifar10_data deeplearning-worker${i}:~/ps_pytorch/src/
  echo "Done writing public key to worker: deeplearning-worker${i}"
 done