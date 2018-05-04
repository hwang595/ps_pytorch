KEY_PEM_NAME=~/.ssh/HongyiScript2.pem
export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`

sudo bash -c "cat hosts >> /etc/hosts"
cp config ~/.ssh/

for i in $(seq 2 $DEEPLEARNING_WORKERS_COUNT);
  do
  ssh -i ${KEY_PEM_NAME} deeplearning-worker${i} 'killall python'
 done