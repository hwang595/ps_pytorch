KEY_PEM_DIR=/home/hwang/My_Code/AWS/HongyiScript.pem
KEY_PEM_NAME=HongyiScript.pem
PUB_IP_ADDR="$1"
echo "Public address of master node: ${PUB_IP_ADDR}"

ssh -o "StrictHostKeyChecking no" ubuntu@${PUB_IP_ADDR}
scp -i ${KEY_PEM_DIR} ${KEY_PEM_DIR} ubuntu@${PUB_IP_ADDR}:~/.ssh
scp -i ${KEY_PEM_DIR} hosts hosts_address config ubuntu@${PUB_IP_ADDR}:~/
scp -i ${KEY_PEM_DIR} -r /home/hwang/My_Code/ps_pytorch ubuntu@${PUB_IP_ADDR}:~/
ssh -i ${KEY_PEM_DIR} ubuntu@${PUB_IP_ADDR} 'cp ps_pytorch/tools/remote_script.sh ~/; bash ~/ps_pytorch/tools/conda_install.sh; cp /home/ubuntu/grad_lossy_compression/tools/killall.sh ~/'
