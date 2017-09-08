#!/usr/bin/env bash

export EFS_MOUNT="shared"

pdsh -R ssh -w deeplearning-worker[1-$DEEPLEARNING_WORKERS_COUNT] "cd ~; if [ ! -e /MXNet-mini-batch ]; then git clone --recursive https://github.com/chao1224/MXNet-mini-batch.git; fi"
pdsh -R ssh -w deeplearning-worker[1-$DEEPLEARNING_WORKERS_COUNT] "cd ~/MXNet-mini-batch; git fetch && git reset --hard origin/master"
# pdsh -R ssh -w deeplearning-worker[1-$DEEPLEARNING_WORKERS_COUNT] "sudo mount -t nfs4 fs-9a13c033.efs.us-west-2.amazonaws.com:/ $EFS_MOUNT"
