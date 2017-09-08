#!/usr/bin/env bash

# append hosts
sudo bash -c "cat hosts >> /etc/hosts"
cp config ~/.ssh/

export DEEPLEARNING_WORKERS_COUNT=`wc -l < hosts`

git config --global user.name hwang595
git config --global user.email hongyiwang@cs.wisc.edu

sudo apt-get update
sudo apt-get install pdsh -y
pdsh -R ssh -w deeplearning-worker[1-$DEEPLEARNING_WORKERS_COUNT] "sudo apt-get update; sudo apt-get install pdsh -y"
