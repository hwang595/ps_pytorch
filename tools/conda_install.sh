#!/usr/bin/env bash
# setup Anaconda env
wget https://repo.continuum.io/archive/Anaconda2-5.1.0-Linux-x86_64.sh
bash Anaconda2-5.1.0-Linux-x86_64.sh -b -p ~/anaconda
rm Anaconda2-5.1.0-Linux-x86_64.sh
echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc 

# Refresh basically
source .bashrc