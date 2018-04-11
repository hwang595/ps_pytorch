#!/usr/bin/env bash
# setup Anaconda env
wget https://repo.continuum.io/archive/Anaconda2-5.1.0-Linux-x86_64.sh
bash Anaconda2-5.1.0-Linux-x86_64.sh -b -p ~/anaconda
rm Anaconda2-5.1.0-Linux-x86_64.sh
echo 'export PATH="~/anaconda/bin:$PATH"' >> ~/.bashrc 

# Refresh basically
source .bashrc

conda update conda

conda update -y -n base conda
conda install pytorch torchvision -y -c pytorch
conda install -y -c anaconda python-blosc
conda install -y -c conda-forge mpi4py