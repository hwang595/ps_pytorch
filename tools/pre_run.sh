#!/usr/bin/env bash
conda update -y -n base conda
conda install pytorch torchvision -y -c pytorch
conda install -y -c anaconda python-blosc
conda install -y -c anaconda mpi4py