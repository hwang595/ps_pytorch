# configure, download, and install OpenMPI
sudo apt-get update
sudo apt-get -y install gcc g++ make
sudo apt-get update
wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.1.tar.gz
tar -xvf openmpi-*
rm -f openmpi-3.0.1.tar.gz
cd ~/openmpi-3.0.1
./configure --prefix="/home/$USER/.openmpi"
make
sudo make install
export PATH="$PATH:/home/$USER/.openmpi/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/.openmpi/lib/"