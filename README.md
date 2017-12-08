# ps_pytorch
implement parameter server (https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf) with PyTorch and OpenMPI

# motivations:
1. PyTorch provides easy-to-use APIs with dynamic computational graph
2. PyTorch dose not offer full distributed packages (there is some comm lib but not support operations with same flexiblity as OpenMPI)
3. mpi4py is a good Python wrapper for MPI
