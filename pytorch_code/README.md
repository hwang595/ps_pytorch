do remeber to transfer the `numpy.ndarray` to `numpy.float64` before push them to MPI send/receive buffer, otherwise there will be some data type transfer issues between numpy and MPI.
