FLAGS=-std=c++11 -Ofast
LIBS=-I/usr/local/opt/openblas/include/ -lblas  -lpthread
CC=g++
MPICC=mpic++

single_machine:
	$(CC) $(FLAGS) src/single_machine_nn.cpp $(LIBS) -o single_machine_nn

single_machine_run:
	rm -f single_machine
	make single_machine
	./single_machine_nn

distributed:
	$(MPICC) $(FLAGS) src/distributed_nn.cpp $(LIBS) -o distributed_nn

distributed_run:
	make distributed
	sudo mpirun -n 8 --allow-run-as-root  ./distributed_nn
