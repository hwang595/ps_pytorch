#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include "distributed/distributed_defines.h"
#include "distributed/worker_nn.h"
#include "distributed/sync_replicas_master_nn.h"
#include "distributed/evaluator_nn.h"

int main(void) {
    srand(time(NULL));

    std::cout << std::fixed << std::showpoint;
    std::cout << std::setprecision(10);

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int n_procs;
    MPI_Comm_size(MPI_COMM_WORLD, &n_procs);

    // Get the rank of the process
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    char hostname[1024];
    gethostname(hostname, 1024);

    // Set NN Params
    NNParams *params = new NNParams();
    int batch_size = 128;
    params->SetBatchsize(batch_size);
    params->AddLayer(batch_size, IMAGE_X*IMAGE_Y);
    params->AddLayer(IMAGE_X*IMAGE_Y, 500);
    params->AddLayer(500, 500);
    params->AddLayer(500, 800);
    params->AddLayer(800, 800);
    params->AddLayer(800, 200);
    params->AddLayer(200, 100);
    params->AddLayer(100, 100);
    params->AddLayer(100, N_CLASSES);
    params->SetLearningRate(1e-3);

    // Load data
    int number_of_images, number_of_test_images, image_size;
    int number_of_labels, number_of_test_labels;
    uchar **images = read_mnist_images(TRAINING_IMAGES, number_of_images, image_size);
    uchar *labels = read_mnist_labels(TRAINING_LABELS, number_of_labels);
    uchar **test_images = read_mnist_images(TEST_IMAGES, number_of_test_images, image_size);
    uchar *test_labels = read_mnist_labels(TEST_LABELS, number_of_test_labels);
    MNISTShuffleDataAndLabels(images, labels, number_of_images);

    std::vector<MPI_Comm> layer_comms(params->GetLayers().size());
    for (int i = 0; i < layer_comms.size(); i++) {
	MPI_Comm_dup(MPI_COMM_WORLD, &layer_comms[i]);
    }

    std::cout << "Machine launched: " << hostname << std::endl;

    if (rank == MASTER_RANK) {
	SyncReplicasMasterNN *master = new SyncReplicasMasterNN(params, layer_comms, n_procs, n_procs-2-4);
	master->Train(test_images, test_labels, number_of_test_images);
	delete master;
    }
    else if (rank == EVALUATOR_RANK) {
	EvaluatorNN *evaluator = new EvaluatorNN(params, layer_comms, rank, n_procs);
	evaluator->Train(test_images, test_labels, number_of_test_images);
	delete evaluator;
    }
    else {
	WorkerNN *worker = new WorkerNN(params, layer_comms, rank, n_procs);
	worker->Train(test_images, test_labels, number_of_test_images);
	delete worker;
    }

    delete params;

    MPI_Barrier(MPI_COMM_WORLD);

    // Finalize the MPI environment.
    MPI_Finalize();
}
