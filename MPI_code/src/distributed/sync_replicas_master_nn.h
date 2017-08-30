#ifndef _SYNC_REPLICAS_MASTER_NN_
#define _SYNC_REPLICAS_MASTER_NN_

#include "distributed_defines.h"

class SyncReplicasMasterNN : public NN {
 public:
   SyncReplicasMasterNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int n_procs, int n_to_collect) : NN(params), layer_comms(layer_comms) {
	this->comm = MPI_COMM_WORLD;
	this->n_to_collect = n_to_collect;
	this->n_procs = n_procs;
	this->cur_step = STEP_START;
	layer_send_requests.resize(layers.size());
	for (int i = 0; i < layers.size(); i++) {
	    for (int j = 0; j < n_procs; j++) {
		layer_send_requests[i].push_back(MPI_REQUEST_NULL);
	    }
	}

	// Preallocate memory for gradient buffers for irecv.
	for (int i = 0; i < layers.size()-1; i++) {
	    grad_buffers.push_back(std::vector<double *>());
	    for (int j = 0; j < N_RECV_REQUESTS_PER_LAYER; j++) {
		grad_buffers[i].push_back((double *)malloc(sizeof(double) * layers[i]->GetLayerCount()));
	    }
	}

	// Set gradients to 0
	for (int i = 0; i < layers.size()-1; i++) {
	    memset(layers[i]->GetGradient(), 0, sizeof(double) * layers[i]->GetLayerCount());
	}

	name = scheme_full_name("SyncReplicasWithBackup", n_to_collect, n_procs);

	// -2 for evaluator and master.
	timeline_out.open("outfiles/timeline_out_" + name);
	timeline_out << name << std::endl;

	SendEvaluatorSchemeName();
    }

    ~SyncReplicasMasterNN() {
	timeline_out.close();
    }

    void Train(uchar **data, uchar *labels, int examples) override {

	std::vector<int> gradients_accumulated(layers.size());
	std::fill(gradients_accumulated.begin(),
		  gradients_accumulated.end(),
		  0);

	AsynchronousFetchGradientsStart();

	start_training_time = GetTimeMillis();

	while (cur_step < N_TRAIN_ITERS) {
	    AsynchronousBroadcastStep();
	    AsynchronousBroadcastLayerWeights();

#if GENERATE_TIMELINE
	    LogReceptionEvent(cur_step, 1);
#endif

	    bool enough_gradients_received = false;
	    while (!enough_gradients_received) {

		// While we don't have enough gradients, keep waiting to receive them.
		int index_received = -1;
		MPI_Status stat;
		MPI_Waitany((layers.size()-1) * N_RECV_REQUESTS_PER_LAYER,
			    gradient_fetch_requests.data(),
			    &index_received,
			    &stat);

		// We push N_RECV_REQUESTS_PER_LAYER per layer. The layer is
		// index / N_RECV_REQUESTS_PER_LAYER
		int layer_received = index_received / N_RECV_REQUESTS_PER_LAYER;
		int copy_index = index_received - layer_received * N_RECV_REQUESTS_PER_LAYER;

#if GENERATE_TIMELINE
		LogReceptionEvent(stat.MPI_TAG, 0);
#endif

		if (stat.MPI_TAG == cur_step) {

		    int count = 0;
		    MPI_Get_count(&stat, MPI_DOUBLE, &count);
		    assert(count == layers[layer_received]->GetLayerCount());

		    gradients_accumulated[layer_received]++;

		    // Sum gradient
		    //layers[layer_received]->ApplyGrad(learning_rate, grad_buffers[layer_received][copy_index]);
		    MatrixAdd(grad_buffers[layer_received][copy_index], layers[layer_received]->GetGradient(), layers[layer_received]->GetGradient(),
			      1, 1,
			      layers[layer_received]->NRows(),
			      layers[layer_received]->NCols(),
			      layers[layer_received]->NCols(),
			      layers[layer_received]->NCols(),
			      layers[layer_received]->NCols());

		    memset(grad_buffers[layer_received][copy_index], 0, sizeof(double) * layers[layer_received]->GetLayerCount());

		    enough_gradients_received = true;
		    for (int i = 0; i < layers.size()-1; i++) {
			enough_gradients_received = enough_gradients_received && gradients_accumulated[i] >= n_to_collect;
		    }

		    std::cout << "Gradients accumulated: ";
		    for (int i = 0; i < layers.size(); i++) {
			std::cout << gradients_accumulated[i] << " ";
		    }
		    std::cout << endl;

		}

		// Received a gradient for this layer... Initiate a new
		// request to receive another one at the layer for the specific copy index.
		AsynchronousFetchGradient(layer_received, copy_index, &gradient_fetch_requests[index_received]);
	    }

	    // Apply average gradient
	    for (int layer = 0; layer < layers.size()-1; layer++) {
		layers[layer]->ApplyGrad(learning_rate / gradients_accumulated[layer],
					 layers[layer]->GetGradient());
		memset(layers[layer]->GetGradient(), 0, sizeof(double) * layers[layer]->GetLayerCount());
	    }

	    std::fill(gradients_accumulated.begin(),
		      gradients_accumulated.end(), 0);

	    cur_step++;
	}

	AsynchronousBroadcastStep();
    }

 protected:
    MPI_Request step_broadcast_req;
    int n_procs, cur_step, n_to_collect;
    double start_training_time;
    string name;
    ofstream timeline_out;
    MPI_Comm comm;
    std::vector<std::vector<MPI_Request> > layer_send_requests;
    std::vector<MPI_Request> gradient_fetch_requests;
    std::vector<MPI_Comm> &layer_comms;
    std::vector<std::vector<double *> > grad_buffers;

    void SendEvaluatorSchemeName() {
	MPI_Send((void *)name.c_str(), name.length()+1, MPI_CHAR, EVALUATOR_RANK, 0, comm);
    }

    void AsynchronousBroadcastStep() {
	for (int i = 0; i < n_procs; i++) {
	    if (i != MASTER_RANK) {
		MPI_Isend(&cur_step, 1, MPI_INT, i, STEP_TAG, comm, &step_broadcast_req);
	    }
	}
    }

    void AsynchronousFetchGradient(int l, int copy, MPI_Request *req) {
	MPI_Irecv(grad_buffers[l][copy],
		  layers[l]->GetLayerCount(),
		  MPI_DOUBLE,
		  MPI_ANY_SOURCE,
		  MPI_ANY_TAG,    // Any gradient from any iteration may be fetched.
		  layer_comms[l],
		  req);
    }

    void AsynchronousFetchGradientsStart() {
	for (int i = 0; i < layers.size()-1; i++) {
	    for (int k = 0; k < N_RECV_REQUESTS_PER_LAYER; k++) {
		gradient_fetch_requests.push_back(MPI_REQUEST_NULL);
	    }
	}

	// We initiate a ton of requests to receive gradients for each layer.
	for (int l = 0; l < layers.size()-1; l++) {
	    for (int k = 0; k < N_RECV_REQUESTS_PER_LAYER; k++) {
		AsynchronousFetchGradient(l, k, &gradient_fetch_requests[l*N_RECV_REQUESTS_PER_LAYER+k]);
	    }
	}
    }

    void LogReceptionEvent(int step, int is_master) {
	double time = GetTimeMillis() - start_training_time;
	timeline_out << time << " " << step << " " << is_master << std::endl;
    }

    void AsynchronousBroadcastLayerWeights() {
	for (int l = 0; l < layers.size()-1; l++) {
	    for (int i = 0; i < n_procs; i++) {
		if (i != MASTER_RANK) {
		    if (layer_send_requests[l][i] != MPI_REQUEST_NULL) {
			MPI_Request_free(&layer_send_requests[l][i]);
		    }

		    MPI_Isend(layers[l]->GetLayer(),
			      layers[l]->GetLayerCount(),
			      MPI_DOUBLE,
			      i,
			      cur_step,
			      layer_comms[l],
			      &layer_send_requests[l][i]);
		}
	    }
	}
    }
};

#endif
