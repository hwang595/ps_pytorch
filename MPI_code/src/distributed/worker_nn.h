#ifndef _WORKER_NN_
#define _WORKER_NN_

#include "distributed_defines.h"

struct LayerSendRequest {
    MPI_Request request;
    int step;
};

typedef struct LayerSendRequest LayerSendRequest;

class WorkerNN : public NN {
 public:
   WorkerNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int rank, int n_procs) : NN(params), layer_comms(layer_comms) {
	this->rank = rank;
	this->n_procs = n_procs;
	this->cur_step = STEP_UNINITIALIZED;
	this->comm = MPI_COMM_WORLD;
	this->next_step = STEP_UNINITIALIZED;
	this->step_fetch_request = MPI_REQUEST_NULL;


	for (int i = 0; i < layers.size(); i++) {
	    layer_cur_step.push_back(STEP_UNINITIALIZED);
	    layer_fetch_requests.push_back(MPI_REQUEST_NULL);
	    layer_send_requests.push_back(MPI_REQUEST_NULL);
	}
    }

    void Train(uchar **data, uchar *labels, int n_examples) override {

	// Boolean indicating whether it's the first pass through training.
	// Not equivalent to step, as a worker can repeat a step.

	SynchronousFetchStep();
	assert(UpdateStep());
	assert(cur_step == STEP_START);

	std::cout << "Worker " << rank << " starting training..." << std::endl;
	bool first = true;

	while (true) {

	    AsynchronousFetchStepUpdate();
	    bool updated = UpdateStep();
	    if (!updated && !first) continue;
	    first = false;
	    std::cout << rank << " " <<cur_step << std::endl;
	    AsynchronousFetchWeights();
	    FillNextBatch(data, labels, n_examples);

	    if (cur_step >= N_TRAIN_ITERS) break;

	    // Forward propagate
	    for (int i = 0; i < layers.size(); i++) {

		// Handle short circuiting.
#if SHORTCIRCUIT
		if (StepChanged()) {
		    std::cout << "SHORTCIRCUIT" << std::endl;
		    break;
	        }
#endif

		// Wait for the synced weight layer to be fetched
		if (i != layers.size()-1) {
		    MPI_Wait(&layer_fetch_requests[i], MPI_STATUS_IGNORE);
		    layer_cur_step[i] = cur_step;
		}

		// Do forward propagation
		layers[i]->ForwardPropagateCore(batch_data_placeholder);
	    }

	    // Back propagate
	    for (int i = layers.size()-1; i >= 0; i--) {

#if SHORTCIRCUIT
		if (StepChanged()) {
		  std::cout << "SHORTCIRCUIT" << std::endl;
		    break;
		}
#endif

		// Check that the previous gradient has been sent
		if (i != layers.size()-1) {
		    if (layer_send_requests[i] != MPI_REQUEST_NULL) {
			MPI_Wait(&layer_send_requests[i], MPI_STATUS_IGNORE);
		    }
		}

		// Backpropagate core.
		layers[i]->BackPropagateCore(batch_labels_placeholder);

		// Send the layer's gradient.
		if (i != layers.size()-1) {

		    // Do a buffered send to avoid having to wait for recv on the other end.
		    MPI_Isend(layers[i]->GetGradient(),
			      layers[i]->GetLayerCount(),
			      MPI_DOUBLE,
			      MASTER_RANK,
			      cur_step,
			      layer_comms[i],
			      &layer_send_requests[i]);
		}
	    }
	}
    }

 protected:

    // The synchronized step (should be the same across workers & master)
    int cur_step, rank, n_procs, next_step;
    MPI_Comm comm;

    // layer_cur_step[i] is the iteration step for the current weights
    std::vector<int> layer_cur_step;

    // Requests for fetching each layer.
    std::vector<MPI_Request> layer_fetch_requests;
    std::vector<MPI_Request> layer_send_requests;

    // Layer communicator handles
    std::vector<MPI_Comm> &layer_comms;

    // Requests for fetching the step.
    MPI_Request step_fetch_request;

    bool StepChanged() {
	return NewStepQueued() || cur_step != next_step;
    }

    // Returns whether fetched new step was different from cur step.
    bool UpdateStep() {
	bool changed = cur_step != next_step;
	cur_step = next_step;
	return changed;
    }

    int NewStepQueued() {
	int found_step = 0;
	MPI_Iprobe(0, STEP_TAG, this->comm, &found_step, MPI_STATUS_IGNORE);
	return found_step;
    }

    void AsynchronousFetchStepUpdate() {
	int completed_step_fetch = 0;
	if (step_fetch_request == MPI_REQUEST_NULL)
	    completed_step_fetch = 1;
	else
	    MPI_Test(&step_fetch_request, &completed_step_fetch, MPI_STATUS_IGNORE);
	if (completed_step_fetch) {
	    AsynchronousFetchStep();
	}
    }

    void AsynchronousFetchStep() {
	while (NewStepQueued()) {
	    MPI_Irecv(&next_step,
		      1,
		      MPI_INT,
		      MASTER_RANK,
		      STEP_TAG,
		      this->comm,
		      &step_fetch_request);
	}
	MPI_Irecv(&next_step,
		  1,
		  MPI_INT,
		  MASTER_RANK,
		  STEP_TAG,
		  this->comm,
		  &step_fetch_request);
    }

    void SynchronousFetchStep() {
	MPI_Recv(&next_step,
		 1,
		 MPI_INT,
		 MASTER_RANK,
		 STEP_TAG,
		 this->comm,
		 MPI_STATUS_IGNORE);
    }

    // Fetch all layer weights asynchronously. (from master).
    void AsynchronousFetchWeights() {

	// Last layer has no weights.
	for (int i = 0; i < layers.size()-1; i++) {
	    // Check if we have already fetched the weights for this
	    // particular step. If so, don't fetch it.
	    if (layer_cur_step[i] < cur_step) {

		// Be sure to free the layer fetch requests
		if (layer_fetch_requests[i] != MPI_REQUEST_NULL) {
		    MPI_Request_free(&layer_fetch_requests[i]);
		}

		MPI_Irecv(layers[i]->GetLayer(),
			  layers[i]->GetLayerCount(),
			  MPI_DOUBLE,
			  MASTER_RANK,
			  cur_step,
			  layer_comms[i],
			  &layer_fetch_requests[i]);
	    }
	}
    }
};

#endif
