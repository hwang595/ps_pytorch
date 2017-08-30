#ifndef _EVALUATOR_NN_
#define _EVALUATOR_NN_

#include "distributed_defines.h"

class EvaluatorNN : public NN {
 public:
   EvaluatorNN(NNParams *params, std::vector<MPI_Comm> &layer_comms, int rank, int n_procs) : NN(params), layer_comms(layer_comms) {
	this->rank = rank;
	this->n_procs = n_procs;
	this->cur_step = STEP_UNINITIALIZED;
	this->comm = MPI_COMM_WORLD;
	this->next_step = STEP_UNINITIALIZED;
	this->step_fetch_request = MPI_REQUEST_NULL;

	for (int i = 0; i < layers.size(); i++) {
	    layer_cur_step.push_back(STEP_UNINITIALIZED);
	    layer_fetch_requests.push_back(MPI_REQUEST_NULL);
	}

	ReceiveMasterSchemeName();
	time_loss_out.open("outfiles/time_loss_out_" + name);
    }

    ~EvaluatorNN() {
	time_loss_out.close();
    }

    void Train(uchar **data, uchar *labels, int n_examples) override {

	time_loss_out << name << std::endl;

	SynchronousFetchStep();

	double start_training_time = GetTimeMillis();

	while (true) {
	    AsynchronousFetchStepUpdate();
	    bool changed = UpdateStep();
	    AsynchronousFetchWeights();

	    if (cur_step >= N_TRAIN_ITERS) break;

	    if (changed) {

		// Wait for the synced weight layer to be fetched
		for (int i = 0; i < layers.size(); i++) {
		    if (i != layers.size()-1) {
			MPI_Wait(&layer_fetch_requests[i], MPI_STATUS_IGNORE);
			layer_cur_step[i] = cur_step;
		    }
		}

		// Evaluate on these weights
		double loss = ComputeLoss(data, labels, n_examples);
		double err_rate = ComputeErrorRate(data, labels, n_examples);
		double time = GetTimeMillis() - start_training_time;
		time_loss_out << cur_step << " " << time << " " << loss << " " << err_rate << std::endl;
	    }
	}
    }

 protected:

    // The synchronized step (should be the same across workers & master)
    int cur_step, rank, n_procs, next_step;
    MPI_Comm comm;
    MPI_Request step_fetch_request;
    string name;
    ofstream time_loss_out;

    // layer_cur_step[i] is the iteration step for the current weights
    std::vector<int> layer_cur_step;

    // Requests for fetching each layer.
    std::vector<MPI_Request> layer_fetch_requests;

    // Layer communicator handles
    std::vector<MPI_Comm> &layer_comms;

    void ReceiveMasterSchemeName() {
	MPI_Status stat;
	MPI_Probe(MASTER_RANK, 0, comm, &stat);
	int n_chars = 0;
	MPI_Get_count(&stat, MPI_CHAR, &n_chars);
	char *name_holder = (char *)malloc(sizeof(char) * n_chars);
	MPI_Recv(name_holder,
		 n_chars,
		 MPI_CHAR,
		 MASTER_RANK,
		 0,
		 comm,
		 MPI_STATUS_IGNORE);

	name = string(name_holder);
	free(name_holder);
    }

    int NewStepQueued() {
	int found_step = 0;
	MPI_Iprobe(0, STEP_TAG, this->comm, &found_step, MPI_STATUS_IGNORE);
	return found_step;
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

    bool UpdateStep() {
	bool changed = cur_step != next_step;
	cur_step = next_step;
	return changed;
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
