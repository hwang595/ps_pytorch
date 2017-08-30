#ifndef _NN_LAYER_
#define _NN_LAYER_

#include <iostream>
#include <cassert>
#include <cstring>
#include <cblas.h>
#include <vector>
#include <random>
#include "../mnist/mnist.h"
#include "../util/util.h"
#include "../distributed/distributed_defines.h"

class NNLayer;

class NNLayer {
 public:

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    NNLayer(int batchsize, int n_rows, int n_cols, bool is_input, bool is_output, int step, double learning_rate) {
	std::cout << "Initializing NNLayer of dimension " << n_rows << "x" << n_cols << std::endl;
	weights = S = Z = F = output = input = D = grad = NULL;
	next = prev = NULL;
	this->step = step;
	this->batchsize = batchsize;
	this->n_rows = n_rows;
	this->n_cols = n_cols;
	this->is_input = is_input;
	this->is_output = is_output;
	this->lr = learning_rate;
	distribution = std::normal_distribution<double>(0, 1);

	if (is_input) {
	    AllocateMemory(&input, (n_rows+1)*batchsize);
	    for (int b = 0; b < batchsize; b++) {
		input[b * (n_rows+1) + n_rows] = 1;
	    }
	}
	if (is_output) {
	    AllocateMemory(&output, batchsize*n_rows);
	}
	AllocateMemory(&S, batchsize*n_rows);

	// We add +1 for the bias column.
	AllocateMemory(&Z, (n_rows+1)*batchsize);
	for (int b = 0; b < batchsize; b++) {
	    Z[b * (n_rows+1) + n_rows] = 1;
	}

	AllocateMemory(&F, n_rows*batchsize);

	if (!is_output) {

	    // We add +1 for the bias weights.
	    int n_rows_to_allocate = n_rows+1;
	    AllocateMemory(&weights, n_rows_to_allocate * n_cols);
	    InitializeGaussian(weights, n_rows_to_allocate * n_cols);
	    AllocateMemory(&grad, n_rows_to_allocate * n_cols);
	}

	AllocateMemory(&D, n_rows*batchsize);
    }

    void WireLayers(NNLayer *prev, NNLayer *next) {
	this->next = next;
	this->prev = prev;
    }

    void ForwardPropagate(double *data) {
	//std::cout << GetDescription() << ": Forward propagate - " << GetTimeMillis() << std::endl;;
	ForwardPropagateCore(data);
	if (next)
	    next->ForwardPropagate(data);
    }

    void ApplyGrad(double learning_rate, double *local_grad) {
	MatrixAdd(weights, local_grad, weights,
		  1, -learning_rate,
		  n_rows+1, n_cols,
		  n_cols, n_cols, n_cols);
    }

    void BackPropagate(double *labels) {
	//std::cout << GetDescription() << ": Back propagate - " << GetTimeMillis() << std::endl;
	BackPropagateCore(labels);
	ApplyGrad(lr, grad);
	if (prev)
	    prev->BackPropagate(labels);
    }

    string GetDescription() {
	return "Layer " + std::to_string(n_rows) + "x" + std::to_string(n_cols);
    }

    void ForwardPropagateCore(double *data) {

	// Be sure to memset next->S as gemm += rather than =.
	if (next) {
	    memset(next->S, 0, sizeof(double) * batchsize * n_cols);
	}

	if (is_input) {
	    // Compute S = Input * W
	    for (int i = 0; i < batchsize; i++) {
		for (int j = 0; j < n_rows; j++) {
		    input[i*(n_rows+1)+j] = data[i*n_rows+j];
		}
	    }
	    MatrixMultiply(input, weights, next->S,
			   batchsize, n_cols, n_rows+1,
			   n_rows+1, n_cols, n_cols);
	}
	else {

	    if (is_output) {

		for (int b = 0; b < batchsize; b++) {
		    Softmax(&S[b*n_rows], &output[b*n_rows], n_rows);
		}
		for (int b = 0; b < batchsize; b++) {
		    for (int r = 0; r < n_rows; r++) {
			Z[b*(n_rows+1)+r] = output[b*n_rows+r];
		    }
		}
		return;
	    }

	    // Compute Z_i = f(S_i)
	    SigmoidActivation(S, Z, batchsize, n_rows, n_rows, n_rows+1);

	    // Compute F_i = f'_i(S_i)^T
	    SigmoidActivationGradient(S, F, batchsize, n_rows, n_rows, n_rows);

	    // Compute S_j = Z_i W_i
	    MatrixMultiply(Z, weights, next->S,
			   batchsize, n_cols, n_rows+1,
			   n_rows+1, n_cols, n_cols);

	}
    }

    void BackPropagateCore(double *labels) {
	memset(D, 0, sizeof(double) * n_rows * batchsize);

	if (is_output) {

	    // Here we actually have D'
	    MatrixAdd(Z, labels, D, 1, -1,
		      batchsize, n_rows,
		      n_rows+1, n_rows, n_rows);
	}
	else {

	    // Compute D' * W'
	    MatrixMultiplyTransB(next->D, weights, D,
				 batchsize, n_rows, n_cols,
				 n_cols, n_cols, n_rows);

	    // Compute D'
	    MultiplyEntrywise(D, F, D,
			      batchsize, n_rows,
			      n_rows, n_rows, n_rows);

	    memset(grad, 0, sizeof(double) * (n_rows+1) * n_cols);
	    if (is_input) {
		MatrixMultiplyTransA(input, next->D, grad,
				     n_rows+1, n_cols, batchsize,
				     n_rows+1, n_cols, n_cols);
	    }
	    else {
		MatrixMultiplyTransA(Z, next->D, grad,
				     n_rows+1, n_cols, batchsize,
				     n_rows+1, n_cols, n_cols);
	    }
	}
    }

    void IncStep() {
	step++;
	//lr *= .90;
    }

    size_t GetLayerCount() {
	return (n_rows+1) * n_cols;
    }

    double *GetLayer() {
	return weights;
    }

    double *GetGradient() {
	return grad;
    }

    int Dimension() {
	return n_rows;
    }

    int NCols() {
	return n_cols;
    }

    int NRows() {
	return n_rows+1;
    }

    double *Output() {
	assert(is_output);
	return output;
    }


    ~NNLayer() {
	if (weights != NULL) free(weights);
	if (S != NULL) free(S);
	if (Z != NULL) free(Z);
	if (F != NULL) free(F);
	if (input != NULL) free(input);
	if (output != NULL) free(output);
    }

    double *weights, *S, *Z, *F, *input, *output, *D, *grad;

 protected:

    // Note that n_cols does account for the implicit column of noes
    // for the bias.
    int n_rows, n_cols, batchsize, step;
    bool is_input, is_output;
    NNLayer *next, *prev;
    double lr;

    void InitializeGaussian(double *ptr, int n_elements) {
	for (int i = 0; i < n_elements; i++) {
	    ptr[i] = distribution(generator);
	}
    }
};

#endif
