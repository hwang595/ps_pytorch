#ifndef _distributed_defines_
#define _distributed_defines_

#include <iostream>
#include <list>
#include <string>
#include <vector>
#include <map>
#include <unistd.h>
#include <time.h>
#include <cassert>
#include "../mnist/mnist.h"
#include "../nn/nn.h"
#include "../nn/nn_params.h"
#include "../nn/nn_layer.h"
#include <mpi.h>

#define STEP_TAG 0
#define STEP_START 1
#define STEP_UNINITIALIZED (STEP_START-1)
#define MASTER_RANK 0
#define EVALUATOR_RANK 1
#define N_RECV_REQUESTS_PER_LAYER 100
#ifndef SHORTCIRCUIT
#define SHORTCIRCUIT true
#endif
#define GENERATE_TIMELINE false
#define N_TRAIN_ITERS 100

string scheme_full_name(string scheme_name, int n_to_collect, int n_procs) {

    // -2 for master and evaluator
    string name = scheme_name + std::to_string(n_to_collect) + "_"  + std::to_string(n_procs-2);
    if (SHORTCIRCUIT) {
	name += "_shortcircuit";
    }
    else {
	name += "_no_shortcircuit";
    }
    return name;
}

#endif
