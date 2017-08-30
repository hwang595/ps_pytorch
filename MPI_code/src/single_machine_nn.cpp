#include <iostream>
#include "mnist/mnist.h"
#include "nn/nn.h"

void test() {
    test_load_data();
    test_nn();
}

int main(void) {
    test();
}
