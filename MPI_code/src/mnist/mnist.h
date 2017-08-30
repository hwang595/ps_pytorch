#ifndef _MNIST_
#define _MNIST_

#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cassert>
#include <string>
#include <fstream>
#include <iterator>
#include <algorithm>

using namespace std;

typedef unsigned char uchar;

#define TRAINING_IMAGES "./data/train-images-idx3-ubyte"
#define TRAINING_LABELS "./data/train-labels-idx1-ubyte"
#define TEST_IMAGES "./data/t10k-images-idx3-ubyte"
#define TEST_LABELS "./data/t10k-labels-idx1-ubyte"
#define IMAGE_X 28
#define IMAGE_Y 28
#define N_CLASSES 10

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

uchar* read_mnist_labels(string full_path, int& number_of_labels) {

    ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        uchar* _dataset = new uchar[number_of_labels];
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_dataset[i], 1);
        }
        return _dataset;
    } else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size) {

    ifstream file(full_path, ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        uchar** _dataset = new uchar*[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new uchar[image_size];
            file.read((char *)_dataset[i], image_size);
        }
        return _dataset;
    } else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

void test_load_images(string path, int n_expected) {
    int n_images_found = 0;
    int image_size = 0;
    uchar** dataset = read_mnist_images(path, n_images_found, image_size);
    assert(n_expected == n_images_found);
    assert(image_size == IMAGE_X*IMAGE_Y);
}

void test_load_labels(string path, int n_expected) {
    int n_labels_found = 0;
    uchar* dataset = read_mnist_labels(path, n_labels_found);
    assert(n_expected == n_labels_found);
}

void MNISTImageToInput(int batchsize, uchar **images, double *output) {

    for (int i = 0; i < batchsize; i++) {
	for (int j = 0; j < IMAGE_X*IMAGE_Y; j++) {
	    output[i * IMAGE_X*IMAGE_Y + j] = images[i][j] / (double)255;
	}
    }
}

void MNISTOneHotLabelsToInput(int batchsize, uchar *labels, double *output) {
    for (int i = 0; i < batchsize; i++) {
	double *output_row = &output[i*N_CLASSES];
	memset(output_row, 0, sizeof(double) * N_CLASSES);
	output_row[(int)labels[i]] = 1;
    }
}

void MNISTSwapImages(uchar **images, int i, int j) {
    uchar tmp[IMAGE_X*IMAGE_Y];
    memcpy(tmp, images[i], sizeof(uchar) * IMAGE_X*IMAGE_Y);
    memcpy(images[i], images[j], sizeof(uchar) * IMAGE_X*IMAGE_Y);
    memcpy(images[j], tmp, sizeof(uchar) * IMAGE_X*IMAGE_Y);
}

void MNISTSwapLabels(uchar *labels, int i, int j) {
    uchar tmp = labels[i];
    labels[i] = labels[j];
    labels[j] = tmp;
}

void MNISTShuffleDataAndLabels(uchar **images, uchar *labels, int n_examples) {
    for (int i = n_examples-1; i >= 0; i--) {
	int to_swap_to = rand() % (i+1);
	MNISTSwapImages(images, i, to_swap_to);
	MNISTSwapLabels(labels, i, to_swap_to);
    }
}

void PrintPicture(double *data) {
    int k = 0;
    for (int i = 0; i < IMAGE_Y; i++) {
	for (int j = 0; j < IMAGE_X; j++) {
	    if (data[k++] > 0) {
		std::cout << "x";
	    }
	    else {
		std::cout << ".";
	    }
	}
	std::cout << std::endl;
    }
}

void test_load_data() {
    std::cout << "Test loading mnist data..." << std::endl;
    test_load_images(TRAINING_IMAGES, 60000);
    test_load_images(TEST_IMAGES, 10000);
    test_load_labels(TRAINING_LABELS, 60000);
    test_load_labels(TEST_LABELS, 10000);
    std::cout << "Test succeeded!" << std::endl;
}

#endif
