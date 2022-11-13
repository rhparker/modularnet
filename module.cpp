#include "module.h"
#include <iostream>
#include <stdio.h>

//
// abstract module class
//

// constructor and destructor
Module::Module(int inputs, int outputs) : 
    inputs(inputs), outputs(outputs), num_layers(0), valid(1), pars(0), train(0) {};
Module::~Module() {}; 
void Module::properties() {};
void Module::add_layers(std::vector< std::vector <int> > config, double sigma) {};

// clear accumulated partial derivaties 
void Module::clear_partial() {
  for (int i = 0; i < num_layers; i++) {
    if (L[i]->pars > 0) {
      L[i]->clear_partial();
    }
  }
}

// compute partial derivative of loss with respect to parmeters 
void Module::partial_param(double* in, double* delta) {
  for (int i = 0; i < num_layers; i++) {
    if (L[i]->pars > 0) {
      L[i]->partial_param(z[i], this->delta[i+1]);
    }
  }
}

// update parameters using accumulated partial derivatives
void Module::update_param(double lr, int batch_size) {
  for (int i = 0; i < num_layers; i++) {
    if (L[i]->pars > 0) {
      L[i]->update_param(lr, batch_size);
    }
  }
}

#ifdef USE_MPI
// syncs layer in all ranks to rank 0
void Module::sync() {
  for (int i = 0; i < num_layers; i++) {
    if (L[i]->pars > 0) {
      L[i]->sync();
    }
  }
}
#endif

//
// Sequential module
//

// constructor and destructor
Sequential::Sequential(std::vector<int> config) : Module(config[1], config[2]) {};

Sequential::~Sequential() {
  if (num_layers > 0) {
    // delete layers
    for (int i = 0; i < num_layers; i++) {
      delete L[i];
    }
    delete[] L;
    // delete layer data
    for (int i = 0; i <= num_layers; i++) {
      delete[] z[i];
      delete[] delta[i];
    }
    delete[] z;
    delete[] delta;
    // delete layer sizes
    delete[] layer_sizes;
    delete[] layer_types;
  }
} 

// add layers
void Sequential::add_layers(std::vector< std::vector <int> > config, double sigma) {
  // allocate layers, sizes, and types
  int ins, outs;
  num_layers = config.size();
  L = new Layer*[num_layers];
  layer_sizes = new int[num_layers+1];
  layer_types = new int[num_layers];

  // iterate over layers
  for (int i = 0; i < num_layers; i++) {
    layer_types[i] = config[i][0];

    switch (layer_types[i]) {
      case LINEAR:
        L[i] = new Linear(config[i], sigma);
        break;

      case DROPOUT:
        if (i == 0) L[i] = new Dropout(config[i]);
        else L[i] = new Dropout( L[i-1]->outputs );
        break;

      case CONV:
        L[i] = new Conv(config[i], sigma);
        break;

      case MAXPOOL:
        L[i] = new Maxpool(config[i]);
        break;

      case SIG:
        if (i == 0) L[i] = new Dropout(config[i]);
        else L[i] = new Sigmoid( L[i-1]->outputs );
        break;

      case RELU:
        if (i == 0) L[i] = new Dropout(config[i]);
        L[i] = new ReLU( L[i-1]->outputs );
        break;

      case SOFTMAX:
        if (i == 0) L[i] = new Dropout(config[i]);
        L[i] = new Softmax( L[i-1]->outputs );
        break;
    }

    // take number of inputs and outputs from newly created layer
    ins  = L[i]->inputs;
    outs = L[i]->outputs;

    // total number of parameters
    pars += L[i]->pars;

    // if input-output mismatch, mark invalid
    // only matters for layers after the first
    if (i > 0 && layer_sizes[i] != ins ) {
      valid = ERROR_SIZE_MISMATCH;
    }
    layer_sizes[i]   = ins;
    layer_sizes[i+1] = outs;
  }

  // allocate layer data z and delta (one more than number of layers)
  z     = new double*[num_layers+1];
  delta = new double*[num_layers+1];
  for (int i = 0; i <= num_layers; i++) {
    z[i] = new double[ layer_sizes[i] ];
    delta[i] = new double[ layer_sizes[i] ];
  }
  // validate number of inputs and outputs from sequential layer
  if (inputs != layer_sizes[0] || outputs != layer_sizes[num_layers]) {
    valid = ERROR_SEQUENTIAL_IO_MISMATCH;
  }
}

// print parameters and properties
void Sequential::properties() {
  printf("Sequential module: %d layers, %d parameters\n", num_layers, pars);
  for (int i = 0; i < num_layers; i++) {
    std::cout << "  "; 
    L[i]->properties();
  }
  if (valid == ERROR_SIZE_MISMATCH) {
    std::cout << "error: layer input/output size mismatch" << std::endl;
  }
  if (valid == ERROR_SEQUENTIAL_IO_MISMATCH) {
    std::cout << "error: input/output size mismatch for sequential layer" << std::endl;
  }
  std::cout << std::endl;
}

// forward propagation on input
void Sequential::forward(double* in, double* out) {
  // copy input into sequential layer
  for (int i = 0; i < inputs; i++) {
    z[0][i] = in[i];
  }
  // forward propagate through network
  for (int i = 0; i < num_layers; i++) {
    L[i]->train = train;
    L[i]->forward(z[i],z[i+1]);
  }
  // copy output from sequential layer into output
  for (int i = 0; i < outputs; i++) {
    out[i] = z[num_layers][i];
  }
}

// forward propagation on output
void Sequential::backward(double* in, double* out, double* delta) {
  // copy output into net
  double outsum = 0;
  for (int i = 0; i < outputs; i++) {
    this->delta[num_layers][i] = out[i];
  }
  // work backwards from last layer
  for (int i = num_layers - 1; i >= 0; i--) {
    L[i]->backward(z[i], this->delta[i+1], this->delta[i]);
  }
  // copy final delta
  for (int i = 0; i < inputs; i++) {
    delta[i] = this->delta[0][i];
  }
}

