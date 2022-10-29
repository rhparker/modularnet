#include "net.h"
#include <iostream>

// constructor
Net::Net(int num_layers) : num_layers(num_layers), pars(0) {
  // allocate layers and sizes
  L = new Layer*[num_layers];
  layer_sizes = new int[num_layers+1];

  // allocate layer data a and delta (one more than number of layers)
  z     = new double*[num_layers+1];
  delta = new double*[num_layers+1];
}

// destructor
Net::~Net() {
  delete[] L;
  delete[] layer_sizes;
}
// forward propagation on input (for training or evaluation)
void Net::forward(double* in, int train) {
  // copy input into net
  for (int i = 0; i < layer_sizes[0]; i++) {
    z[0][i] = in[i];
  }
  // forward propagate through network
  for (int i = 0; i < num_layers; i++) {
    L[i]->train = train;
    L[i]->forward(z[i],z[i+1]);
  }
}

// backward propagation on output
void Net::backward(double* out) {
  // copy output into net
  for (int i = 0; i < layer_sizes[num_layers]; i++) {
    delta[num_layers][i] = out[i];
  }
  // work backwards from last layer
  for (int i = num_layers - 1; i > 0; i--) {
    L[i]->backward(z[i], delta[i+1], delta[i]);
  }
}

// clear accumulated partial derivaties 
void Net::clear_partial() {
  for (int i = 0; i < num_layers; i++) {
    if (L[i]->pars > 0) {
      L[i]->clear_partial();
    }
  }
}

// update/accumulate partial derivaties 
void Net::partial_param() {
  for (int i = 0; i < num_layers; i++) {
    if (L[i]->pars > 0) {
      L[i]->partial_param(z[i], delta[i+1]);
    }
  }
}

// update parameters using accumulated partial derivatives
void Net::update_param(double lr, int batch_size) {
  for (int i = 0; i < num_layers; i++) {
    if (L[i]->pars > 0) {
      L[i]->update_param(lr, batch_size);
    }
  }
}

#ifdef USE_MPI
// sync paramaters of all ranks in all layers to rank 0
void Net::sync() {
  for (int i = 0; i < num_layers; i++) {
    if (L[i]->pars > 0) {
      L[i]->sync();
    }
  }
}
#endif

