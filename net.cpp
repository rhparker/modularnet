#include "net.h"
#include <iostream>

// constructor
Net::Net(std::vector< std::vector <int> > config) : 
      num_modules( config.size() ), valid(1), pars(0) {
  int ins, outs;

  // allocate modules, sizes, and types
  M = new Module*[num_modules];
  module_sizes = new int[num_modules+1];
  module_types = new int[num_modules];

  // iterate over modules
  for (int i = 0; i < num_modules; i++) {
    module_types[i] = config[i][0];

    switch (module_types[i]) {
      case SEQUENTIAL:
        M[i] = new Sequential(config[i]);
        break;
    }

    // take number of inputs and outputs from newly created module
    ins  = M[i]->inputs;
    outs = M[i]->outputs;

    // total number of parameters
    pars += M[i]->pars;

    // if input-output mismatch, mark invalid
    // only matters for modules after the first
    if (i > 0 && module_sizes[i] != ins ) {
      valid = ERROR_SIZE_MISMATCH;
    }
    module_sizes[i]   = ins;
    module_sizes[i+1] = outs;
  }

  // allocate module data z and delta (one more than number of modules)
  z     = new double*[num_modules+1];
  delta = new double*[num_modules+1];
  for (int i = 0; i <= num_modules; i++) {
    z[i] = new double[ module_sizes[i] ];
    delta[i] = new double[ module_sizes[i] ];
  }
}

// destructor
Net::~Net() {
  // delete modules
  for (int i = 0; i < num_modules; i++) {
    delete M[i];
  }
  delete[] M;
  // delete module data
  for (int i = 0; i <= num_modules; i++) {
    delete[] z[i];
    delete[] delta[i];
  }
  delete[] z;
  delete[] delta;
  // delete module sizes
  delete[] module_sizes;
}

void Net::add_layers(int module_id, std::vector< std::vector <int> > config, double sigma) {
  if (module_id >= 0 && module_id < num_modules) {
    M[module_id]->add_layers(config, sigma);
    pars += M[module_id]->pars;
  }
}

// forward propagation on input (for training or evaluation)
void Net::forward(double* in, int train) {
  // copy input into net
  for (int i = 0; i < module_sizes[0]; i++) {
    z[0][i] = in[i];
  }
  // forward propagate through network
  for (int i = 0; i < num_modules; i++) {
    M[i]->train = train;
    M[i]->forward(z[i],z[i+1]);
  }
}

// backward propagation on output
void Net::backward(double* out) {
  // copy output into net
  for (int i = 0; i < module_sizes[num_modules]; i++) {
    delta[num_modules][i] = out[i];
  }
  // work backwards from last module
  for (int i = num_modules - 1; i >= 0; i--) {
    M[i]->backward(z[i], delta[i+1], delta[i]);
  }
}

// clear accumulated partial derivaties 
void Net::clear_partial() {
  for (int i = 0; i < num_modules; i++) {
    if (M[i]->pars > 0) {
      M[i]->clear_partial();
    }
  }
}

// update/accumulate partial derivaties 
void Net::partial_param() {
  for (int i = 0; i < num_modules; i++) {
    if (M[i]->pars > 0) {
      M[i]->partial_param(z[i], delta[i+1]);
    }
  }
}

// update parameters using accumulated partial derivatives
void Net::update_param(double lr, int batch_size) {
  for (int i = 0; i < num_modules; i++) {
    if (M[i]->pars > 0) {
      M[i]->update_param(lr, batch_size);
    }
  }
}

// print properties
void Net::properties() {
  std::cout << "Modules: " << num_modules << ",  Parameters: " << pars 
      << std::endl << std::endl;
  for (int i = 0; i < num_modules; i++) {
    M[i]->properties();
  }
  if (valid == ERROR_SIZE_MISMATCH) {
    std::cout << "error: module input/output size mismatch" << std::endl;
  }
  std::cout << std::endl;
}

#ifdef USE_MPI
// sync paramaters of all ranks in all modules to rank 0
void Net::sync() {
  for (int i = 0; i < num_modules; i++) {
    if (M[i]->pars > 0) {
      M[i]->sync();
    }
  }
}
#endif

