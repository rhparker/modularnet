#include <random>
#include <vector>

#include "layer.h"

#ifndef _MODULE
#define _MODULE

// types of modules
#define SEQUENTIAL 1001

// module errors
#define ERROR_SEQUENTIAL_IO_MISMATCH -2

//
// abstract module class
//

class Module {
  public:
    // number of inputs, outputs, and parameters
    int inputs;
    int outputs;
    int pars;

    // are we training or not?
    int train;

    // number of layers
    int num_layers;
    // input/output sizes of layers
    int* layer_sizes;
    // types of layers
    int* layer_types;
    // is network valid?
    int valid;

    // layers
    Layer** L;
    // z: data (inputs and outputs from layers) 
    double** z;
    // delta: partial derivatives with respect to layer outputs z
    double** delta;

    // constructor and destructor
    Module(int inputs, int outputs);
    virtual ~Module(); 
    virtual void add_layers(std::vector< std::vector <int> > config, double sigma);

    // print parameters and properties
    virtual void properties();

    // forward and backward propagation
    virtual void forward(double* in, double* out) = 0;
    virtual void backward(double* in, double* out, double* delta) = 0;

    // compute partial derivative of loss with respect to parmeters 
    void partial_param(double* in, double* delta);

    // clear accumulated partial derivaties 
    void clear_partial();

    // update parameters using accumulated partial derivatives
    void update_param(double lr, int batch_size);

#ifdef USE_MPI
    // syncs layer in all ranks to rank 0
    void sync();
#endif
};

//
// sequential module
//

class Sequential : public Module {
  public:
    // constructor and destructor
    Sequential(std::vector<int> config);
    ~Sequential(); 

    // add layers
    void add_layers(std::vector< std::vector <int> > config, double sigma);

    // // print parameters and properties
    void properties();

    // forward and backward propagation
    void forward(double* in, double* out);
    void backward(double* in, double* out, double* delta);

    // clear accumulated partial derivaties 
    void clear_partial();

    // compute partial derivative of loss with respect to parmeters 
    void partial_param(double* in, double* delta);

    // update parameters using accumulated partial derivatives
    void update_param(double lr, int batch_size);
};

#endif