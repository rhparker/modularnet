#include "module.h"
#include <vector>

#ifndef _NET
#define _NET

class Net {
  public:
    // number of modules
    int num_modules;
    // input/output sizes of modules
    int* module_sizes;
    // types of layers
    int* module_types;
    // is network valid?
    int valid;
    
    // total number of parameters
    int pars;

    // layers
    Module** M;

    // z: data (inputs and outputs from modules) 
    double** z;
    // delta: partial derivatives with respect to module outputs z
    double** delta;

    // constructor and destructor
    Net(std::vector< std::vector <int> > config);
    ~Net(); 

    // add layers to a module
    void add_layers(int module_id, std::vector< std::vector <int> > config, double sigma);

    // forward propagation on input
    void forward(double* in, int train);

    // backward propagation on output
    void backward(double* out);

    // clear accumulated partial derivaties 
    void clear_partial();

    // accumulate partial derivatives with respect to parameters
    void partial_param();

    // update parameters using accumulated partial derivatives
    void update_param(double lr, int batch_size);

    // print properties
    void properties();

#ifdef USE_MPI
    // sync paramaters of all ranks in all layers to rank 0
    void sync();
#endif

};

#endif