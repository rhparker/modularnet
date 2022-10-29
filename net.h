#include "layer.h"

class Net {
  public:
    int num_layers;
    int* layer_sizes;
    
    // total number of parameters
    int pars;

    // layers
    Layer** L;

    // z: data (inputs and outputs from layers) 
    double** z;
    // delta: partial derivatives with respect to layer outputs z
    double** delta;

    // constructor and destructor
    Net(int num_layers);
    ~Net(); 

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

#ifdef USE_MPI
    // sync paramaters of all ranks in all layers to rank 0
    void sync();
#endif

};