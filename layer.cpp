#include <random>
#include <iostream>
#include <cmath>
#include <stdio.h>

#ifdef USE_MPI
  #include "mpi.h"
  #include "mpiutil.h"
#endif

#include "layer.h"
#include "mpiutil.h"


// macro for getting weight(i,j) from weight array stored as 1D flattened parameter array
#define weight(p,i,j) p[ i*inputs + j ]
// macro for getting bias(i) from bias array stored at end of 1D flattened parameter array
#define bias(p,i) p[ num_weights + i ]

// macro for generic index in 2D
#define idx(n,i,j) (n*i + j)
// macro for generic index in 3D
#define idx3(m,n,c,i,j) (m*n*c + n*i + j)
// macro for convolutional kernel
#define ker3(p,co,ci,i,j) p[ ( (2*ker_m+1)*(2*ker_n+1)*input_c*co + (2*ker_m+1)*(2*ker_n+1)*ci + (2*ker_n+1)*i + j ) ] 

//
// abstract layer class
//

// constructor and destructor
Layer::Layer(int inputs, int outputs) : 
    inputs(inputs), outputs(outputs), pars(0), train(0) {};
Layer::~Layer() {}; 
void Layer::print_params() {};
void Layer::properties() {};
void Layer::partial_param(double* in, double* delta) {};

// clear accumulated partial derivaties 
void Layer::clear_partial() {
  for (int i = 0; i < pars; i++) {
    partial[i] = 0;
  }
}

// update parameters using accumulated partial derivatives
void Layer::update_param(double lr, int batch_size) {
#ifdef USE_MPI
  double* update = new double[pars];
  MPI_Allreduce(partial, update, pars, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (int i = 0; i < pars; i++) {
    param[i] -= (lr/batch_size)*update[i];
  }
  delete[] update;
#else
  for (int i = 0; i < pars; i++) {
    param[i] -= (lr/batch_size)*partial[i];
  }
#endif
}

// syncs layer in all ranks to rank 0
#ifdef USE_MPI
void Layer::sync() {
  int numprocs, myid;
  whoami(numprocs, myid);
  if (pars > 0 && numprocs > 1) {
    MPI_Bcast(param, pars, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
  }
}
#endif

//
// fully connected linear layer
//

// constructor and destructor
Linear::Linear(std::vector<int>  config, double sigma) 
      : Layer(config[1], config[2]) {

  // number of parameters (weights and biases)
  // parameters is a linear vector: weights first (as linear vector), then biases
  num_weights = inputs*outputs;
  pars = num_weights + outputs;

  param   = new double[pars];
  partial = new double[pars];

  // random device initialization
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  // instance of class std::normal_distribution with mean 0, std dev sigma
  std::normal_distribution<double> d(0, sigma); 

  // initialize weights to normal(0, sigma)
  for (int i = 0; i < num_weights; i++) {
    param[i] = d(gen);
  }
  // initialize biases to 0
  for (int i = num_weights; i < pars; i++) {
    param[i] = 0;
  }
}

// destructor
Linear::~Linear() {
  delete[] param;
  delete[] partial;
}

// print weights and biases
void Linear::print_params() {
  // weights
  std::cout << "Weights: " << std::endl;
  // iterate over rows
  for (int i = 0; i < outputs; i++) {
    // go across each row, i.e. iterate over columns
    for (int j = 0; j < inputs; j++) {
      std::cout << weight(param,i,j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  // biases
  std::cout << "Biases: " << std::endl;
  for (int i = 0; i < outputs; i++) {
    std::cout << bias(param,i) << " ";
  }
  std::cout << std::endl << std::endl;
}

// print properties
void Linear::properties() {
  std::cout << "Linear layer: ";
  std::cout << inputs << " inputs, ";
  std::cout << outputs << " outputs" << std::endl;
}

// forward propagation
void Linear::forward(double* in, double* out) {
  // iterate over rows
#pragma omp parallel for
  for (int i = 0; i < outputs; i++) {
    out[i] = bias(param,i);
    // iterate over columns
    for (int j = 0; j < inputs; j++) {
      out[i] += weight(param,i,j) * in[j];
    }
  }
}

// backward propagation
void Linear::backward(double* in, double* out, double* delta) {
  for (int i = 0; i < inputs; i++) {
    delta[i] = 0;
    for (int j = 0; j < outputs; j++ ) {
      delta[i] += out[j] * weight(param,j,i);
    }
  }
}

// compute partial derivatives with respect to parameters
void Linear::partial_param(double* in, double* delta) {
  // bias partials
  for (int j = 0; j < outputs; j++) {
    bias(partial,j) += delta[j];
  }
  // weight partials
  for (int j = 0; j < outputs; j++) {
    for (int k = 0; k < inputs; k++) {
      weight(partial,j,k) += delta[j]*in[k];
    }
  }
}

//
// sigmoid activation layer
//

// sigmoid activation function (single element)
inline double sig(double x) {
  return 1.0/(1.0 + exp(-x));
}

// derivative of sigmoid (single element)
inline double d_sig(double x) {
  return sig(x)*(1.0 - sig(x));
}

// constructor and destructor
Sigmoid::Sigmoid(std::vector<int> config) : Layer(config[1], config[1]) {};

Sigmoid::~Sigmoid() {};

// print properties
void Sigmoid::properties() {
  std::cout << "Sigmoid activation layer: inputs/outputs " << inputs << std::endl;
}

// forward propagation
void Sigmoid::forward(double* in, double* out) {
  // iterate over inputs
  for (int i = 0; i < inputs; i++) {
    out[i] = sig(in[i]);
  }
}

// backward propagation
void Sigmoid::backward(double* in, double* out, double* delta) {
  for (int i = 0; i < inputs; i++) {
    delta[i] = d_sig(in[i]) * out[i];
  }
}

//
// sigmoid activation layer
//

// rectified linear unit (single element)
inline double rec(double x) {
  return fmax(x, 0.0);
}

// derivative of sigmoid (single element)
inline double d_rec(double x) {
  return (x <= 0) ? 0.0 : 1.0;
}

// constructor and destructor
ReLU::ReLU(std::vector<int> config) : Layer(config[1], config[1]) {};

ReLU::~ReLU() {};

// print properties
void ReLU::properties() {
  std::cout << "ReLU activation layer: inputs/outputs " << inputs << std::endl;
}

// forward propagation
void ReLU::forward(double* in, double* out) {
  // iterate over inputs
  for (int i = 0; i < inputs; i++) {
    out[i] = rec(in[i]);
  }
}

// backward propagation
void ReLU::backward(double* in, double* out, double* delta) {
  for (int i = 0; i < inputs; i++) {
    delta[i] = d_rec(in[i]) * out[i];
  }
}


//
// softmax activation layer
//

// constructor and destructor
Softmax::Softmax(std::vector<int> config) : Layer(config[1], config[1]) {};

Softmax::~Softmax() {};

// print properties
void Softmax::properties() {
  std::cout << "Softmax activation layer: inputs/outputs " << inputs << std::endl;
}

// forward propagation
void Softmax::forward(double* in, double* out) {
  double normalizer = 0.0;
  for (int i = 0; i < inputs; i++) {
    out[i] = exp(in[i]);
    normalizer += out[i];
  }
  // divide each entry by normalizer
  for (int i = 0; i < inputs; i++) {
    out[i] /= normalizer;
  }
}

// backward propagation
void Softmax::backward(double* in, double* out, double* delta) {
  for (int i = 0; i < inputs; i++) {
    delta[i] = out[i];
  }
}

//
// dropout layer
//

// constructor and destructor
Dropout::Dropout(std::vector<int> config) :
    Layer(config[1], config[1]),
    gen(rd()), d(std::uniform_real_distribution<>(0,1)), 
    drop_prob(0.25) {

  // allocate data for mask
  mask = new int[inputs];
};

Dropout::~Dropout() {
  delete[] mask;
};

void Dropout::set_dropout(double prob) {
  drop_prob = prob;
}

// print properties
void Dropout::properties() {
  std::cout << "Dropout layer: dropout probability " << drop_prob << std::endl;
}

// forward propagation
void Dropout::forward(double* in, double* out) {
  // pass through if not training
  if (train == 0) {
    for (int i = 0; i < inputs; i++) {
      out[i] = in[i];
    }
  }
  // if we are training, the do dropout
  else {
    // generate random mask, store in mask
    for (int i = 0; i < outputs; i++) {
      mask[i] = ( d(gen) > drop_prob ) ? 1 : 0;
    }
    // apply mask
    for (int i = 0; i < inputs; i++) {
      out[i] = in[i] * mask[i] / (1-drop_prob);
    }
  }
}

// backward propagation
void Dropout::backward(double* in, double* out, double* delta) {
  for (int i = 0; i < inputs; i++) {
    delta[i] = out[i] * mask[i] / (1-drop_prob);
  }
}

//
// Max pool with 2D window, multiple layers
//

// constructor
Maxpool::Maxpool(std::vector<int> config) :
      Layer(config[1]*config[2]*config[3], 0),
      channels(config[1]),
      input_m(config[2]), input_n(config[3]), 
      window_m(config[4]), window_n(config[5]), 
      stride_m(config[6]), stride_n(config[7]) {

  // output size is channels * ceil(input_m / sm) * ceil(input_n/sn)
  output_m = (int) ceil ( ((double)input_m) / ((double) stride_m) ); 
  output_n = (int) ceil ( ((double)input_n) / ((double) stride_n) ); 
  outputs = channels * output_m * output_n;

  // allocate argmax 
  argmax = new int[outputs];
}

// destructor
Maxpool::~Maxpool() {
  delete[] argmax;
};

// print properties
void Maxpool::properties() {
  printf("Max pool layer: inputs %d (%d channels, %d x %d), outputs: %d (%d channels, %d x %d), window (%d x %d), stride (%d x %d)\n",
    inputs, channels, input_m, input_n,
    outputs, channels, output_m, output_n,
    2*window_m+1, 2*window_n+1, stride_m, stride_n);
};

// forward propagation
void Maxpool::forward(double* in, double* out) {
  int row, col, center;
  // do one channel at at time
  for (int c = 0; c < channels; c++) {
    // row of output
    for (int i = 0; i < output_m; i++) {
      // column of output
      for (int j = 0; j < output_n; j++) {
        // initialize current max to center of window
        center = idx3( input_m, input_n, c, stride_m*i, stride_n*j );
        out[ idx3(output_m, output_n, c, i, j) ] = in[ center ];
        // if training, save argmax
        if (train == 1) {
          argmax[ idx3(output_m, output_n, c, i, j) ] = center;
        }
        // row of window
        for (int wi = 0; wi < (2*window_m+1); wi++ ) {
          // column of window
          for (int wj = 0; wj < (2*window_n+1); wj++) {
            row = stride_m*i + wi - window_m;
            col = stride_n*j + wj - window_n;
            // only grab elements of input if we are in bounds
            // this effectively pads with zeros
            if (row >= 0 && row < input_m && col >= 0 && col < input_n) {
              // update max if we are greater than current max
              if ( in[ idx3(input_m, input_n, c, row, col) ] > out[ idx3(output_m, output_n, c, i, j) ] ) {
                out[ idx3(output_m, output_n, c, i, j) ] = in[ idx3( input_m, input_n, c, row, col) ];
                // if training, update argmax as well
                if (train == 1) {
                  // std::cout << idx3(input_m, input_n, c, row, col) << std::endl;
                  argmax[ idx3(output_m, output_n, c, i, j) ] = idx3(input_m, input_n, c, row, col);
                }
              }
            }
          }
        }
      }
    }
  }
}

//  backward propagation
void Maxpool::backward(double* in, double* out, double* delta) {
  // initialize all delta to 0
  for (int i = 0; i < inputs; i++) {
    delta[i] = 0;
  }
  // add outputs to input corresponding to argmax
  for (int i = 0; i < outputs; i++) {
    delta[ argmax[i] ] += out[i];
  }
}

//
// convolution with 2D kernel, multiple input and output layers
//

// constructor
// for now enforce stride 1
Conv::Conv(std::vector<int> config, double sigma) : 
      Layer(config[1]*config[2]*config[3], 0), 
      input_c(config[1]), input_m(config[2]), input_n(config[3]), 
      output_c(config[4]),
      ker_m(config[5]), ker_n(config[6]), 
      stride_m(1), stride_n(1) {

  // output for each channel is ceil(input_m / sm) x ceil(input_n/sn)
  output_m = (int) ceil ( ((double)input_m) / ((double) stride_m) ); 
  output_n = (int) ceil ( ((double)input_n) / ((double) stride_n) ); 
  outputs = output_c * output_m * output_n;

  // number of weight parameters ( 2km+1 x 2kn+1 ) * input_c * output_c
  num_weights = output_c * input_c * (2*ker_m+1) * (2*ker_n+1);

  // total parameters is weights + biases
  pars = num_weights + outputs;
  param   = new double[pars];
  partial = new double[pars];

  // random device initialization
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  // instance of class std::normal_distribution with mean 0, std dev sigma
  std::normal_distribution<double> d(0, sigma); 

  // initialize parameters to normal(0, sigma)
  for (int i = 0; i < pars; i++) {
    param[i] = d(gen);
  }
  // initialize biases to 0
  for (int i = num_weights; i < pars; i++) {
    param[i] = 0; 
  }
}

// destructor
Conv::~Conv() {
  delete[] param;
  delete[] partial;
}

// print properties
void Conv::properties() {
  printf("Convolution layer: inputs %d (%d channels, %d x %d), outputs %d (%d channels, %d x %d), kernel (%d x %d), stride (%d x %d)\n",
    inputs, input_c, input_m, input_n,
    outputs, output_c, output_m, output_n,
    2*ker_m+1, 2*ker_n+1, stride_m, stride_n);
}

void Conv::print_params() {
  std::cout << "Kernel: " << std::endl;
  // iterate over outputs
  for (int co = 0; co < output_c; co++) {
  // iterate over inputs
    for (int ci = 0; ci < input_c; ci++) {
      printf("Output layer: %d,  Input layer: %d \n", co, ci);
      // iterate over rows
      for (int i = 0; i < (2*ker_m+1); i++) {
        // iterate over columns
        for (int j = 0; j < (2*ker_n+1); j++) {
          std::cout << ker3(param,co,ci,i,j) << "  ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
  // biases
  std::cout << "Biases: " << std::endl;
  for (int i = 0; i < outputs; i++) {
    std::cout << bias(param,i) << " ";
  }
  std::cout << std::endl << std::endl;
}


// forward propagation
void Conv::forward(double* in, double* out) {
  // initialize outputs to biases
  for (int i = 0; i < outputs; i++) {
    out[i] = bias(param, i);
  }
  int row, col;
  // iterate over output channels
  for (int co = 0; co < output_c; co++) {
  // iterate over input channels
    for (int ci = 0; ci < input_c; ci++) {
      // row of output
      for (int i = 0; i < output_m; i++) {
        // column of output
        for (int j = 0; j < output_n; j++) {
          // row of kernel
          for (int ki = 0; ki < (2*ker_m+1); ki++ ) {
            // column of kernel
            for (int kj = 0; kj < (2*ker_n+1); kj++) {
              // index of row and column we need in input
              row = stride_m*i + ki - ker_m;
              col = stride_n*j + kj - ker_n;
              // if we are in bounds, then multiply by appropriate kernel element
              // to perform convolution; this effectively pads with zeros
              if (row >= 0 && row < input_m && col >= 0 && col < input_n) {
                out[ idx3(output_m, output_n, co, i, j) ] += 
                  ker3(param,co,ci,ki,kj) *
                  in[ idx3(input_m, input_n, ci, row, col) ];
              }
            }
          }
        }
      }
    }
  }
}

// backward propagation
// for now enforce stride 1
void Conv::backward(double* in, double* out, double* delta) {
  int row, col;
  // initialize deltas to 0
  for (int i = 0; i < inputs; i++) {
    delta[i] = 0;
  }
  // iterate over input channels
  for (int ci = 0; ci < input_c; ci++) {
    // iterate over output channels
    for (int co = 0; co < output_c; co++) {
      // row of input (delta is same size as input)
      for (int i = 0; i < input_m; i++) {
        // column of input (delta is same size as input)
        for (int j = 0; j < input_n; j++) {
          // row of kernel
          for (int ki = 0; ki < (2*ker_m+1); ki++ ) {
            // column of kernel
            for (int kj = 0; kj < (2*ker_n+1); kj++) {
              // index of row and column we need from output
              row = i + ki - ker_m;
              col = j + kj - ker_n;
              // if we are in bounds, then multiply by appropriate kernel element
              // for backpropagation we flip the kernel (horiz and vert)
              if (row >= 0 && row < output_m && col >= 0 && col < output_n) {
                delta[ idx3(input_m, input_n, ci, i, j) ] += 
                  ker3(param,co,ci,(2*ker_m-ki), (2*ker_n-kj) ) *
                  out[ idx3(output_m, output_n, co, row, col) ];
              }
            }
          }
        }
      }
    }
  }
}

// compute partial derivative of loss with respect to parmeters 
void Conv::partial_param(double* in, double* delta) {
  int row, col;
  // bias partials
  for (int i = 0; i < outputs; i++) {
    bias(partial,i) += delta[i];
  }
  // iterate over outputs
  for (int co = 0; co < output_c; co++) {
  // iterate over inputs
    for (int ci = 0; ci < input_c; ci++) {
      // row of kernel
      for (int ki = 0; ki < (2*ker_m+1); ki++ ) {
        // column of kernel
        for (int kj = 0; kj < (2*ker_n+1); kj++) {
          // row of output
          for (int i = 0; i < output_m; i++) {
            // column of output
            for (int j = 0; j < output_n; j++) {
              row = stride_m*i + ki - ker_m;
              col = stride_n*j + kj - ker_n;
              if (row >= 0 && row < input_m && col >= 0 && col < input_n) {
                ker3(partial,co,ci,ki,kj) += 
                  delta[ idx3(output_m, output_n, co, i, j) ] *
                  in[ idx3(output_m, input_n, ci, row, col) ];
              }
            }
          }
        }
      }
    }
  }
}



