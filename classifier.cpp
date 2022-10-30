// multi-layer classifier

#include <iostream>
#include <cmath>
#include <algorithm>    
#include <vector>
#include <ctime> 
#include <random>
#include <cstdlib>

#include "classifier.h"

// timer
#ifdef USE_MPI
  #include "mpi.h"
  #include "mpiutil.h"
  #define get_time() MPI_Wtime()
#else
  #include <chrono>
  #define get_time() std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count()
#endif

// random number generator
int myrandom(int i) { 
  return std::rand() % i;
}

// argmax function 
// returns argmax of values, which has length len
unsigned int argmax(int len, double* values) {
  double current_max;
  unsigned int current_arg = 0;
  current_max = values[0];
  for (int i = 0; i < len; i++) {
    if (values[i] > current_max) {
      current_max = values[i];
      current_arg = i;
    }
  }
  return current_arg;
}

// constructor : passes through to Sequential
Classifier::Classifier(std::vector< std::vector <int> > config, double sigma) 
              : Sequential(config, sigma) {};

// destructor
Classifier::~Classifier() {};

// cross-entropy loss
double Classifier::compute_loss(int cnt, double** data, unsigned int* labels) {

  int numprocs, myid;
#ifdef USE_MPI
  whoami(numprocs, myid);
#else
  numprocs = 1;
  myid = 0;
#endif

  // start timer
  double start_time = get_time();

  // determine interval for this rank
  int is = ((int) (cnt/numprocs))*myid;
  int ie = ((int) (cnt/numprocs))*(myid+1);
  if (myid == numprocs-1) ie = cnt;

  // running total of number correct
  double my_correct = 0;
  double my_loss = 0;
  double total_correct;

  // we are not training
  int train = 0;

  // iterate over all samples
  for (int i = is; i < ie; i++) {
    // allocate output
    forward(data[i], train);
    // increment number correct if classification output from network 
    // (argmax of probability vector) matches label
    if ( argmax( layer_sizes[num_layers], z[num_layers] ) == labels[i] ) {
      my_correct += 1;
    }
    // update cross-entropy with sample
    my_loss -= log( z[num_layers][ labels[i] ] );
  }

#ifdef USE_MPI
  MPI_Allreduce(&my_correct, &total_correct, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&my_loss, &loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  total_correct = my_correct;
  loss = my_loss;
#endif
  // accuracy is total correct divide by count
  accuracy = total_correct/cnt;

  // return total time elapsed
  return get_time() - start_time;
}

// run one epoch of training using mini-batch stochastic gradient descent
// cnt:  number of data samples
// data: array containing data
// labels: labels corresponding to data
// lr: learning rate
// wd: weight decay parameter (unused for now)
// batch_size: size of each mini-batch
double Classifier::train_epoch(int cnt, double** data, unsigned int* labels, 
                                  double lr, double wd, unsigned int batch_size) {

  int numprocs, myid;
#ifdef USE_MPI
  whoami(numprocs, myid);
#else
  numprocs = 1;
  myid = 0;
#endif

  // start timer
  double start_time = get_time();

  // we are training
  int train = 1;

  // number of batches
  int this_batch_size;
  int num_batches = ceil( ((double) cnt) / batch_size );

  // allocate array for vector to feed back into backpropagation
  double* out = new double[ layer_sizes[num_layers] ];

  // randomly shuffle training samples
  int* order = new int[cnt];
  if (myid == 0) {
    std::srand ( unsigned ( std::time(0) ) );
    for (int i = 0; i < cnt; i++) {
      order[i] = i;
    }
    std::random_shuffle(order, order+cnt, myrandom);
  }

#ifdef USE_MPI
  MPI_Bcast(order, cnt, MPI_INT, 0, MPI_COMM_WORLD);
#endif

  // iterate over batches
  for (int b = 0; b < num_batches; b++) {

#ifdef PROGRESS
    if (myid == 0) printf("Batch %d out of %d\n", b, num_batches);
#endif

    // clear partial derivatives
    clear_partial();

    // if not enough samples left for a full batch, use what we have left
    if (b == num_batches-1) {
      this_batch_size = cnt - (num_batches-1)*batch_size;
    }
    else {
      this_batch_size = batch_size;
    }
    
    // compute contributions to paritals from each training sample in batch
    // determine this processor's interval
    int is = ((int) (this_batch_size/numprocs))*myid;
    int ie = ((int) (this_batch_size/numprocs))*(myid+1);
    if (myid == numprocs-1) ie = this_batch_size;

    for (int i = is; i < ie; i++) {
      // index of training sample in data array
      int index = order[ b*batch_size + i ];

      // step 1: forward propagation on training sample
      forward(data[index], train);

      // step 2: backward propagation
      // compute output, which we feed back; put in last component of delta
      unsigned int yj;
      for (int j = 0; j < layer_sizes[num_layers]; j++) {
        yj = (j == labels[index]);
        out[j] = z[num_layers][j] - yj;
      }
      backward(out);

      // step 3: accumulate parameter partials using results of backpropagation
      partial_param();
    }

    // step 4: now that we have finished with our mini-batch, update net parameters
    // using accumulated partial derivatives for entire mini-batch
    // (stochastic gradient descent)
    update_param(lr, this_batch_size);
  }

  delete[] order;
  
  // return total time
  return get_time() - start_time;
}

