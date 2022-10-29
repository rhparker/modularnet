// trains classifier neural network on MNIST data

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>

#ifdef MPI
  #include "mpi.h"
  #include "mpiutil.h"
#endif

#include "classifier.h"
#include "loadmnist.h"

int main(int argc, char* argv[]) {

  int numprocs, myid;

#ifdef MPI
  // initialize MPI
  int ierr;
  ierr = MPI_Init(&argc, &argv);
  if (ierr != 0) {
    std::cerr << " error in MPI_Init = " << ierr << std::endl;
    return 1;
  }
  whoami(numprocs, myid);
#else
  numprocs = 1;
  myid = 0;
#endif

  // properties of MNIST data
  int input_size = 784;
  int output_size = 10;

  // loss and accuracy
  double train_acc, train_loss, test_acc, test_loss;

  // timers
  double batch_time, loss_time, timer;

  // arrays to be allocated for data and labels
  // training
  unsigned int train_cnt;
  double** train_data;
  unsigned int* train_labels;
  // test
  unsigned int test_cnt;
  double** test_data;
  unsigned int* test_labels;

  // load training data
  train_cnt = mnist_load("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte", 
        train_data, train_labels);
  if (train_cnt <= 0) {
    printf("An error occured loading training data: %d\n", train_cnt);
  } 
  else {
    if (myid == 0) printf("training image count: %d\n", train_cnt);
  }

  // load test data
  test_cnt = mnist_load("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte", 
        test_data, test_labels);
  if (test_cnt <= 0) {
    printf("An error occured loading test data: %d\n", test_cnt);
  } 
  else {
     if (myid == 0) printf("test image count: %d\n\n", test_cnt);
  }

  //
  // initialize network
  // 

  // std dev for weight initialization
  double sigma = 0.1;

  // // one layer linear
  // std::vector< std::vector <int > > config = {
  //   {LINEAR,784, 10},
  //   {SOFTMAX, 10}
  // };

  // // two layers linear
  // std::vector< std::vector <int > > config = {
  //   {LINEAR,784, 64},
  //   {RELU, 64},
  //   {LINEAR,64, 10},
  //   {SOFTMAX, 10}
  // };

  // // two layers linear with dropout
  // std::vector< std::vector <int > > config = {
  //   {LINEAR,784, 64},
  //   {RELU, 64},
  //   {DROPOUT, 64},
  //   {LINEAR,64, 10},
  //   {SOFTMAX, 10}
  // };

  // // one convolutional layer
  // std::vector< std::vector <int > > config = {
  //   {LINEAR,784, 64},
  //   {RELU, 64},
  //   {DROPOUT, 64},
  //   {CONV, 1,8,8,1,1,1},
  //   {MAXPOOL,1,8,8,1,1,2,2},
  //   {RELU, 16},
  //   {LINEAR,16, 10},
  //   {SOFTMAX, 10}
  // };

  // // one convolutional layer, multi-channel
  // std::vector< std::vector <int > > config = {
  //   {LINEAR,784, 64},
  //   {RELU, 64},
  //   {DROPOUT, 64},
  //   {CONV, 1,8,8,4,1,1},
  //   {MAXPOOL,4,8,8,1,1,2,2},
  //   {RELU, 64},
  //   {LINEAR,64, 10},
  //   {SOFTMAX, 10}
  // };

  // LeNet modernized
  std::vector< std::vector <int > > config = {
    {CONV, 1,28,28,6,2,2},
    {RELU, 4704},
    {DROPOUT, 4704},
    {MAXPOOL, 6,28,28,1,1,2,2},
    {CONV, 6,14,14,16,2,2},
    {RELU, 3136},
    {DROPOUT, 3136},
    {MAXPOOL, 16,14,14,1,1,2,2},
    {LINEAR,784,120},
    {RELU, 120},
    {DROPOUT,120},
    {LINEAR, 120,84},
    {RELU, 84},
    {DROPOUT,84},
    {LINEAR, 84,10},
    {SOFTMAX, 10}
  };

  Classifier C( config, sigma );
#ifdef MPI
  C.sync();
#endif

  // print network properties
  if (myid == 0) {
      C.properties();
    std::cout << std::endl;
  }
  
  //
  // run training epochs
  //

  // initial accuracy and cross entropy (pre-training)
  if (myid == 0) {
    std::cout 
      << std::left
      << std::setw(8)  << "epoch"
      << std::setw(20) << "cross-entropy loss"
      << std::setw(20) << "train accuracy"
      << std::setw(20) << "test accuracy" 
      << std::setw(20) << "loss time"
      << std::setw(20) << "training time"
      << std::endl;
  }

  // compute loss for training data and test data
  loss_time  = C.compute_loss(train_cnt, train_data, train_labels);
  train_acc = C.accuracy;
  train_loss = C.loss;
  loss_time += C.compute_loss(test_cnt, test_data, test_labels);
  test_acc = C.accuracy;

  if (myid == 0) {
    std::cout 
      << std::left
      << std::setw(8) << 0
      << std::setw(20) << train_loss
      << std::setw(20) << train_acc
      << std::setw(20) << test_acc
      << std::setw(20) << loss_time
      << std::endl;
  }

  int epochs = 5;
  int batch_size = 256;
  double learning_rate = 0.1;
  double weight_decay = 0;

  // run training epochs
  for (int i = 1; i <= epochs; i++) {

    batch_time = C.train_epoch(train_cnt, train_data, train_labels, 
          learning_rate, weight_decay, batch_size);

    loss_time  = C.compute_loss(train_cnt, train_data, train_labels);
    train_acc = C.accuracy;
    train_loss = C.loss;
    loss_time += C.compute_loss(test_cnt, test_data, test_labels);
    test_acc = C.accuracy;

    if (myid == 0) {
      std::cout 
        << std::left
        << std::setw(8) << i
        << std::setw(20) << train_loss
        << std::setw(20) << train_acc
        << std::setw(20) << test_acc
        << std::setw(20) << loss_time
        << std::setw(20) << batch_time
        << std::endl;
    }
  }

  // unallocate training data
  for (int i = 0; i < train_cnt; i++) {
    delete[] train_data[i];
  }
  delete[] train_data;
  delete[] train_labels;

  // unallocate test data
  for (int i = 0; i < test_cnt; i++) {
    delete[] test_data[i];
  }
  delete[] test_data;
  delete[] test_labels;

#ifdef MPI
  // finalize MPI
  ierr = MPI_Finalize();
#endif

  return 0;
}

