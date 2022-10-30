// trains classifier neural network on MNIST data

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <vector>

#ifdef USE_MPI
  #include "mpi.h"
  #include "mpiutil.h"
#endif

#include "classifier.h"
#include "loadmnist.h"

#define FASHION

#ifdef FASHION
  #define TRAIN_IMAGES "fashion/train-images-idx3-ubyte"
  #define TRAIN_LABELS "fashion/train-labels-idx1-ubyte"
  #define TEST_IMAGES  "fashion/t10k-images-idx3-ubyte"
  #define TEST_LABELS  "fashion/t10k-labels-idx1-ubyte"
#else
  #define TRAIN_IMAGES "mnist/train-images-idx3-ubyte"
  #define TRAIN_LABELS "mnist/train-labels-idx1-ubyte"
  #define TEST_IMAGES  "mnist/t10k-images-idx3-ubyte"
  #define TEST_LABELS  "mnist/t10k-labels-idx1-ubyte"
#endif

int main(int argc, char* argv[]) {

  int numprocs, myid;

#ifdef USE_MPI
  // initialize MPI
  int ierr;
  ierr = MPI_Init(&argc, &argv);
  if (ierr != 0) {
    std::cerr << " error in MPI_Init = " << ierr << std::endl;
    return 1;
  }
  whoami(numprocs, myid);
  if (myid == 0) std::cout << "Number of MPI ranks: " << numprocs << std::endl << std::endl;
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
  double total_time = 0;

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
  train_cnt = mnist_load(TRAIN_IMAGES, TRAIN_LABELS, train_data, train_labels);
  if (train_cnt <= 0) {
    printf("An error occured loading training data: %d\n", train_cnt);
  } 
  else {
    if (myid == 0) printf("training image count: %d\n", train_cnt);
  }

  // load test data
  test_cnt = mnist_load(TEST_IMAGES, TEST_LABELS, test_data, test_labels);
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

  // learning rate
  double learning_rate;

  // // one layer linear
  // std::vector< std::vector <int > > config = {
  //   {LINEAR,784, 10},
  //   {SOFTMAX, 10}
  // };
  // learning_rate = 0.1;

  // // two layers linear
  // std::vector< std::vector <int > > config = {
  //   {LINEAR,784, 64},
  //   {RELU, 64},
  //   {LINEAR,64, 10},
  //   {SOFTMAX, 10}
  // };
  // learning_rate = 0.1;

  // // two layers linear with dropout
  // std::vector< std::vector <int > > config = {
  //   {LINEAR,784, 64},
  //   {RELU, 64},
  //   {DROPOUT, 64},
  //   {LINEAR,64, 10},
  //   {SOFTMAX, 10}
  // };
  // learning_rate = 0.1;

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
  // learning_rate = 0.1;

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
  // learning_rate = 0.1;

  // // LeNet modernized
  // std::vector< std::vector <int > > config = {
  //   {CONV, 1,28,28,6,2,2},
  //   {RELU, 4704},
  //   {MAXPOOL, 6,28,28,1,1,2,2},
  //   {CONV, 6,14,14,16,2,2},
  //   {RELU, 3136},
  //   {MAXPOOL, 16,14,14,1,1,2,2},
  //   {LINEAR,784,120},
  //   {RELU, 120},
  //   {DROPOUT,120},
  //   {LINEAR, 120,84},
  //   {RELU, 84},
  //   {DROPOUT,84},
  //   {LINEAR, 84,10},
  //   {SOFTMAX, 10}
  // };
  // learning_rate = 0.1;

  // mini-Alexnet, modified for MNIST
  std::vector< std::vector <int > > config = {
    {CONV,    1,28,28,32,2,2 },
    {RELU,    25088 },
    {MAXPOOL, 32,28,28,1,1,2,2},
    {CONV,    32,14,14,64,1,1},
    {RELU,    12544},
    {CONV,    64,14,14,64,1,1},
    {RELU,    12544},
    {CONV,    64,14,14,32,1,1},
    {RELU,    6272},
    {MAXPOOL, 32,14,14,1,1,2,2},
    {LINEAR, 1568, 1024},
    {RELU, 1024},
    {DROPOUT, 1024},
    {LINEAR, 1024, 1024},
    {RELU, 1024},
    {DROPOUT, 1024},
    {LINEAR, 1024, 10},
    {SOFTMAX, 10}
  };
  learning_rate = 0.02;

  Classifier C( config, sigma );
#ifdef USE_MPI
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
  total_time += loss_time;

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

  int epochs = 10;
  int batch_size = 256;
  double weight_decay = 0;

  // run training epochs
  for (int i = 1; i <= epochs; i++) {

    batch_time = C.train_epoch(train_cnt, train_data, train_labels, 
          learning_rate, weight_decay, batch_size);
    total_time += batch_time;

    loss_time  = C.compute_loss(train_cnt, train_data, train_labels);
    train_acc = C.accuracy;
    train_loss = C.loss;
    loss_time += C.compute_loss(test_cnt, test_data, test_labels);
    total_time += loss_time;

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

  // print total time
  if (myid == 0) {
    std::cout << "Total time: " << total_time << std::endl;
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

#ifdef USE_MPI
  // finalize MPI
  ierr = MPI_Finalize();
#endif

  return 0;
}

