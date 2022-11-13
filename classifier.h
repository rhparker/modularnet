// Classifier with all fully connected layers
// all but last layer use sigmoid activation function
// last layer uses softmax to get probability vector
// loss function is cross-entropy

#include "net.h"

#ifndef _CLASSIFIER
#define _CLASSIFIER

class Classifier : public Net {
  public:
    // stores current training accuracy, cross-entropy loss
    double accuracy;
    double loss;

    // constructor and destructor
    Classifier(std::vector< std::vector <int> > config);
    ~Classifier(); 

    // compute cross-entropy loss and accuracy
    double compute_loss(int cnt, double** data, unsigned int* labels);

    // train for one epoch
    double train_epoch(int cnt, double** data, unsigned int* labels, 
                          double lr, double wd, unsigned int batch_size);
};

#endif