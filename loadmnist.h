// MNIST data loader 
// original version by Nuri Park - https://github.com/projectgalateia/mnist
// modified to use c++ new for allocation and to return images as a single array
// loads data as double


// MNIST image size
#define IMAGESIZE 784

// data loader
unsigned int mnist_load(
	const char* image_filename,
	const char* label_filename,
	double** &data,
	unsigned int* &labels);