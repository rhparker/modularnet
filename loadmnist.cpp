#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "loadmnist.h"

/*
 * Load a unsigned int from raw data.
 * MSB first.
 */

static unsigned int mnist_bin_to_int(char *v)
{
	int i;
	unsigned int ret = 0;

	for (i = 0; i < 4; ++i) {
		ret <<= 8;
		ret |= (unsigned char)v[i];
	}

	return ret;
}

/*
 * MNIST dataset loader.
 * Returns 0 if successed.
 * Check comments for the return codes.
 */

unsigned int mnist_load(
	const char* image_filename,
	const char* label_filename,
	double** &data,
	unsigned int* &labels)
{
	int return_code = 0;
	int i;
	char tmp[4];
	size_t ret;

	unsigned int image_cnt, label_cnt;
	unsigned int image_dim[2];

	FILE *ifp = fopen(image_filename, "rb");
	FILE *lfp = fopen(label_filename, "rb");

	if (!ifp || !lfp) {
		return_code = -1; /* No such files */
		if (ifp) fclose(ifp);
		if (lfp) fclose(lfp);
		return return_code;
	}

	ret = fread(tmp, 1, 4, ifp);
	if (mnist_bin_to_int(tmp) != 2051) {
		return_code = -2; /* Not a valid image file */
		if (ifp) fclose(ifp);
		if (lfp) fclose(lfp);
		return return_code;
	}

	ret = fread(tmp, 1, 4, lfp);
	if (mnist_bin_to_int(tmp) != 2049) {
		return_code = -3; /* Not a valid label file */
		if (ifp) fclose(ifp);
		if (lfp) fclose(lfp);
		return return_code;
	}

	ret = fread(tmp, 1, 4, ifp);
	image_cnt = mnist_bin_to_int(tmp);

	ret = fread(tmp, 1, 4, lfp);
	label_cnt = mnist_bin_to_int(tmp);

	if (image_cnt != label_cnt) {
		return_code = -4; /* Element counts of 2 files mismatch */
		if (ifp) fclose(ifp);
		if (lfp) fclose(lfp);
		return return_code;
	}

	for (i = 0; i < 2; ++i) {
		ret = fread(tmp, 1, 4, ifp);
		image_dim[i] = mnist_bin_to_int(tmp);
	}

	if (image_dim[0] != 28 || image_dim[1] != 28) {
		return_code = -2; /* Not a valid image file */
		if (ifp) fclose(ifp);
		if (lfp) fclose(lfp);
		return return_code;
	}

	// allocate memory for data and labels
	data = new double*[image_cnt];
	labels = new unsigned int[image_cnt];
	for (i = 0; i < image_cnt; i++) {
		data[i] = new double[IMAGESIZE];
	}

	// read data
	for (i = 0; i < image_cnt; i++) {
		unsigned char read_data[IMAGESIZE];

		// read and store data
		ret = fread(read_data, 1, IMAGESIZE, ifp);
		for (int j = 0; j < IMAGESIZE; j++) {
			data[i][j] = read_data[j] / 255.0;
		}
		// read and store label
		ret = fread(tmp, 1, 1, lfp);
		labels[i] = tmp[0];
	}

	// close files
	if (ifp) fclose(ifp);
	if (lfp) fclose(lfp);

	// return number of images loaded
	return image_cnt;
}

