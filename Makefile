###################################################################
#  Makefile for neural net
###################################################################

# compilers
CXX = g++
CC  = gcc

MPICXX = mpicxx
MPICC = mpicc

# flags
CFLAGS   = -O2
CXXFLAGS = -O2 -std=c++11
FFLAGS   = -O2
CPPFLAGS_MPI = -DUSE_MPI

# makefile targets
all : train-mnist-mpi

train-mnist-mpi : train-mnist.cpp
	$(MPICXX) $(CXXFLAGS) $(CPPFLAGS_MPI) train-mnist.cpp loadmnist.cpp layer.cpp net.cpp classifier.cpp mpiutil.cpp module.cpp -lm -o train-mnist

train-mnist : train-mnist.cpp
	$(CXX) $(CXXFLAGS) train-mnist.cpp loadmnist.cpp layer.cpp net.cpp classifier.cpp module.cpp -lm -o train-mnist


clean :
	\rm -f *.o *.out train-mnist temp

####### End of Makefile #######