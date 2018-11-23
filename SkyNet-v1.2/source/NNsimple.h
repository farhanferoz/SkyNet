#ifndef __NNSIMPLE_H__
#define __NNSIMPLE_H__ 1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
	int nlayers;			// number of layers
	int nin;				// number of inputs
	int nout;				// number of outputs
	int *nnodes;			// number of non-recurrent nodes per layer
	int *rnodes;			// number of recurrent nodes per layer
	int *linear;			// is layer i linear? 0=no, 1=yes
	int recurrent;			// recurrent network flag
	int norbias;			// recurrent initial bias
	int nweights;			// total number of weights
	int nnweights;			// total number of non-recurrent weights
	int totnnodes;			// total number of non-recurrent nodes
	int totrnodes;			// total number of recurrent nodes
	float *weights;			// array of weights
	float ***w;				// 3D array of arranged weights
	int classnet;			// classification network flag
	float *scale;			// scaling factor for otrans
	float *offset;			// offset for otrans
	int otrans;				// otrans flag
} NetworkVariables;

// reads in network
void readNNsimple(const char *filename, NetworkVariables *net);

// reads in output transformation
void readOTrans(const char *filename, NetworkVariables *net);

// arranges weights from 1D to 3D
void arrangeweights(NetworkVariables *net);

// performs a forward evaluation
void forwardOne(int ntime, float *in, float *out, NetworkVariables *net);

// free allocated arrays
void clearNNsimple(NetworkVariables *net);

// sigmoid function for use in forward
float sigmoid(float x, int flag);

#endif
