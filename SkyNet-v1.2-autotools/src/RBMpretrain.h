#include "NeuralNetwork.h"
#include "TrainingData.h"

void arraystranstoweights(size_t nin, size_t nhid, float **vishid, float *hidbias, NeuralNetwork *nn, float startidx);
void pretrainnet(NeuralNetwork *nn, TrainingData &td, bool autoencoder, int _nepoch, float stdev, long seed);
