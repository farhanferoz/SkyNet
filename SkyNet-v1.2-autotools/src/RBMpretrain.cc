#include "RBMpretrain.h"

void arraystranstoweights(	size_t nin, 		// second index of vishid^t
				size_t nhid, 		// first index of vishid^t
				float **vishid, 	// dimensions [nhid][nin], hid-in weights
				float *hidbias,		// dimension[nhid], bias on hid units
				NeuralNetwork *nn,	// pointer to nn
				float startidx)		// index of first weight in nn to be set
{
	size_t k = startidx;
	
	for( size_t i = 0; i < nhid; i++ )
	{
		for( size_t j = 0; j < nin+1; j++ )
		{
			float w = j != nin ? vishid[i][j] : hidbias[i];
			nn->setweights(k, 1, &w);
			k++;
		}
	}
}



void pretrainnet(		NeuralNetwork *nn, 	// pointer to nn
				TrainingData &td, 	// training data
				bool autoencoder, 	// pretrain an autoencoder?
				int _nepoch,		// no. of training epoch, default value of 10 is used if nepoch <= 0
				float stdev, 		// standard deviation of initial (non-bias) weights
				long seed)		// seed for random no. generator
{
	// no. of layers to pre-train
	size_t depth = autoencoder ? nn->nlayers/2 : nn->nlayers-2;
	std::vector <size_t> nhid;
	for( size_t i = 0; i < depth; i++ ) nhid.push_back(nn->nnodes[i+1]-1);
	
	float *inputs;
	
	for( size_t n = 0; n < depth; n++ )
	{
		// initialize the NN
	
		size_t nlayers = 2;
		size_t nnodes[nlayers];
		nnodes[0] = n == 0 ? td.nin+1 : nhid[n-1]+1;			// nodes (including biases) in the input layer
		nnodes[1] = nhid[n];

		RBM rbm(nnodes, nn->linear[n+1], td.ndata, _nepoch);
	
		if( n == 0 ) inputs = td.inputs;
		
		float mean[] = {0E0, 0E0, 0E0};		// means of the normal distributions of initial weights for in-hid, hid-bias, in-bias respectively
		float sd[] = {stdev, 0E0, 0E0};		// standard deviations of the normal distributions of initial weights for in-hid, hid-bias, in-bias respectively
		rbm.setnormalweights(mean, sd, seed);	// set the intial weights
		
		rbm.pretrain(inputs, seed);
		
		if( n > 0 ) delete [] inputs;
		
		if( n < depth-1 )
		{
			inputs = new float [td.ndata*nhid[n]];
			for( size_t i = 0; i < td.ndata*nhid[n]; i++ ) inputs[i] = rbm.batchposhidprob[i];
		}
		
		// copy the RBM weights to full NN weights
		float weights[rbm.nweights];
		rbm.getweights(weights);
		nn->setweights(nn->ncumweights[n], nn->ncumweights[n+1]-nn->ncumweights[n], weights);
		
		if( autoencoder )
		{
			size_t nin = nnodes[0]-1, nout = nnodes[1];
			float **vishid = new float* [nin]; for( size_t i = 0; i < nin; i++ ) vishid[i] = new float [nout];
			float *hidbias = new float [nout];
			
			rbm.weightstoarrays(vishid, hidbias);
			size_t startidx = nn->ncumweights[nn->nlayers-2-n];
			arraystranstoweights(nout, nin, vishid, rbm.visbias, nn, startidx);
			
			for( size_t i = 0; i < nin; i++ ) delete [] vishid[i]; delete [] vishid;
			delete [] hidbias;
		}
	}
	
	// initialize the weights in the final layer for non-autoencoders
	if( !autoencoder )
	{
		size_t nout = nn->nnodes[nn->nlayers-2]-1, nin = nn->nnodes[nn->nlayers-1];
		float **vishid = new float* [nin]; for( size_t i = 0; i < nin; i++ ) vishid[i] = new float [nout];
		float *visbias = new float [nin];
		
		for( size_t i = 0; i < nin; i++ )
		{
			visbias[i] = 0E0;
			for( size_t j = 0; j < nout; j++ ) vishid[i][j] = gasdev(&seed)*stdev;
		}

		size_t startidx = nn->ncumweights[nn->nlayers-2];
		arraystranstoweights(nout, nin, vishid, visbias, nn, startidx);
			
		for( size_t i = 0; i < nin; i++ ) delete [] vishid[i]; delete [] vishid;
		delete [] visbias;
	}
	
	nn->arrangeweights();
	//nn->weightsAdjustActivation(autoencoder);
	//nn->inversearrangeweights();
}
