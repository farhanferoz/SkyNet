#include "NeuralNetwork.h"
#include <fstream>
#include <cblas.h>

#include "myrand.cc"

typedef axisAlignedTransform trans_t;
//typedef histEqualTransform2 trans_t;

void ErrorExit(std::string msg)
{
	std::cerr << msg << "\n";
	abort();
}


		
void arraystranstoweights(	size_t nin, 	// second index of vishid^t
				size_t nhid, 	// first index of vishid^t
				float **vishid, // dimensions [nhid][nin], hid-in weights
				float *hidbias,	// dimension[nhid], bias on hid units
				float *weights)	// output weight array
{
	int k = -1;
	
	for( size_t i = 0; i < nhid; i++ )
		for( size_t j = 0; j < nin+1; j++ )
			weights[++k] = j != nin ? vishid[i][j] : hidbias[i];
}



int main(int argc, char *argv[])
{
	int myid = 0, ncpus = 1;
	
#ifdef PARALLEL
 	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&ncpus);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#endif
	
	bool classnet = true;
	bool recurrent = false;
	bool whiten = true;
	bool autoencoder = true;
	bool train = false;
	
	float stdev = 0.1E0;	// standard deviation of initial set of (non-bias) weights
	
	
	
	// initialize the training data
	
	TrainingData wtd;
	{
		std::string datafile;
		if( classnet )
			datafile = "data/cdata_train.txt";
		else
			datafile = "data/MNIST/ORIG/MNIST_AE_train.txt";
			//datafile = "data/MNIST/ORIG/MNIST_RBMAE_test.txt";
		TrainingData td(datafile, classnet);
		float sigma = 0.01;
		for( size_t i = 0; i < td.cumoutsets[td.ndata]*td.nout; i++ ) td.acc[i] = 1E0/sigma;
		
		
		
		// whiten the data
		
		trans_t itrans, otrans;
		if (whiten)
		{
			if( classnet )
				td.generateWhiteData(&itrans, wtd, 1);
			else
				td.generateWhiteData(&itrans, &otrans, wtd);
			
			//std::cerr << itrans.scale << "\t" << itrans.off << "\n";
			//std::cerr << otrans.scale << "\t" << otrans.off << "\n";
		}
		else
		{
			wtd = td;
		}
	}
	
	
	
	// set up the random no. generator
	long seed = rand();
	/*seed += 123321;
	seed *= -1;*/
	//seed = 1;
	
	
	std::vector <size_t> nhid;
	nhid.push_back(1000);
	nhid.push_back(500);
	nhid.push_back(250);
	nhid.push_back(30);
	
	// no. of layers to pre-train
	size_t depth = nhid.size();
	
	
	
	// create the full NN
	size_t NNnlayers = autoencoder ? nhid.size()*2+1 : nhid.size()+2;
	int ncumweights[NNnlayers];	// cumulative weights for each layer
	size_t NNnnodes[NNnlayers];
	bool linear[NNnlayers];
	NNnnodes[0] = wtd.nin+1;
	linear[0] = true;
	ncumweights[0] = 0;
	NNnnodes[NNnlayers-1] = wtd.nout;
	for( size_t i = 0; i < nhid.size(); i++ )
	{
		NNnnodes[i+1] = nhid[i]+1;
		ncumweights[i+1] = ncumweights[i] + (NNnnodes[i+1]-1)*NNnnodes[i]; // no. of weights in this layer = nhid*(nin+1)
	}
	if( autoencoder )
	{
		size_t j = nhid.size();
		for( int i = nhid.size()-2; i >= 0; i-- )
		{
			NNnnodes[++j] = nhid[i]+1;
			ncumweights[j] = ncumweights[j-1] + (NNnnodes[j]-1)*NNnnodes[j-1]; // no. of weights in this layer = nhid*(nin+1)
		}
		
		for( size_t i = 1; i < NNnlayers; i++ ) linear[i] = false;
		linear[nhid.size()] = true;
	}
	else
	{
		for( size_t i = 1; i < NNnlayers-1; i++ ) linear[i] = false;
		linear[NNnlayers-1] = true;
	}
	ncumweights[NNnlayers-1] = ncumweights[NNnlayers-2] + NNnnodes[NNnlayers-1]*NNnnodes[NNnlayers-2]; // no. of weights in final/output layer = nout*nin
	FeedForwardNeuralNetwork nn(NNnlayers, NNnnodes, recurrent, linear);
	float weights[nn.nweights];
	
	if( !train )
	{
//std::ifstream wfile("RBMweights.txt"), bfile("RBMbiases.txt");
		nn.read("SincTestNNetwork");
//nn.read("networks/MNIST/MNIST_RBMAE-nh500-250-30-250-500_network.txt");
		nn.getweights(weights);
		
		//gsl_matrix_float_view dataview = gsl_matrix_float_view_array(wtd.inputs, wtd.ndata, wtd.nin);
		//gsl_matrix_float *data = &dataview.matrix;
		//gsl_matrix_float *wprob = gsl_matrix_float_alloc(wtd.ndata, wtd.nin+1);

		float *data = (float *) malloc(wtd.ndata * wtd.nin * sizeof(float));
		float *wprob = (float *) malloc(wtd.ndata * (wtd.nin+1) * sizeof(float));
		for (size_t i=0; i<wtd.ndata; i++)
			for (size_t j=0; j<wtd.nin; j++)
				data[i * wtd.nin + j] = wtd.inputs[i * wtd.nin + j];
			
		//  column vector with all 1s
		//gsl_vector_float *ones = gsl_vector_float_alloc(wtd.ndata);
		//gsl_vector_float_set_all(ones, 1E0);

		float *ones = (float *) malloc(wtd.ndata * sizeof(float));
		for (size_t i=0; i<wtd.ndata; i++)
			ones[i] = 1.0;
		
		for( size_t n = 1; n < NNnlayers; n++ )
		{
			size_t nlayers = 2;
			size_t nnodes[nlayers];
			nnodes[0] = NNnnodes[n-1];			// nodes (including biases) in the input layer
			nnodes[1] = n == NNnlayers-1 ? NNnnodes[n] : NNnnodes[n]-1;
			size_t nin = nnodes[0]-1, nhid = nnodes[1];
std::cout << "layer " << n+1 << " of " << NNnlayers << ": " << nin << "-" << nhid << ".\n";
			
			RBM rbm(nnodes, linear[n], 1);
			rbm.setweights(&weights[ncumweights[n-1]]);
			
			if( n == 1 )
			{
				//gsl_vector_float *colvec = gsl_vector_float_alloc(wtd.ndata);
				float *colvec = (float *) malloc(wtd.ndata * sizeof(float));
				for( size_t i = 0; i < wtd.nin; i++ )
				{
					//gsl_matrix_float_get_col(colvec, data, i);
					//gsl_matrix_float_set_col(wprob, i, colvec);
					for (size_t j=0; j<wtd.ndata; j++)
						colvec[j] = data[j * wtd.nin + i];
					for (size_t j=0; j<wtd.ndata; j++)
						wprob[j * wtd.nin + i] = colvec[j];
				}
			
				// append a column of 1s to wprob matrix
				//gsl_matrix_float_set_col(wprob, wtd.nin, ones);
				for (size_t i=0; i<wtd.ndata; i++)
					wprob[i * wtd.nin + wtd.nin] = ones[i];

				//gsl_vector_float_free(colvec);
				free(colvec);
			}
			
			
			// setup the weights matrix
			
			float **vishid = new float* [nin]; for( size_t i = 0; i < nin; i++ ) vishid[i] = new float [nhid];
			float *hidbias = new float [nhid];
				
			rbm.weightstoarrays(vishid, hidbias);

//for( size_t i = 0; i < nin; i++ )
//	for( size_t j = 0; j < nhid; j++ )
//		wfile >> vishid[i][j];
//for( size_t j = 0; j < nhid; j++ ) bfile >> hidbias[j];
//rbm.arraystoweights(vishid, hidbias);
//rbm.getweights(&weights[ncumweights[n-1]]);
			
			//gsl_matrix_float *w = gsl_matrix_float_alloc(nin+1, nhid);
			float *w = (float *) malloc((nin+1) * nhid *sizeof(float));

			for( size_t i = 0; i < nin; i++ )
				for( size_t j = 0; j < nhid; j++ )
					//gsl_matrix_float_set(w, i, j, vishid[i][j]);
					w[i * nhid + j] = vishid[i][j];
			//for( size_t j = 0; j < nhid; j++ ) gsl_matrix_float_set(w, nin, j, hidbias[j]);
			for (size_t i=0; i<nhid; i++) w[nin * nhid + i] = hidbias[j];
				
			for( size_t i = 0; i < nin; i++ ) delete [] vishid[i]; delete [] vishid;
			delete [] hidbias;
			
			
			// calculate wprob * w
			/*gsl_matrix_float *wprobnew = gsl_matrix_float_alloc(wtd.ndata, nhid);
			gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1E0, wprob, w, 0E0, wprobnew);
			gsl_matrix_float_free(w);
			gsl_matrix_float_free(wprob);*/

			float *wprobnew = (float *) malloc(wtd.ndata * nhid *sizeof(float));
			cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,wtd.ndata,nhid,wtd.nin+1,1.0,wprob,wtd.nin+1,w,nhid,0.0,wprobnew,nhid);
			free(w);
			free(wprob);
			
			/*wprob = gsl_matrix_float_alloc(wtd.ndata, nhid+1);
			for( size_t i = 0; i < wtd.ndata; i++ )
				for( size_t j = 0; j < nhid; j++ )
					gsl_matrix_float_set(wprob, i, j, linear[n] ? gsl_matrix_float_get(wprobnew, i, j) : rbm.sigmoid(gsl_matrix_float_get(wprobnew, i, j), 1));*/
			wprob = (float *) malloc(wtd.ndata * (nhid+1) * sizeof(float));
			for (size_t i=o; i<wtd.ndata; i++)
				for (size_t j=0; j<nhid; j++)
					wprob[i * (nhid+1) + j] = linear[n] ? wprobnew[i * (nhid+1) + j] : rbm.sigmoid(wprobnew[i * (nhid+1) + j],1);

			// append a column of 1s to wprob matrix
			//gsl_matrix_float_set_col(wprob, nhid, ones);
			for (size_t i=0; i<wtd.ndata; i++)
				wprob[i * (nhid+1) + nhid] = ones[i];

			//gsl_matrix_float_free(wprobnew);
			free(wprobnew);
		}
//wfile.close();
//bfile.close();
//nn.setweights(weights);
//nn.write("MNISTRMBNetwork");
		
		float eSq  = 0E0;
		for( size_t i = 0; i < wtd.ndata; i++ )
			for( size_t j = 0; j < wtd.nout; j++ )
			{
				//eSq += pow(gsl_matrix_float_get(wprob, i, j)-gsl_matrix_float_get(data, i, j), 2.0);
				eSq += pow(wprob[i * (nhid+1) + j] - data[i * wtd.nin + j],2.0);
/*if(gsl_matrix_float_get(wprob, i, j)<0.5)
{
	std::cout << i << " " << j << " " << gsl_matrix_float_get(wprob, i, j) << "\n";
	getchar();
}*/
			}
		eSq /= float(wtd.ndata);
		
		std::cout << "squared error = " << eSq << "\n";
		
			
		//gsl_matrix_float_free(wprob);
		//gsl_vector_float_free(ones);
		free(wprob);
		free(ones);
		free(data);

		PredictedData pd(nn.totnnodes, nn.totrnodes, wtd.ndata, wtd.nin, wtd.nout, wtd.cuminsets[wtd.ndata], wtd.cumoutsets[wtd.ndata]);
		float logL;
		eSq = 0E0;
		nn.logLike(wtd, pd, logL);
		nn.ErrorSquared(wtd, pd);
		for( size_t i = 0; i < wtd.nout; i++ ) eSq += pd.errsqr[i];
		std::cout << "logL = " << logL << ", squared error = " << eSq << "\n";
	}
	else
	{
		float *inputs;
		
		for( size_t n = 0; n < depth; n++ )
		{
			// input data
			if( n == 0 )
			{
				inputs = new float [wtd.ndata*wtd.nin];
				for( size_t i = 0; i < wtd.ndata*wtd.nin; i++ ) inputs[i] = wtd.inputs[i];
			}
		
			// initialize the NN
		
			size_t nlayers = 2;
			size_t nnodes[nlayers];
			nnodes[0] = n == 0 ? wtd.nin+1 : nhid[n-1]+1;			// nodes (including biases) in the input layer
			nnodes[1] = nhid[n];
		
		
			RBM rbm(nnodes, linear[n+1], wtd.ndata);
		
			if( n == 0 ) inputs = wtd.inputs;
			
			float mean[] = {0E0, 0E0, 0E0};		// means of the normal distributions of initial weights for in-hid, hid-bias, in-bias respectively
			float sd[] = {stdev, 0E0, 0E0};		// standard deviations of the normal distributions of initial weights for in-hid, hid-bias, in-bias respectively
			rbm.setnormalweights(mean, sd, seed);	// set the intial weights
			
			rbm.pretrain(inputs, seed);
		
			// dump the network
			//rbm.write("SincTestRMBNetwork");
			
			delete [] inputs;
			if( n < depth-1 )
			{
				inputs = new float [wtd.ndata*nhid[n]];
				for( size_t i = 0; i < wtd.ndata*nhid[n]; i++ ) inputs[i] = rbm.batchposhidprob[i];
			}
			
			// copy the RBM weights to full NN weights
			rbm.getweights(&weights[ncumweights[n]]);
			
			if( autoencoder )
			{
				size_t nin = nnodes[0]-1, nout = nnodes[1];
				float **vishid = new float* [nin]; for( size_t i = 0; i < nin; i++ ) vishid[i] = new float [nout];
				float *hidbias = new float [nout];
				
				rbm.weightstoarrays(vishid, hidbias);
				arraystranstoweights(nout, nin, vishid, rbm.visbias, &weights[ncumweights[NNnlayers-2-n]]);
				
				for( size_t i = 0; i < nin; i++ ) delete [] vishid[i]; delete [] vishid;
				delete [] hidbias;
			}
		}
		
		// initialize the weights in the final layer for non-autoencoders
		if( !autoencoder )
		{
			size_t nout = NNnnodes[NNnlayers-2]-1, nin = NNnnodes[NNnlayers-1];
			float **vishid = new float* [nin]; for( size_t i = 0; i < nin; i++ ) vishid[i] = new float [nout];
			float *visbias = new float [nin];
			
			for( size_t i = 0; i < nin; i++ )
			{
				visbias[i] = 0E0;
				for( size_t j = 0; j < nout; j++ ) vishid[i][j] = gasdev(&seed)*stdev;
			}

			arraystranstoweights(nout, nin, vishid, visbias, &weights[ncumweights[NNnlayers-2]]);
				
			for( size_t i = 0; i < nin; i++ ) delete [] vishid[i]; delete [] vishid;
			delete [] visbias;
		}
		
		nn.setweights(weights);
		nn.write("SincTestNNetwork");
		
		PredictedData pd(nn.totnnodes, nn.totrnodes, wtd.ndata, wtd.nin, wtd.nout, wtd.cuminsets[wtd.ndata], wtd.cumoutsets[wtd.ndata]);
		float logL, eSq = 0E0;
		nn.logLike(wtd, pd, logL);
		nn.ErrorSquared(wtd, pd);
		for( size_t i = 0; i < wtd.nout; i++ ) eSq += pd.errsqr[i];
		std::cout << "logL = " << logL << ", squared error = " << eSq << "\n";
	}
	
	
#ifdef PARALLEL
	MPI_Finalize();
#endif
	
}
