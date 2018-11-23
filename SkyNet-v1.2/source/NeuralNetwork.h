#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__ 1

#ifdef PARALLEL
#include <mpi.h>
#endif
#include <cmath>
#include <cassert>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <iomanip>
#include <float.h>
#include <time.h>
#include <limits>

extern "C" {
	#include <cblas.h>
}

#include "whiten.h"
#include "TrainingData.h"
#include "PredictedData.h"
#include "myrand.h"

enum actfct {
	act_linear,
	act_sigmoid,
	act_tanh,
	act_relu,
	act_softsign,
	act_softplus
};

template<typename T>
	std::ostream &operator <<(std::ostream &os, const std::vector<T> &v)
{
	os << "(";
	for(size_t i = 0; i < v.size(); i++)
		os << v[i] << " ";
	os << ") ";
	return os;
} 

// debugging - representation of bit pattern of a particular data type
//template<typename T>
//	static inline int ff(const T &t) { return *((int *)&t); };

template<typename T>
	static inline T ff(const T &t) { return t; };



class NeuralNetwork
{
	protected:
		float			*weights;	// O the definition of the network (including biases)
		bool			*setzero;	// I which weights to set to 0
		float			***w;
	public:
		int			myid;		// MPI processor ID
		int			ncpus;		// MPI total no. of processors
		size_t			nlayers;	// O no. of layers, including input and output layer
		size_t			*nnodes;   	// I Number in non-recurrent nodes in each layers, including the bias units
		size_t			*rnodes;   	// I Number in recurrent nodes in each layers
		bool      		recurrent;    	// I RNN?
		size_t			loglikef;	// which loglike function to use? 1 = SSE, 2 = HHP
		size_t			nweights;	// O total number of weights
		size_t			*ncumweights;	// O no. of cumulative non-recurrent weights for each layer
		size_t			nnweights;	// O total number of non-recurrent weights
		size_t			nrweights;	// O total number of recurrent weights
		size_t 			totnnodes;	// O total no. of non-recurrent nodes
		size_t 			totrnodes;	// O total no. of recurrent nodes
		bool			norbias;	// blank initial state for RNN?
		//bool			*linear;	// whether a layers is linear or not? (by default only the output & input layers are linear, rest are sigmoid)
		int 			*linear;    // the choice of activation function
		bool			pflag;		// parallel flag

		size_t nparams() const { return nweights; };

		void dump(std::ostream &os)
		{
			os << "nlayers = " << nlayers << "\n";
			os << "nnodes = ";
			for(size_t i = 0; i < nlayers; i++) os << nnodes[i] << "\t";
			os << "recurrent = " << recurrent << "\n";
			os << "weights = ";
			for(size_t i = 0; i < nweights; i++) os << weights[i] << "\t";
		}

		NeuralNetwork()
		{
			nlayers = nweights = 0;
			int _nlayers = 0;
			size_t *_nnodes = NULL;
			pflag = true;
			//Init(_nlayers, _nnodes, false, false);
		}
		
		void freememory()
		{
			if( nweights > 0 )
			{
				delete [] weights;
				delete [] ncumweights;
				
				for( size_t i = 1; i < nlayers; i++ )
				{
					size_t jend = i == nlayers-1 ? nnodes[i] : nnodes[i]-1;
					for( size_t j = 0; j < jend; j++ ) delete [] w[i-1][j];;
					delete [] w[i-1];
				}
				delete [] w;
			}
			if( nlayers > 0 )
			{
				delete [] nnodes;
				delete [] linear;
				if( recurrent ) delete [] rnodes;
			}
		}

		~NeuralNetwork()
		{
			freememory();
		}

		NeuralNetwork(size_t _nlayers, size_t *_nnodes)
		{
			nlayers = nweights = 0;
			bool _recurrent = false, _norbias = false;
			Init(_nlayers, _nnodes, _recurrent, _norbias);
		}

		NeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent)
		{
			nlayers = nweights = 0;
			bool _norbias = false;
			Init(_nlayers, _nnodes, _recurrent, _norbias);
		}

		NeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, bool _norbias)
		{
			nlayers = nweights = 0;
			Init(_nlayers, _nnodes, _recurrent, _norbias);
		}

		NeuralNetwork(size_t _nlayers, size_t *_nnodes, int *_linear)
		{
			nlayers = nweights = 0;
			bool _recurrent = false, _norbias = false;
			Init(_nlayers, _nnodes, _recurrent, _norbias, _linear);
		}

		NeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, int *_linear)
		{
			nlayers = nweights = 0;
			bool _norbias = false;
			Init(_nlayers, _nnodes, _recurrent, _norbias, _linear);
		}

		NeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, bool _norbias, int *_linear)
		{
			nlayers = nweights = 0;
			Init(_nlayers, _nnodes, _recurrent, _norbias, _linear);
		}

		NeuralNetwork(const NeuralNetwork &n)
		{
			nlayers = nweights = 0;
			Clone(n);
		}

		NeuralNetwork &operator = (const NeuralNetwork &n)
		{
			Clone(n);
			return *this;
		}

		void Clone(const NeuralNetwork &n)
		{			
			Init(n.nlayers, n.nnodes, n.recurrent, n.norbias, n.linear);
			for(size_t i = 0; i < n.nweights; i++)
			{
				weights[i] = n.weights[i];
				setzero[i] = n.setzero[i];
			}
			arrangeweights();
		}

		bool operator != (const NeuralNetwork &s) const
		{
			return !((*this) == s);
		}

		bool operator == (const NeuralNetwork &s) const
		{
			if (nlayers != s.nlayers) return false;
			for(size_t i = 0; i < nlayers; i++)
			{
				if (nnodes[i] != s.nnodes[i]) return false;
				if (linear[i] != s.linear[i]) return false;
			}
			if (recurrent != s.recurrent) return false;
			if (norbias != s.norbias) return false;

			if (nweights != s.nweights) return false;
			const float eps = 1e-6f;
			for(size_t i = 0; i < nweights; i++)
			{
				if (fabs(weights[i] - s.weights[i]) > eps) 
				{
					std::cerr <<" Weight fail "<< weights[i] << " " << s.weights[i] << " " << fabs(weights[i] - s.weights[i]) << "\n";
					return false;
				}
				if (setzero[i] != s.setzero[i]) return false;
			}
			return true;
		}

		// mean and RMS of the weights and biases
		void weightInfo(float &av, float &sig)
		{
			av = sig = 0E0;
			float nav = 0E0;
			for(size_t i = 0; i < nweights; i++)
			{
				av += weights[i];
				sig += weights[i] * weights[i];
				nav++;
			}
			av /= nav;
			sig = sqrt(fmax(sig/nav - av*av, 0E0));
		}

		// Sum_i (w_i)^2
		void getEW(float *EW)
		{
			EW[0] = 0E0;
			for( size_t i = 0; i < nweights; i++) EW[0] += weights[i] * weights[i] / 2.0;
		}
		
		void Init(size_t _nlayers, size_t *_nnodes, bool _recurrent, bool _norbias)
		{
			int *_linear = NULL;
			if( _nlayers > 0 )
			{
				// set input & output layers as linear & rest as non-linear (sigmoid)
				//_linear = new bool [_nlayers];
				//_linear[0] = _linear[_nlayers-1] = true;
				//for( size_t i = 1; i < _nlayers-1; i++ ) _linear[i] = false;
				_linear = new int [_nlayers];
				_linear[0] = _linear[_nlayers-1] = 0;
				for( size_t i = 1; i < _nlayers-1; i++ ) _linear[i] = 1;
			}
			
			Init(_nlayers, _nnodes, _recurrent, _norbias, _linear);
			if( _nlayers > 0 ) delete [] _linear;
		}
		
		void Init(size_t _nlayers, size_t *_nnodes, bool _recurrent, bool _norbias, int *_linear)
		{
			myid = 0;
			ncpus = 1;
#ifdef PARALLEL
			MPI_Comm_rank(MPI_COMM_WORLD,&myid);
			MPI_Comm_size(MPI_COMM_WORLD, &ncpus);
#endif
			freememory();
			
			pflag = true;
			recurrent = _recurrent;
			norbias = _norbias;
			if( !recurrent ) norbias = false;
			totnnodes = totrnodes = 0;
			nlayers = _nlayers;
			nnodes = new size_t [nlayers];
			ncumweights = new size_t [nlayers];
			linear = new int [nlayers];
			if( recurrent ) rnodes = new size_t [nlayers];

			for(size_t i = 0; i < nlayers; i++)
			{
				linear[i] = _linear[i];
				nnodes[i] = _nnodes[i];
				totnnodes += _nnodes[i];
				
				if( recurrent )
				{
					if( !norbias && i > 0 && i < nlayers-1 )
						rnodes[i] = nnodes[i]-1;
					else
						rnodes[i] = 0;
					
					totrnodes += rnodes[i];
				}
			}
			
			if( nlayers >= 2 )
			{				
				nweights = nnweights = nrweights = 0;
				ncumweights[0] = 0;
				for( int i = 1; i < nlayers; i++ )
				{
					if( i == nlayers - 1 )
					{
						nnweights += nnodes[i]*nnodes[i-1];
					}
					else
					{
						nnweights += (nnodes[i]-1)*nnodes[i-1];
						if( recurrent ) nrweights += pow(float(nnodes[i]-1), 2) + rnodes[i];
					}
					ncumweights[i] = nnweights;
				}
				nweights = nnweights + nrweights;
					
				weights = new float [nweights];
				setzero = new bool [nweights];
				for( size_t i = 0; i < nweights; i++ )
				{
					weights[i] = 0E0;
					setzero[i] = false;
				}
			
				w = new float **[nlayers-1];
				for( size_t i = 1; i < nlayers; i++ )
				{
					size_t jend = i == nlayers-1 ? nnodes[i] : nnodes[i]-1;
					w[i-1] = new float* [jend];
					for( size_t j = 0; j < jend; j++ ) w[i-1][j] = new float [nnodes[i-1]];
				}
				arrangeweights();
			}		
		}
		
		void SetZero(size_t nincat, size_t *ninclasses)
		{
			if( nincat == 1 ) return;
			
			// sanity checks
			if( nlayers <= 2 ) throw std::runtime_error("With input groupings at least 3 layers are needed in network architecture.");
			if( nincat != nnodes[1]-1 ) throw std::runtime_error("With input groupings, nodes in the second layer (excluding bias node) should be equal to input groups.");
			size_t k = 0;
			for( size_t i = 0; i < nincat; i++ ) k += ninclasses[i];
			if( k != nnodes[0]-1 ) throw std::runtime_error("With input groupings, nodes in the first layer (excluding bias node) should be equal to all nodes in input groups.");

			
			for( size_t i = 0; i < nincat*nnodes[0]; i++ ) setzero[i] = true;
			
			size_t p = 0;
			for( size_t i = 0; i < nincat; i++ )
			{
				for( size_t j = 0; j < i; j++ ) p += ninclasses[j];
				for( size_t j = 0; j < ninclasses[i]; j++ )
				{
					setzero[p] = false;
					p++;
				}
				for( size_t j = i+1; j < nincat; j++ ) p += ninclasses[j];
				setzero[p] = false;
				p++;
			}
		}

		void read(const std::string &fn)
		{
			float temp;
			std::ifstream fin(fn.c_str());
			read(fin, &temp, &temp, &temp, false);
			fin.close();
		}

		void read(const std::string &fn, float *rate, float *alphas, float *scale)
		{
			std::ifstream fin(fn.c_str());
			read(fin, rate, alphas, scale, true);
			fin.close();
		}

		void read(std::istream &is, float *rate, float *alpha, float *scale, bool readextra)
		{
			size_t _nin;
			std::vector<size_t> _nhid;
			size_t *_nnodes = NULL;
			size_t _nout, _nlayers;
			bool _recurrent = recurrent;
			bool _norbias = norbias;
			std::vector<float> _weights;
			std::vector<int> _linear;
			float _alpha, _rate=0.1;

			while(is)
			{
				std::string line;
				std::getline(is, line);
				std::stringstream ss(line);
				std::string tag;
				ss >> tag;
				if (tag == std::string("nin"))
				{
					ss >> _nin;
					_nin++;
					
					// input layer is always linear
					_linear.resize(std::max(int(_linear.size()), 1));
					//_linear[0] = true;
					_linear[0] = 0;
				}
				else if (tag == std::string("nh"))
				{
					int p = 0;
					ss >> p;
					_nhid.resize(std::max(int(_nhid.size()), p+1));
					ss >> _nhid[p];
					_nhid[p]++;
					
					// read whether this hidden layers is linear or not
					_linear.resize(std::max(int(_linear.size()), p+2));
					//bool lin;
					int lin;
					ss >> lin;
					_linear[p+1] = lin;
				}
				else if (tag == std::string("nout"))
				{
					ss >> _nout;
					
					// read whether this output layers is linear or not
					//bool lin;
					int lin;
					ss >> lin;
					_linear.push_back(lin);
				}
				else if (tag == std::string("recurrent"))
					ss >> _recurrent;
				else if (tag == std::string("weights"))
				{
					while(ss)
					{
						float sw = log(0E0);
						ss >> sw;
						//if (!finite(sw)) break;
						if (fabs(sw) > std::numeric_limits<float>::max()) break;
						if (!ss) break;
						_weights.push_back(sw);
					}
				}
				else if (tag == std::string("alphas"))
				{
					float sw = log(0E0);
					ss >> sw;
					if( readextra ) _alpha = sw;
				}
				else if (tag == std::string("rate"))
				{
					float sw = log(0E0);
					ss >> sw;
					if( readextra ) _rate = sw;
				}
				else if (tag == std::string("scale"))
				{
					int i = 0;
					while(ss)
					{
						float sw = log(0E0);
						ss >> sw;
						//if (!finite(sw)) break;
						if (fabs(sw) > std::numeric_limits<float>::max()) break;
						if (!ss) break;
						if( readextra ) scale[i] = sw;
						i++;
					}
				}
			}
			
			if( readextra ) *alpha = _alpha;
			if( readextra ) *rate = _rate;
			_nlayers = _nhid.size()+2;
			_nnodes = new size_t [_nlayers];
			//bool *_linear1 = new bool [_nlayers];
			int *_linear1 = new int [_nlayers];
			for( size_t i = 0; i < _nlayers; i++ ) _linear1[i] = _linear[i];
			_nnodes[0] = _nin;
			for( size_t i = 0; i < _nhid.size(); i++ ) _nnodes[i+1] = _nhid[i];
			_nnodes[_nlayers-1] = _nout;
			Init(_nlayers, _nnodes, _recurrent, _norbias, _linear1);
			size_t nc = std::min(nweights, _weights.size());
			//std::cerr << "Loaded " << _weights.size() << " weights\n";
			//std::cerr << "Using " << nc << " weights\n";
			std::copy(&_weights[0], &_weights[0] + nc, &weights[0]);
			arrangeweights();
		}

		void write(const std::string &fn)
		{
			if( myid == 0 )
			{
				std::ofstream fout(fn.c_str());
				write(fout, false, 0);
				fout.close();
			}
		}

		void write(const std::string &fn, bool autoencoder, size_t net)
		{
			if( myid == 0 )
			{
				std::ofstream fout(fn.c_str());
				write(fout, autoencoder, net);
				fout.close();
			}
		}

		void write(const std::string &fn, float *alpha, float *scale)
		{
			if( myid == 0 )
			{
				std::ofstream fout(fn.c_str());
				write(fout, false, 0);
				
				int done = 0;
				fout << "#prior alphas\n";
				fout << "alphas ";
				fout << std::scientific << std::setprecision(10) << *alpha << " ";
				fout << "\n";
				
				done = 0;
				fout << "#noise scale\n";
				fout << "scale ";
				for( int i = 0; i < nnodes[nlayers-1]; i++ ) fout << std::scientific << std::setprecision(10) << scale[i] << " ";
				fout << "\n";
				
				fout.close();
			}
		}

		void write(const std::string &fn, float rate, float *alpha, float *scale)
		{
			if( myid == 0 )
			{
				std::ofstream fout(fn.c_str());
				write(fout, false, 0);
				
				int done = 0;
				fout << "#prior alphas\n";
				fout << "alphas ";
				fout << std::scientific << std::setprecision(10) << *alpha << " ";
				fout << "\n";
				
				fout << "#structural damping coefficient\n";
				fout << "rate ";
				fout << std::scientific << std::setprecision(10) << rate << " ";
				fout << "\n";
				
				done = 0;
				fout << "#noise scale\n";
				fout << "scale ";
				for( int i = 0; i < nnodes[nlayers-1]; i++ ) fout << std::scientific << std::setprecision(10) << scale[i] << " ";
				fout << "\n";
				
				fout.close();
			}
		}

		void write(const std::string &fn, float *alpha)
		{
			if( myid == 0 )
			{
				std::ofstream fout(fn.c_str());
				write(fout, false, 0);
				
				int done = 0;
				fout << "#prior alphas\n";
				fout << "alphas ";
				fout << std::scientific << std::setprecision(10) << *alpha << " ";
				fout << "\n";
				
				fout.close();
			}
		}

		void write(const std::string &fn, float rate, float *alpha)
		{
			if( myid == 0 )
			{
				std::ofstream fout(fn.c_str());
				write(fout, false, 0);
				
				int done = 0;
				fout << "#prior alphas\n";
				fout << "alphas ";
				fout << std::scientific << std::setprecision(10) << *alpha << " ";
				fout << "\n";
				
				fout << "#structural damping coefficient\n";
				fout << "rate ";
				fout << std::scientific << std::setprecision(10) << rate << " ";
				fout << "\n";
				
				fout.close();
			}
		}

		void write(std::ostream &os, bool autoencoder, size_t net) const
		{
			// find the output layer for autoencoder
			int outlayer = nlayers-1;
			int inlayer = 0;
			int nw = nweights;
			if( autoencoder )
			{
				int k = 1;
				for( size_t i = 2; i < nlayers-1; i++ )
					if( nnodes[i] < nnodes[k] )
						k = i;
						
				if( net == 1 )		// input-to-autencoder
					outlayer = k;
				else if( net == 2 )	// autencoder-to-output
					inlayer = k;
					
				nw = ncumweights[outlayer]-ncumweights[inlayer];
			}
			int outnodes = nnodes[outlayer];
			if( outlayer != nlayers-1 ) outnodes--;
		
			os << "#number of inputs\n";
			os << "nin " << nnodes[inlayer]-1 << " " << linear[inlayer] << "\n";
			for(size_t i = inlayer; i < outlayer-1; i++)
			{
				os << "#number of hidden in " << i-inlayer << "'th layer\n";
				os << "nh " << i-inlayer << " " << nnodes[i+1]-1 << " " << linear[i+1] << "\n";
			}
			os << "#number of outputs\n";
			os << "nout " << outnodes << " " << linear[outlayer] << "\n";
			os << "#recurrent boolean flag\n";
			os << "recurrent " << recurrent << "\n";
			os << "#parameters of the network - weights and biases together " << nw << " in total\n";
			int start = ncumweights[inlayer];
			int done = 0;
			std::ios_base::fmtflags originalFormat = os.flags();
			while(done < int(nw))
			{
				os << "weights ";
				int quantum = std::min(100, int(nw) - done);
				for(int i = done; i < done + quantum; i++) 
					os << std::scientific << std::setprecision(10) << weights[start+i] << " ";
				done += quantum;
				os << "\n";
			}
			os.flags(originalFormat);
			formatWeights(os, inlayer, outlayer);
		}

		void writeSimple(const std::string &fn, bool classnet)
		{
			if( myid == 0 )
			{
				std::ofstream fout(fn.c_str());
				
				int outlayer = nlayers-1;
				int inlayer = 0;
				int nw = nweights;
				
				fout << "#nlayers\n" << nlayers << "\n";
				fout << "#nin\n" << nnodes[inlayer]-1 << "\t" << linear[inlayer] << "\n";
				fout << "#nhidden\n";
				for (int i=inlayer; i<outlayer-1; i++)
					fout << nnodes[i+1]-1 << "\t" << linear[i+1] << "\n";
				fout << "#nout\n" << nnodes[outlayer] << "\t" << linear[outlayer] << "\n";
				fout << "#recurrent\n" << recurrent << "\n";
				fout << "#norbias\n" << norbias << "\n";
				fout << "#classnet\n" << classnet << "\n";
				fout << "#weights\n";
				for (int i=0; i<nw; i++)
					fout << std::scientific << std::setprecision(10) << weights[i] << "\n";
				
				fout.close();
			}
		}

		void formatWeights(std::ostream &os, int inlayer, int outlayer) const
		{
			size_t k = 0;
			size_t start = ncumweights[inlayer];
			
			// write the normal weights
			for(size_t layer = inlayer+1; layer <= outlayer; layer++)
			{
				os << "#layer " << layer-1-inlayer << " weights\n";
				os << "# ";
				size_t iend = layer == nlayers-1 ? nnodes[layer] : nnodes[layer]-1;
				for( size_t i = 0; i < iend; i++ )
				{
					for( size_t j = 0; j < nnodes[layer-1]; j++ )
					{
						os << weights[start+k] << " ";
						k++;
					}
				}
				os << "\n";
			}
			k = ncumweights[nlayers-1];
			
			// write the recurrent weights
			if( recurrent )
			{
				for(size_t layer = 1; layer < outlayer; layer++)
				{
					os << "#layer " << layer-1 << " recurrent weights\n";
					os << "# ";
					for( size_t i = 0; i < pow(float(nnodes[layer]-1), 2) + rnodes[layer]; i++ )
					{
						os << weights[k] << " ";
						k++;
					}
					os << "\n";
				}
			}
		}

		// this needs more thought: this is obviously the wrong pattern
		
		void whitenWeights(const whiteTransform *itrans, const whiteTransform *otrans)
		{
			const axisAlignedTransform *ax = dynamic_cast<const axisAlignedTransform *>(itrans);
			if (ax)
			{
				whitenWeights(*ax, *dynamic_cast<const axisAlignedTransform *>(otrans));			
				return;
			}
			
			throw std::runtime_error("Cannot whiten with this transform");
		}

		// this needs more thought: this is obviously the wrong pattern
		
		void unwhitenWeights(const whiteTransform *itrans, const whiteTransform *otrans)
		{
			const axisAlignedTransform *ax = dynamic_cast<const axisAlignedTransform *>(itrans);
			if (ax)
			{
				unwhitenWeights(*ax, *dynamic_cast<const axisAlignedTransform *>(otrans));			
				return;
			}
			
			throw std::runtime_error("Cannot unwhiten with this transform");
		}

		// this needs more thought: this is obviously the wrong pattern
		
		void whitenWeights(const whiteTransform *itrans, const int flag)
		{
			const axisAlignedTransform *ax = dynamic_cast<const axisAlignedTransform *>(itrans);
			if (ax)
			{
				whitenWeights(*ax, flag);			
				return;
			}
			
			throw std::runtime_error("Cannot whiten with this transform");
		}

		// this needs more thought: this is obviously the wrong pattern
		
		void unwhitenWeights(const whiteTransform *itrans, const int flag)
		{
			const axisAlignedTransform *ax = dynamic_cast<const axisAlignedTransform *>(itrans);
			if (ax)
			{
				unwhitenWeights(*ax, flag);			
				return;
			}
			
			throw std::runtime_error("Cannot unwhiten with this transform");
		}
		
		// given that this network has been trained on original data set and result set,
		// adjust the weights so that it can be applied to data and results related to the 
		// original ones by an affine transform

		// hence (after this call) the network will be as if it has been trained on input/output that has been transformed:
		// input[i] = (input'[i] - itrans.off[i]) / itrans.scale[i]
		// output[i] = (output'[i] - otrans.off[i]) / otrans.scale[i]
		
		void whitenWeights(const axisAlignedTransform &itrans, const axisAlignedTransform &otrans)
		{
			/*assert(itrans.scale.size() == nnodes[0]-1);
			assert(itrans.off.size() == nnodes[0]-1);
			assert(otrans.scale.size() == nnodes[nlayers-1]);
			assert(otrans.off.size() == nnodes[nlayers-1]);*/
			
			//incorporate offsets and scalings into input layer weights and biases
			int iend = nlayers == 2 ? nnodes[1] : nnodes[1]-1;
			for(size_t i=0; i<iend; i++)
			{
				for(size_t j=0; j<nnodes[0]-1; j++)
				{
					w[0][i][nnodes[0]-1] += w[0][i][j] * itrans.off[j];
					w[0][i][j] *= itrans.scale[j];
				}
			}
			
			
			//incorporate offsets and scalings into output layer weights and biases
			for(size_t i=0; i<nnodes[nlayers-1]; i++)
			{
				for(size_t j=0; j<nnodes[nlayers-2]; j++)
				{
					w[nlayers-2][i][j] /= otrans.scale[i];
				}
				w[nlayers-2][i][nnodes[nlayers-2]-1] -= otrans.off[i] / otrans.scale[i];
			}
			
			inversearrangeweights();
		}
		
		// given that this network has been trained on a data set and result set,
		// adjust the weights so that it can be applied to data and results related to the 
		// original ones by an (inverse) affine transform

		// hence if (before this call) the network has been trained on input/output that has been transformed:
		// input[i] = (input'[i] - itrans.off[i]) / itrans.scale[i]
		// output[i] = (output'[i] - otrans.off[i]) / otrans.scale[i]
		
		// then afterwards the weights will be as if it had been trained on inputs and outputs
		
		void unwhitenWeights(const axisAlignedTransform &itrans, const axisAlignedTransform &otrans)
		{
			/*assert(itrans.scale.size() == nnodes[0]-1);
			assert(itrans.off.size() == nnodes[0]-1);
			assert(otrans.scale.size() == nnodes[nlayers-1]);
			assert(otrans.off.size() == nnodes[nlayers-1]);*/
			
			//incorporate offsets and scalings into input layer weights and biases
			int iend = nlayers == 2 ? nnodes[1] : nnodes[1]-1;
			for(size_t i=0; i<iend; i++)
			{
				for(size_t j=0; j<nnodes[0]-1; j++)
				{
					w[0][i][j] /= itrans.scale[j];
					w[0][i][nnodes[0]-1] -= w[0][i][j] * itrans.off[j];
				}
			}
			
			
			//incorporate offsets and scalings into output layer weights and biases
			for(size_t i=0; i<nnodes[nlayers-1]; i++)
			{
				for(size_t j=0; j<nnodes[nlayers-2]; j++)
				{
					w[nlayers-2][i][j] *= otrans.scale[i];
				}
				w[nlayers-2][i][nnodes[nlayers-2]-1] += otrans.off[i];
			}
			
			inversearrangeweights();
		}
		
		void whitenWeights(const axisAlignedTransform &itrans, const int flag)
		{
			/*assert(otrans.scale.size() == nnodes[nlayers-1]);
			assert(otrans.off.size() == nnodes[nlayers-1]);*/
			
			if( flag == 1 )
			{
				//incorporate offsets and scalings into input layer weights and biases
				int iend = nlayers == 2 ? nnodes[1] : nnodes[1]-1;
				for(size_t i=0; i<iend; i++)
				{
					for(size_t j=0; j<nnodes[0]-1; j++)
					{
						w[0][i][nnodes[0]-1] += w[0][i][j] * itrans.off[j];
						w[0][i][j] *= itrans.scale[j];
					}
				}
			}
			else if( flag == 2 )
			{
				//incorporate offsets and scalings into output layer weights and biases
				for(size_t i=0; i<nnodes[nlayers-1]; i++)
				{
					for(size_t j=0; j<nnodes[nlayers-2]; j++)
					{
						w[nlayers-2][i][j] /= itrans.scale[i];
					}
					w[nlayers-2][i][nnodes[nlayers-2]-1] -= itrans.off[i] / itrans.scale[i];
				}
			}
			
			inversearrangeweights();
		}
		
		void unwhitenWeights(const axisAlignedTransform &itrans, const int flag)
		{
			/*assert(itrans.scale.size() == nnodes[0]-1);
			assert(itrans.off.size() == nnodes[0]-1);*/
			
			if( flag == 1 )
			{
				//incorporate offsets and scalings into input layer weights and biases
				int iend = nlayers == 2 ? nnodes[1] : nnodes[1]-1;
				for(size_t i=0; i<iend; i++)
				{
					for(size_t j=0; j<nnodes[0]-1; j++)
					{
						w[0][i][j] /= itrans.scale[j];
						w[0][i][nnodes[0]-1] -= w[0][i][j] * itrans.off[j];
					}
				}
			}
			else if( flag == 2 )
			{
				//incorporate offsets and scalings into output layer weights and biases
				for(size_t i=0; i<nnodes[nlayers-1]; i++)
				{
					for(size_t j=0; j<nnodes[nlayers-2]; j++)
					{
						w[nlayers-2][i][j] *= itrans.scale[i];
					}
					w[nlayers-2][i][nnodes[nlayers-2]-1] += itrans.off[i];
				}
			}
			
			inversearrangeweights();
		}
		
		// utility function
		inline void getnstartend(int ndata, int &nstart, int &nend) const
		{
			if( pflag )
			{
				int ndpercpu = (int) ceil((float) ndata/ (float) ncpus);
				nstart = myid * ndpercpu;
				nend = (int) fmin(ndata, (myid+1)*ndpercpu);
			}
			else
			{
				nstart = 0;
				nend = ndata;
			}
		}

		// choose the activation function
		inline float activation(int choice, float x, int flag) const
		{
			float ret;

			switch (choice)
			{
				case act_linear:
					ret = linearact(x,flag);
					break;
				case act_sigmoid:
					ret = sigmoid(x,flag);
					break;
				case act_tanh:
					ret = tanh1(x,flag);
					break;
				case act_relu:
					ret = rectify(x,flag);
					break;
				case act_softsign:
					ret = softsign(x,flag);
					break;
				case act_softplus:
					ret = softplus(x,flag);
					break;
				default:
					throw std::runtime_error("invalid selection of activation function.");
			}

			return ret;
		}

		// utility function
		inline float linearact(float x, int flag) const
		{
			if( flag == 1 || flag == 3 )
				return x;
			else if( flag == 2)
				return 1;
			else
				throw std::runtime_error("wrong flag passed to linearact function.");
		}
		
		// utility function
		inline float tanh1(float x, int flag) const
		{
			float ret;
			
			if( flag == 1 )
			{
				ret = tanh(x);
				//if(!finite(ret)) ret = x < 0E0 ? -1E0 : 1E0;
				if (fabs(ret) > std::numeric_limits<float>::max()) ret = x < 0E0 ? -1E0 : 1E0;
			}
			else if( flag == 2 )
			{
				ret = 1E0 - tanh(x)*tanh(x);
			}
			else if( flag == 3 )
			{
				ret = tanh(x);
				//if(!finite(ret)) ret = x < 0E0 ? -1E0 : 1E0;
				if (fabs(ret) > std::numeric_limits<float>::max()) ret = x < 0E0 ? -1E0 : 1E0;
				ret = 0.5 * ( ret + 1.0 );
			}
			else
			{
				throw std::runtime_error("wrong flag passed to sigmoid function.");
			}
			
			return ret;
		}
		
		// utility function
		inline float sigmoid(float x, int flag) const
		{
			if( flag == 1 || flag == 3 )
			{
				if( x >= 0E0 )
					return 1E0 / (1E0 + exp(-x));
				else
					return exp(x) / (1E0 + exp(x));
			}
			else if( flag == 2 )
			{
				if( x >= 0E0 )
					return exp(-x) / pow((1E0 + exp(-x)), 2.0);
				else
					return exp(x) / pow((1E0 + exp(x)), 2.0);
			}
			else
			{
				throw std::runtime_error("wrong flag passed to sigmoid function.");
			}
		}

		// utility function
		inline float rectify(float x, int flag) const
		{
			if( flag == 1 )
			{
				if( x >= 0E0 )
					return x;
				else
					return 0E0;
			}
			else if( flag == 2 )
			{
				if( x >= 0E0 )
					return 1E0;
				else
					return 0E0;
			}
			else if( flag == 3 )
			{
				return x;
			}
			else
			{
				throw std::runtime_error("wrong flag passed to rectify function.");
			}
		}

		// utility function
		inline float softsign(float x, int flag) const
		{
			if( flag == 1 )
			{
				return x / (1E0 + fabs(x));
			}
			else if( flag == 2 )
			{
				return 1E0 / pow(1E0 + fabs(x),2E0);
			}
			else if( flag == 3 )
			{
				float ret = x / (1E0 + fabs(x));
				ret = 0.5 * ( ret + 1.0 );
				return ret;
			}
			else
			{
				throw std::runtime_error("wrong flag passed to softsign function.");
			}
		}
		
		// utility function
		inline float softplus(float x, int flag) const
		{
			if( flag == 1 || flag == 3)
			{
				return log(1E0 + exp(x));
			}
			else if( flag == 2 )
			{
				return sigmoid(x,1);
			}
			else
			{
				throw std::runtime_error("wrong flag passed to softsign function.");
			}
		}
		
		// utility function
		inline float lognfact(int n) const
		{
			float ret = 0E0;
			for( int i = 2; i < n; i++ )
				ret += log(n);
			
			return ret;
		}
		
		// utility function
		// LogSumExp(x,y)=log(exp(x)+exp(y))
		inline float LogSumExp(float x, float y) const
		{
			if( x >= y )
				return ( x + log( 1E0 + exp( y - x ) ) );
			else
				return ( y + log( 1E0 + exp( x - y ) ) );
		}
		
		int allocateweights(float ***wt)
		{
			wt = new float **[nlayers-1];
			for( size_t i = 1; i < nlayers; i++ )
			{
				size_t jend = i == nlayers-1 ? nnodes[i] : nnodes[i]-1;
				wt[i-1] = new float* [jend];
				for( size_t j = 0; j < jend; j++ )
				{
					wt[i-1][j] = new float [nnodes[i-1]];
					
					for( size_t k = 0; k < nnodes[i-1]; k++ ) wt[i-1][j][k] = 0E0;
				}
			}
			
			return 0;
		}
		
		int setweights(float *Cube)
		{
			for( int i = 0; i < nweights; i++ ) weights[i] = setzero[i] ? 0E0 : Cube[i];
			arrangeweights();
			
			return 0;
		}
		
		int setweights(int startidx, int n, float *Cube)
		{
			if( startidx >= nweights || startidx+n > nweights ) throw std::runtime_error("wrong startidx or n passed to setweights.\n");
			
			for( int i = startidx; i < startidx+n; i++ ) weights[i] = setzero[i] ? 0E0 : Cube[i-startidx];
			
			return 0;
		}
		
		int getweights(float *Cube)
		{
			for( int i = 0; i < nweights; i++ ) Cube[i] = weights[i];
			return 0;
		}
		
		int arrangeweights()
		{
			int p = -1;
			for( int i = 1; i < nlayers; i++ )
			{
				int jend = i == nlayers-1 ? nnodes[i] : nnodes[i]-1;
				for( int j = 0; j < jend; j++ )
				{
					for( int k = 0; k < nnodes[i-1]; k++ )
					{
						p++;
						w[i-1][j][k] = weights[p];
					}
				}
			}
			
			return 0;
		}
		
		int inversearrangeweights()
		{
			int p = -1;
			for( int i = 1; i < nlayers; i++ )
			{
				int jend = i == nlayers-1 ? nnodes[i] : nnodes[i]-1;
				for( int j = 0; j < jend; j++ )
				{
					for( int k = 0; k < nnodes[i-1]; k++ )
					{
						p++;
						weights[p] = w[i-1][j][k];
					}
				}
			}
			
			return 0;
		}

		int weightsAdjustActivation(bool autoencoder)
		{
			int i,j,jend,k;

			for( i=1; i<nlayers; i++ )
			{
				if( (i<nlayers-1 || autoencoder) && (linear[i-1]==2 || linear[i-1]==4) )
				{
					jend = i==nlayers-1 ? nnodes[i] : nnodes[i]-1;
					for( j=0; j<jend; j++ )
					{
						for( k=0; k<nnodes[i-1]; k++ )
						{
							w[i-1][j][k] = 2.0 * w[i-1][j][k] - 1.0;
						}
					}
				}
			}

			return 0;
		}
		
		int logPrior(float *alpha, float &logPrior)
		{
			// get E_{w_i} where i indicates the weight class
			float EW[1];
			getEW(EW);
			
			logPrior = -EW[0] * alpha[0];
		
			return 0;
		}
		
		int grad_prior(float *alpha, float *grad)
		{
			for( size_t i = 0; i < nweights; i++ ) grad[i] = -alpha[0] * weights[i];
		
			return 0;
		}
		
		int alphaIv(float *v, float *alpha, float *alphaIv)
		{
			for( size_t i = 0; i < nweights; i++) alphaIv[i] = alpha[0] * v[i];
		
			return 0;
		}
		
		int deriv(TrainingData &td, PredictedData &pd, int m, int n, float *deriv)
		{
			float g[nweights];
			for( int i = 0; i < nweights; i++ ) deriv[i] = 0E0;
			
			float e[td.nout]; // vector of ones
			for( int i = 0; i < td.nout; i++ ) e[i] = 1E0;
			backward(e, td, pd, n, false, g);
			for( int i = 0; i < nweights; i++ ) deriv[i] += g[i];
		
			return 0;
		}
		
		virtual int correlations(TrainingData &td, PredictedData &pd) = 0;
		
		virtual int CorrectClass(TrainingData &td, PredictedData &pd) = 0;
		
		virtual int ErrorSquared(TrainingData &td, PredictedData &pd) = 0;
		
		virtual int logZ(float *alpha, float logdetB, TrainingData &td, PredictedData &pd, float &logZ) = 0;
		
		virtual int forward(TrainingData &td, PredictedData &pd, int n, bool incerr) = 0;
		
		virtual int forwardOne(int ntime, float *in, float *out) = 0;
		
		virtual int logLike(TrainingData &td, PredictedData &pd, float &logL, bool doforward) = 0;
		
		virtual int logLike(TrainingData &td, PredictedData &pd, float &logL) = 0;
		
		virtual int logC(TrainingData &td, float &logC) = 0;
		
		virtual int backward(float *u, TrainingData &td, PredictedData &pd, int n, bool incerr, float *grad) = 0;
		
		virtual int backward(float *ul, float *us, TrainingData &td, PredictedData &pd, int n, bool incerr, float *gradl, float *grads) = 0;
		
		virtual int grad(TrainingData &td, PredictedData &pd, float *grad) = 0;
		
		virtual int Rforward(float *Cube, TrainingData &td, PredictedData &pd, int n, bool incerr, float *Rv) = 0;
		
		virtual int Rforward(float *Cube, TrainingData &td, PredictedData &pd, int n, bool incerr, float *Rv, float *Sv) = 0;
		
		virtual int Av(float *v, TrainingData &td, PredictedData &pd, float *Av) = 0;
		
		virtual int Av(float *v, TrainingData &td, PredictedData &pd, float alpha, float mu, float *Av) = 0;
		
		virtual float HHPscore(TrainingData &td, PredictedData &pd, bool doforward) = 0;
		
		virtual float HHPscore(TrainingData &td, PredictedData &pd) = 0;
};

class FeedForwardNeuralNetwork : public NeuralNetwork
{
	public:
		FeedForwardNeuralNetwork()
		{
			size_t _nlayers = 0;
			size_t *_nnodes = NULL;
			Init(_nlayers, _nnodes, false, false);
			loglikef = 1;
		}

		~FeedForwardNeuralNetwork()
		{
		}

		FeedForwardNeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, bool _norbias)
		{
			Init(_nlayers, _nnodes, _recurrent, _norbias);
			loglikef = 1;
		}

		FeedForwardNeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent)
		{
			Init(_nlayers, _nnodes, _recurrent, false);
			loglikef = 1;
		}

		FeedForwardNeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, bool _norbias, size_t _loglikef)
		{
			Init(_nlayers, _nnodes, _recurrent, _norbias);
			loglikef = _loglikef;
		}

		FeedForwardNeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, size_t _loglikef)
		{
			Init(_nlayers, _nnodes, _recurrent, false);
			loglikef = _loglikef;
		}
		
		FeedForwardNeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, bool _norbias, int *_linear)
		{
			Init(_nlayers, _nnodes, _recurrent, _norbias, _linear);
			loglikef = 1;
		}

		FeedForwardNeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, int *_linear)
		{
			Init(_nlayers, _nnodes, _recurrent, false, _linear);
			loglikef = 1;
		}

		FeedForwardNeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, bool _norbias, size_t _loglikef, int *_linear)
		{
			Init(_nlayers, _nnodes, _recurrent, _norbias, _linear);
			loglikef = _loglikef;
		}

		FeedForwardNeuralNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, size_t _loglikef, int *_linear)
		{
			Init(_nlayers, _nnodes, _recurrent, false, _linear);
			loglikef = _loglikef;
		}

		FeedForwardNeuralNetwork(const NeuralNetwork &n)
		{
			Clone(n);
			loglikef = n.loglikef;
		}
		
		
		int forward(TrainingData &td, PredictedData &pd, int n, bool incerr)
		{
			/*calculate the NN output*/
			
			if( loglikef == 2 ) incerr = false;
			
			int q  = -1, r = td.cuminsets[n]*totnnodes;
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				int p = td.cuminsets[n]*totnnodes + t*totnnodes; // starting index of non-recurrent nodes
				int rp = td.cuminsets[td.ndata]*totnnodes + n*totrnodes; // starting index of recurrent nodes
				int nrw = nnweights; // starting index of recurrent weights
				if( recurrent && t > 0 ) nrw += totrnodes;
				
				for( int i = 0; i < nlayers; i++ )
				{
					// set up the RNN initial state
					if( recurrent && !norbias && t == 0 && i > 0 && i < nlayers-1 )
					{
						for( int j = 0; j < rnodes[i]; j++ )
						{
							pd.in[rp] = 0E0;
							pd.out[rp] = pd.y[rp] = 1E0;
							rp++;
						}
					}
					
					for( int j = 0; j < nnodes[i]; j++ )
					{
						q++;
						pd.in[r+q] = 0E0;
						
						if( i < nlayers-1 && j == nnodes[i]-1 )
						{
							// node with the bias
							pd.out[r+q] = pd.y[r+q] = 1E0;
						}
						else
						{
							if( i > 0 )
							{
								pd.in[r+q] = 0E0;
								
								// contribution from non-recurrent connections
								for( int k = 0; k < nnodes[i-1]; k++ )
									pd.in[r+q] += w[i-1][j][k] * pd.out[p+k];
									
								// contribution from recurrent connections
								if( recurrent && i < nlayers-1 && j < nnodes[i]-1 )
								{
									if( t == 0 )
									{
										if( !norbias )
										{
											pd.in[r+q] += weights[nrw];
											nrw++;
										}
									}
									else
									{
										for( int k = 0; k < nnodes[i]-1; k++ )
										{
											pd.in[r+q] += weights[nrw] * pd.out[r+q-j-totnnodes+k];
											nrw++;
										}
									}
								}
								
								//pd.out[r+q] = linear[i] ? pd.in[r+q] : sigmoid(pd.in[r+q], 1);
								pd.out[r+q] = activation(linear[i], pd.in[r+q], 1);
								pd.y[r+q] = pd.out[r+q];
								if( i == nlayers-1 && incerr && t >= td.ntimeignore[n] ) pd.y[r+q] *= td.acc[td.cumoutsets[n]*td.nout+(t-td.ntimeignore[n])*td.nout+j];
							}
							else
							{
								pd.out[r+q] = pd.y[r+q] = td.inputs[td.cuminsets[n]*td.nin+t*td.nin+j];
							}
						}
					}
					
					if( i > 0 ) p += nnodes[i-1];
				}
			}
			
			q = td.cuminsets[n]*totnnodes+totnnodes-td.nout; // first output node at t = 0
			int j = td.cumoutsets[n]*td.nout;
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				for( int i = 0; i < td.nout; i++ )
				{
					if( t >= td.ntimeignore[n] )
					{
						pd.Chi[j] = ( pd.out[q+i] - td.outputs[j] ) * td.acc[j];
						
						j++;
					}
				}
					
				q += totnnodes;
			}
			return 0;
		}
		
		int forwardOne(int ntime, float *in, float *out)
		{
			/*calculate the NN output*/ 
			
			float pd_in[ntime*totnnodes+totrnodes], pd_out[ntime*totnnodes+totrnodes];
			int q  = -1;
			for( int t = 0; t < ntime; t++ )
			{
				int p = t*totnnodes; // starting index of non-recurrent nodes
				int rp = ntime*totnnodes; // starting index of recurrent nodes
				int nrw = nnweights; // starting index of recurrent weights
				if( recurrent && t > 0 ) nrw += totrnodes;
				
				for( int i = 0; i < nlayers; i++ )
				{
					// set up the RNN initial state
					if( recurrent && !norbias && t == 0 && i > 0 && i < nlayers-1 )
					{
						for( int j = 0; j < rnodes[i]; j++ )
						{
							pd_in[rp] = 0E0;
							pd_out[rp] = 1E0;
							rp++;
						}
					}
					
					for( int j = 0; j < nnodes[i]; j++ )
					{
						q++;
						pd_in[q] = 0E0;
						
						if( i < nlayers-1 && j == nnodes[i]-1 )
						{
							// node with the bias
							pd_out[q] = 1E0;
						}
						else
						{
							if( i > 0 )
							{
								pd_in[q] = 0E0;
								
								// contribution from non-recurrent connections
								for( int k = 0; k < nnodes[i-1]; k++ )
									pd_in[q] += w[i-1][j][k] * pd_out[p+k];
									
								// contribution from recurrent connections
								if( recurrent && i < nlayers-1 && j < nnodes[i]-1 )
								{
									if( t == 0 )
									{
										if( !norbias )
										{
											pd_in[q] += weights[nrw];
											nrw++;
										}
									}
									else
									{
										for( int k = 0; k < nnodes[i]-1; k++ )
										{
											pd_in[q] += weights[nrw] * pd_out[q-j-totnnodes+k];
											nrw++;
										}
									}
								}
								
								//pd_out[q] = linear[i] ? pd_in[q] : sigmoid(pd_in[q], 1);	
								pd_out[q] = activation(linear[i], pd_in[q], 1);
								if( i == nlayers-1 ) out[j] = pd_out[q];
							}
							else
							{
								pd_out[q] = in[j];
							}
						}
					}
					
					if( i > 0 ) p += nnodes[i-1];
				}
			}
		
			return 0;
		}
		
		int logLike(TrainingData &td, PredictedData &pd, float &logL)
		{
			bool doforward = true;
			logLike(td, pd, logL, doforward);
			return 0;
		}
		
		int logLike(TrainingData &td, PredictedData &pd, float &logL, bool doforward)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
					
			logL = 0E0;
			//for( int i = 0; i < td.nout; i++ ) pd.errsqr[i] = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				if( doforward ) forward(td, pd, n, false);
				
				int q = td.cumoutsets[n]*td.nout;
				int r = (td.cuminsets[n]+td.ntimeignore[n])*totnnodes+totnnodes-td.nout;
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					for( int i = 0; i < td.nout; i++ )
					{
						if( loglikef == 1 )
						{
							logL -= pow(pd.Chi[q], 2.0) / 2.0;
						}
						else if( loglikef == 2 )
						{
							logL += pow(pd.out[r+i] - td.outputs[q], 2.0);
						}
						
						//pd.errsqr[i] += pow(pd.out[r+i] - td.outputs[q], 2.0);
						
						q++;
					}
					r += totnnodes;
				}
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(&logL, 1, MPI_FLOAT, 0, myid*2, MPI_COMM_WORLD);
					//MPI_Send(pd.errsqr, td.nout, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						//float loglike, esqr[td.nout];
						float loglike;
						MPI_Status status;
						MPI_Recv(&loglike, 1, MPI_FLOAT, i, i*2, MPI_COMM_WORLD, &status);
						//MPI_Recv(esqr, td.nout, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						
						logL += loglike;
						//for( int j = 0; j < td.nout; j++ ) pd.errsqr[j] += esqr[j];
					}
				}
				
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(&logL, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
				//MPI_Bcast(pd.errsqr, td.nout, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
#endif
			
			if( loglikef == 2 ) logL = -sqrt( logL / float(td.cumoutsets[td.ndata]*td.nout) );
			//for( int i = 0; i < td.nout; i++ ) pd.errsqr[i] /= float(td.cumoutsets[td.ndata]);
		
			return 0;
		}
		
		float HHPscore(TrainingData &td, PredictedData &pd)
		{
			bool doforward = false;
			return HHPscore(td, pd, doforward);
		}
		
		float HHPscore(TrainingData &td, PredictedData &pd, bool doforward)
		{
			float score;
			size_t loglikeforig = loglikef;
			loglikef = 2;
			logLike(td, pd, score, doforward);
			loglikef = loglikeforig;
		
			return score;
		}
		
		int logC(TrainingData &td, float &logC)
		{
			logC = 0E0;
			
			if( loglikef == 1 )
			{
				logC = (float) (td.cumoutsets[td.ndata]*td.nout) * log(2.0*M_PI) / -2.0;
				for( size_t i = 0; i < td.cumoutsets[td.ndata]*td.nout; i++ ) logC += log(td.acc[i]);
			}
		
			return 0;
		}
		
		int backward(float *u, TrainingData &td, PredictedData &pd, int n, bool incerr, float *grad)
		{
			if( loglikef == 2 ) incerr = false;
			
			float dEdx[td.ntime[n]*totnnodes+totrnodes], dEdy[td.ntime[n]*totnnodes+totrnodes];
			
			for( int i = 0; i < nweights; i++ ) grad[i] = 0E0;
			
			int q = td.ntime[n]*totnnodes;
			int rq = td.ntime[n]*totnnodes+totrnodes;
			for( int t = td.ntime[n]-1; t >= 0; t-- )
			{
				int nrw = nweights;
				for( int i = nlayers-1; i >= 0; i-- )
				{
					q -= nnodes[i];
					if( recurrent && i < nlayers-1 && i > 0 ) nrw -= (int) pow(float(nnodes[i]-1), 2);
					int nrwc = nrw;
					
					for( int j = 0; j < nnodes[i]; j++ )
					{
						if( i == nlayers-1 )
						{
							if( t < td.ntimeignore[n] )
							{
								dEdy[q+j] = 0E0;
							}
							else
							{
								dEdy[q+j] = u[(t-td.ntimeignore[n])*td.nout+j];
								if( incerr ) dEdy[q+j] *= td.acc[td.cumoutsets[n]*td.nout+(t-td.ntimeignore[n])*td.nout+j];
							}
							//dEdx[q+j] = linear[i] ? dEdy[q+j] : sigmoid(pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
							dEdx[q+j] = activation(linear[i], pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
						}
						else
						{
							dEdy[q+j] = 0E0;
							
							// contribution from non-recurrent connections
							int kend = i == nlayers-2 ? nnodes[i+1] : nnodes[i+1]-1;
							for( int k = 0; k < kend; k++ )
								dEdy[q+j] += w[i][k][j] * dEdx[q+nnodes[i]+k];
							
							// contribution from recurrent connections	
							if( recurrent && i > 0 && i < nlayers-1 && t < td.ntime[n]-1 && j < nnodes[i]-1 )
							{
								nrwc = nrw + j;
								for( int k = 0; k < nnodes[i]-1; k++ )
								{
									dEdy[q+j] += weights[nrwc] * dEdx[q+totnnodes+k];
									nrwc += nnodes[i]-1;
								}
							}
							
							if( i == 0 || j == nnodes[i]-1 )
								dEdx[q+j] = 0E0;		// nodes with the bias
							else
								//dEdx[q+j] = linear[i] ? dEdy[q+j] : sigmoid(pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
								dEdx[q+j] = activation(linear[i], pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
						}
					}
					
					if( recurrent && !norbias && t == 0 && i > 0 && i < nlayers-1 )
					{
						rq -= rnodes[i];
						int nrwc = nnweights;
						for( size_t k = 0; k < i-1; k++ ) nrwc += rnodes[k];
						
						for( int j = 0; j < rnodes[i]; j++ )
						{
							dEdy[rq+j] = weights[nrwc] * dEdx[q+j];
							dEdx[rq+j] = 0E0;
							nrwc++;
						}
					}
				}
			}
			
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				int q = t*totnnodes + nnodes[0]; // index of first node in the hidden layer
				int r = td.cuminsets[n]*totnnodes + t*totnnodes;
				int p = -1;
				int nrw = nnweights-1;
				if( recurrent && t > 0 ) nrw += totrnodes;
				
				for( int i = 1; i < nlayers; i++ )
				{
					if( recurrent && i < nlayers-1 )
					{
						for( int j = 0; j < nnodes[i]-1; j++ )
						{
							if( t == 0 )
							{
								if( !norbias )
								{
									nrw++;
									grad[nrw] = dEdx[q+j];
								}
							}
							else
							{
								for( int k = 0; k < nnodes[i]-1; k++ )
								{
									nrw++;
									grad[nrw] += pd.y[r+nnodes[i-1]-totnnodes+k] * dEdx[q+j];
								}
							}
						}
					}
					
					int jend = i == nlayers-1 ? nnodes[i] : nnodes[i]-1;
					for( int j = 0; j < jend; j++ )
					{
						for( int k = 0; k < nnodes[i-1]; k++ )
						{
							p++;
							grad[p] += pd.y[r+k] * dEdx[q+j];
						}
					}
					q += nnodes[i];
					r += nnodes[i-1];
				}
			}
		
			return 0;
		}
		
		int backward(float *ul, float *us, TrainingData &td, PredictedData &pd, int n, bool incerr, float *gradl, float *grads)
		{
			if( loglikef == 2 ) incerr = false;
			
			float dEdx[td.ntime[n]*totnnodes+totrnodes], dEdy[td.ntime[n]*totnnodes+totrnodes];
			
			float dSdx[td.ntime[n]*totnnodes+totrnodes], dSdy[td.ntime[n]*totnnodes+totrnodes];
			for( int i = 0; i < td.ntime[n]*totnnodes+totrnodes; i++ ) dSdx[i] = dSdy[i] = 0E0;
			
			for( int i = 0; i < nweights; i++ ) gradl[i] = grads[i] = 0E0;
			
			int q = td.ntime[n]*totnnodes;
			int rq = td.ntime[n]*totnnodes+totrnodes;
			for( int t = td.ntime[n]-1; t >= 0; t-- )
			{
				int nrw = nweights;
				for( int i = nlayers-1; i >= 0; i-- )
				{
					q -= nnodes[i];
					if( recurrent && i < nlayers-1 && i > 0 ) nrw -= (int) pow(float(nnodes[i]-1), 2);
					int nrwc = nrw;
					
					for( int j = 0; j < nnodes[i]; j++ )
					{
						if( i == nlayers-1 )
						{
							if( t < td.ntimeignore[n] )
							{
								dEdy[q+j] = 0E0;
							}
							else
							{
								dEdy[q+j] = ul[(t-td.ntimeignore[n])*td.nout+j];
								if( incerr ) dEdy[q+j] *= td.acc[td.cumoutsets[n]*td.nout+(t-td.ntimeignore[n])*td.nout+j];
							}
							//dEdx[q+j] = linear[i] ? dEdy[q+j] : sigmoid(pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
							dEdx[q+j] = activation(linear[i], pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
						}
						else
						{
							dEdy[q+j] = 0E0;
							if( nlayers > 2 && i == nlayers-2 && j != nnodes[i]-1 ) dSdy[q+j] = us[t*(nnodes[nlayers-2]-1)+j];
							
							// contribution from non-recurrent connections
							int kend = i == nlayers-2 ? nnodes[i+1] : nnodes[i+1]-1;
							for( int k = 0; k < kend; k++ )
							{
								dEdy[q+j] += w[i][k][j] * dEdx[q+nnodes[i]+k];
								if( nlayers > 2 && i < nlayers-2 ) dSdy[q+j] += w[i][k][j] * dSdx[q+nnodes[i]+k];
							}
							
							// contribution from recurrent connections	
							if( recurrent && i > 0 && i < nlayers-1 && t < td.ntime[n]-1 && j < nnodes[i]-1 )
							{
								nrwc = nrw + j;
								for( int k = 0; k < nnodes[i]-1; k++ )
								{
									dEdy[q+j] += weights[nrwc] * dEdx[q+totnnodes+k];
									if( nlayers > 2 ) dSdy[q+j] += weights[nrwc] * dSdx[q+totnnodes+k];
									nrwc += nnodes[i]-1;
								}
							}
							
							if( i == 0 || j == nnodes[i]-1 )
							{
								dEdx[q+j] = 0E0;		// nodes with the bias
							}
							else
							{
								//dEdx[q+j] = linear[i] ? dEdy[q+j] : sigmoid(pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
								//if( nlayers > 2 ) dSdx[q+j] = linear[i] ? dSdy[q+j] : sigmoid(pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dSdy[q+j];
								dEdx[q+j] = activation(linear[i], pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
								if( nlayers > 2 ) dSdx[q+j] = activation(linear[i], pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dSdy[q+j];
							}
						}
					}
					
					if( recurrent && !norbias && t == 0 && i > 0 && i < nlayers-1 )
					{
						rq -= rnodes[i];
						int nrwc = nnweights;
						for( size_t k = 0; k < i-1; k++ ) nrwc += rnodes[k];
						
						for( int j = 0; j < rnodes[i]; j++ )
						{
							dEdy[rq+j] = weights[nrwc] * dEdx[q+j];
							dEdx[rq+j] = 0E0;
							dSdy[rq+j] = weights[nrwc] * dSdx[q+j];
							dSdx[rq+j] = 0E0;
							nrwc++;
						}
					}
				}
			}
			
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				int q = t*totnnodes + nnodes[0]; // index of first node in the hidden layer
				int r = td.cuminsets[n]*totnnodes + t*totnnodes;
				int p = -1;
				int nrw = nnweights-1;
				if( recurrent && t > 0 ) nrw += totrnodes;
				
				for( int i = 1; i < nlayers; i++ )
				{
					if( recurrent && i < nlayers-1 )
					{
						for( int j = 0; j < nnodes[i]-1; j++ )
						{
							if( t == 0 )
							{
								if( !norbias )
								{
									nrw++;
									gradl[nrw] = dEdx[q+j];
									grads[nrw] = dSdx[q+j];
								}
							}
							else
							{
								for( int k = 0; k < nnodes[i]-1; k++ )
								{
									nrw++;
									gradl[nrw] += pd.y[r+nnodes[i-1]-totnnodes+k] * dEdx[q+j];
									grads[nrw] += pd.y[r+nnodes[i-1]-totnnodes+k] * dSdx[q+j];
								}
							}
						}
					}
					
					int jend = i == nlayers-1 ? nnodes[i] : nnodes[i]-1;
					for( int j = 0; j < jend; j++ )
					{
						for( int k = 0; k < nnodes[i-1]; k++ )
						{
							p++;
							gradl[p] += pd.y[r+k] * dEdx[q+j];
							if( nlayers > 2 && i < nlayers-1 ) grads[p] += pd.y[r+k] * dSdx[q+j];
						}
					}
					q += nnodes[i];
					r += nnodes[i-1];
				}
			}
		
			return 0;
		}
		
		int grad(TrainingData &td, PredictedData &pd, float *grad)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			float f = 1E0;
			if( loglikef == 2 )
			{
				logLike(td, pd, f, false);
				f *= float(td.cumoutsets[td.ndata]*td.nout);
			}
			
			float g[nweights];
			for( int i = 0; i < nweights; i++ ) grad[i] = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				float e[(td.ntime[n]-td.ntimeignore[n])*td.nout]; // dE/dy where E is the error function
				int r = td.cuminsets[n]*totnnodes+(td.ntimeignore[n]+1)*totnnodes-td.nout;
				size_t k = 0;
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					for( int i = 0; i < td.nout; i++ )
					{
						if( loglikef == 1 )
						{
							e[k] = -pd.Chi[td.cumoutsets[n]*td.nout+k];
						}
						else if( loglikef == 2 )
						{
							e[k] = ( pd.out[r+i] - td.outputs[td.cumoutsets[n]*td.nout+k] ) / f;
						}
						k++;
					}
					r += totnnodes;
				}
				backward(e, td, pd, n, true, g);
				for( int i = 0; i < nweights; i++ ) if( !setzero[i] ) grad[i] += g[i];
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(grad, nweights, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float gradient[nweights];
						MPI_Status status;
						MPI_Recv(gradient, nweights, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						for( int j = 0; j < nweights; j++ ) grad[j] += gradient[j];
					}
				}
				
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(grad, nweights, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
#endif
		
			return 0;
		}
		
		int Rforward(float *Cube, TrainingData &td, PredictedData &pd, int n, bool incerr, float *Rv)
		{
			if( loglikef == 2 ) incerr = false;
			
			float Rin[td.ntime[n]*totnnodes], Rout[td.ntime[n]*totnnodes];
			for( int i = 0; i < td.ntime[n]*totnnodes; i++ ) Rin[i] = Rout[i] = 0E0;
			
			/*calculate the R derivative*/
			
			int q  = -1, r = td.cuminsets[n]*totnnodes;
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				int p = td.cuminsets[n]*totnnodes + t*totnnodes;
				int nrw = nnweights; // starting index of recurrent weights
				if( recurrent && t > 0 ) nrw += totrnodes;
				int s = -1; // starting index of the vector
				
				for( int i = 0; i < nlayers; i++ )
				{
					for( int j = 0; j < nnodes[i]; j++ )
					{
						q++;
						
						if( i > 0 && !( i < nlayers-1 && j == nnodes[i]-1 ) ) // not a bias or input layer node
						{
							// contribution from non-recurrent connections
							for( int k = 0; k < nnodes[i-1]; k++ )
							{
								s++;
								Rin[q] += w[i-1][j][k] * Rout[p-r+k] + Cube[s] * pd.y[p+k];
							}
								
							// contribution from recurrent connections
							if( recurrent && i < nlayers-1 && j < nnodes[i]-1 )
							{
								if( t == 0 )
								{
									if( !norbias )
									{
										Rin[q] += Cube[nrw];
										nrw++;
									}
								}
								else
								{
									for( int k = 0; k < nnodes[i]-1; k++ )
									{
										Rin[q] += weights[nrw] * Rout[q-j-totnnodes+k] + Cube[nrw] * pd.y[r+q-j-totnnodes+k];
										nrw++;
									}
								}
							}
							
							//Rout[q] = linear[i] ? Rin[q] : Rin[q] * sigmoid(pd.in[r+q], 2);
							Rout[q] = activation(linear[i], pd.in[r+q], 2) * Rin[q];
							if( i == nlayers-1 && incerr && t >= td.ntimeignore[n] ) Rout[q] *= td.acc[td.cumoutsets[n]*td.nout+(t-td.ntimeignore[n])*td.nout+j];
						}
					}
					if( i > 0 ) p += nnodes[i-1];
				}
			}
			
			for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
			{
				int q = (t+1)*totnnodes-td.nout;
				for( int i = 0; i < td.nout; i++ )
					Rv[(t-td.ntimeignore[n])*td.nout+i] = Rout[q+i];
			}
			
			return 0;
		}
		
		int Rforward(float *Cube, TrainingData &td, PredictedData &pd, int n, bool incerr, float *Rv, float *Sv)
		{
			if( loglikef == 2 ) incerr = false;
			
			float Rin[td.ntime[n]*totnnodes], Rout[td.ntime[n]*totnnodes];
			for( int i = 0; i < td.ntime[n]*totnnodes; i++ ) Rin[i] = Rout[i] = 0E0;
			
			/*calculate the R derivative*/
			
			int q  = -1, r = td.cuminsets[n]*totnnodes;
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				int p = td.cuminsets[n]*totnnodes + t*totnnodes;
				int nrw = nnweights; // starting index of recurrent weights
				if( recurrent && t > 0 ) nrw += totrnodes;
				int s = -1; // starting index of the vector
				
				for( int i = 0; i < nlayers; i++ )
				{
					for( int j = 0; j < nnodes[i]; j++ )
					{
						q++;
						
						if( i > 0 && !( i < nlayers-1 && j == nnodes[i]-1 ) ) // not a bias or input layer node
						{
							// contribution from non-recurrent connections
							for( int k = 0; k < nnodes[i-1]; k++ )
							{
								s++;
								Rin[q] += w[i-1][j][k] * Rout[p-r+k] + Cube[s] * pd.y[p+k];
							}
								
							// contribution from recurrent connections
							if( recurrent && i < nlayers-1 && j < nnodes[i]-1 )
							{
								if( t == 0 )
								{
									if( !norbias )
									{
										Rin[q] += Cube[nrw];
										nrw++;
									}
								}
								else
								{
									for( int k = 0; k < nnodes[i]-1; k++ )
									{
										Rin[q] += weights[nrw] * Rout[q-j-totnnodes+k] + Cube[nrw] * pd.y[r+q-j-totnnodes+k];
										nrw++;
									}
								}
							}
							
							//Rout[q] = linear[i] ? Rin[q] : Rin[q] * sigmoid(pd.in[r+q], 2);
							Rout[q] = activation(linear[i], pd.in[r+q], 2) * Rin[q];
							if( i == nlayers-1 && incerr && t >= td.ntimeignore[n] ) Rout[q] *= td.acc[td.cumoutsets[n]*td.nout+(t-td.ntimeignore[n])*td.nout+j];
						}
					}
					if( i > 0 ) p += nnodes[i-1];
				}
			}
			
			for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
			{
				int q = (t+1)*totnnodes-td.nout;
				for( int i = 0; i < td.nout; i++ )
					Rv[(t-td.ntimeignore[n])*td.nout+i] = Rout[q+i];
			}
			
			if( nlayers > 2 )
			{
				for( int t = 0; t < td.ntime[n]; t++ )
				{
					int q = (t+1)*totnnodes-td.nout-nnodes[nlayers-2];
					for( int i = 0; i < nnodes[nlayers-2]-1; i++ )
						Sv[t*(nnodes[nlayers-2]-1)+i] = Rout[q+i];
				}
			}
			
			return 0;
		}
		
		int Av(float *v, TrainingData &td, PredictedData &pd, float *Av)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			
			float Av1[nweights];
			for( int i = 0; i < nweights; i++ )
			{
				Av[i] = Av1[i] = 0E0;
				if( setzero[i] ) v[i] = 0E0;
			}
			
			for( int n = nstart; n < nend; n++ )
			{
				float Rv[(td.ntime[n]-td.ntimeignore[n])*td.nout];
				for( int i = 0; i < (td.ntime[n]-td.ntimeignore[n])*td.nout; i++ ) Rv[i] = 0E0;
				
				// calculate R.v
				Rforward(v, td, pd, n, true, Rv);

				// calculate R^{t}.R.v
				backward(Rv, td, pd, n, true, Av1);
				for( int i = 0; i < nweights; i++ ) if( !setzero[i] ) Av[i] += Av1[i];
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(Av, nweights, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float Adotv[nweights];
						MPI_Status status;
						MPI_Recv(Adotv, nweights, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						for( int j = 0; j < nweights; j++ ) Av[j] += Adotv[j];
					}
				}
				
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(Av, nweights, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
#endif

			return 0;
		}
		
		int Av(float *v, TrainingData &td, PredictedData &pd, float alpha, float mu, float *Av)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			
			float Av1[nweights], Sv1[nweights];
			for( int i = 0; i < nweights; i++ )
			{
				Av[i] = Av1[i] = Sv1[i] = 0E0;
				if( setzero[i] ) v[i] = 0E0;
			}
			
			for( int n = nstart; n < nend; n++ )
			{
				float Rv[(td.ntime[n]-td.ntimeignore[n])*td.nout];
				for( int i = 0; i < (td.ntime[n]-td.ntimeignore[n])*td.nout; i++ ) Rv[i] = 0E0;
				int k = nlayers == 2 ? 0 : td.ntime[n]*(nnodes[nlayers-2]-1);
				float Sv[k];
				for( int i = 0; i < k; i++ ) Sv[i] = 0E0;
				
				// calculate R.v
				Rforward(v, td, pd, n, true, Rv, Sv);
				// calculate R^{t}.R.v
				backward(Rv, Sv, td, pd, n, true, Av1, Sv1);
				for( int i = 0; i < nweights; i++ ) if( !setzero[i] ) Av[i] += Av1[i] + alpha*mu*Sv1[i];
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(Av, nweights, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float Adotv[nweights];
						MPI_Status status;
						MPI_Recv(Adotv, nweights, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						for( int j = 0; j < nweights; j++ ) Av[j] += Adotv[j];
					}
				}
				
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(Av, nweights, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
#endif

			return 0;
		}
		
		int correlations(TrainingData &td, PredictedData &pd)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			double sumxy[td.nout], sumx[td.nout], sumx2[td.nout], sumy[td.nout], sumy2[td.nout];
			
			for( int i = 0; i < td.nout; i++ ) pd.corr[i] = sumxy[i] = sumx[i] = sumx2[i] = sumy[i] = sumy2[i] = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				int p = td.cuminsets[n]*totnnodes;
				int l = td.cumoutsets[n]*td.nout;
				
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					int q = (t+1)*totnnodes-td.nout;
					
					for( int i = 0; i < td.nout; i++ )
					{
						double y = pd.out[p+q+i]/double(td.cumoutsets[td.ndata]);
						double x = td.outputs[l]/double(td.cumoutsets[td.ndata]);
						sumx[i] += x;
						sumx2[i] += x*x;
						sumy[i] += y;
						sumy2[i] += y*y;
						sumxy[i] += x*y;
						
						l++;
					}
				}
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
		
				if( myid != 0 )
				{
					MPI_Send(sumx, td.nout, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
					MPI_Send(sumx2, td.nout, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
					MPI_Send(sumy, td.nout, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
					MPI_Send(sumy2, td.nout, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
					MPI_Send(sumxy, td.nout, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						double sx[td.nout], sx2[td.nout], sy[td.nout], sy2[td.nout], sxy[td.nout];
						MPI_Status status;
						MPI_Recv(sx, td.nout, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
						MPI_Recv(sx2, td.nout, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
						MPI_Recv(sy, td.nout, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
						MPI_Recv(sy2, td.nout, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
						MPI_Recv(sxy, td.nout, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
	
						for( int j = 0; j < td.nout; j++ )
						{
							sumx[j] += sx[j];
							sumx2[j] += sx2[j];
							sumy[j] += sy[j];
							sumy2[j] += sy2[j];
							sumxy[j] += sxy[j];
						}
					}
				}
			}
#endif
			
			if( myid == 0 )
			{
				for( int i = 0; i < td.nout; i++ )
					pd.corr[i] = (float) ( td.cumoutsets[td.ndata] * sumxy[i] - sumx[i] * sumy[i] ) / ( sqrt( td.cumoutsets[td.ndata] * sumx2[i] - pow(sumx[i], 2.0) ) * sqrt( td.cumoutsets[td.ndata] * sumy2[i] - pow(sumy[i], 2.0) ) );
			}
			
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(pd.corr, td.nout, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
				
			return 0;
		}
		
		int ErrorSquared(TrainingData &td, PredictedData &pd)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			for( int i = 0; i < td.nout; i++ ) pd.errsqr[i] = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				int p = td.cuminsets[n]*totnnodes;
				int l = td.cumoutsets[n]*td.nout;
				
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					int q = (t+1)*totnnodes-td.nout;
					
					for( int i = 0; i < td.nout; i++ )
					{
						float y = pd.out[p+q+i];
						pd.errsqr[i] += pow(td.outputs[l] - y, 2.0);
						
						l++;
					}
				}
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
		
				if( myid != 0 )
				{
					MPI_Send(pd.errsqr, td.nout, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float esqr[td.nout];
						MPI_Status status;
						MPI_Recv(esqr, td.nout, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
	
						for( int j = 0; j < td.nout; j++ ) pd.errsqr[j] += esqr[j];
					}
				}
			}
#endif
			
			if( myid == 0 )
			{
				for( int i = 0; i < td.nout; i++ )
					pd.errsqr[i] /= float(td.cumoutsets[td.ndata]);
			}
			
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(pd.errsqr, td.nout, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
				
			return 0;
		}
		
		int logZ(float *alpha, float logdetB, TrainingData &td, PredictedData &pd, float &logZ)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			logPrior(alpha, logZ);
			
			float logL = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				int q = td.cumoutsets[n]*td.nout;
				int r = (td.cuminsets[n]+td.ntimeignore[n])*totnnodes+totnnodes-td.nout;
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					for( int i = 0; i < td.nout; i++ )
					{
						if( loglikef == 1 )
						{
							logL -= pow(pd.Chi[q], 2.0) / 2.0;
						}
						else if( loglikef == 2 )
						{
							logL += pow(pd.out[r+i] - td.outputs[q], 2.0);
						}
						
						q++;
					}
					r += totnnodes;
				}
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(&logL, 1, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float loglike;
						MPI_Status status;
						MPI_Recv(&loglike, 1, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						logL += loglike;
					}
				}
			}
#endif
			
			logZ += logL;
			
			if( loglikef == 1 )
			{
				logZ += -logdetB/2.0 - td.cumoutsets[td.ndata]*td.nout * log(2.0*M_PI) / 2.0;
				for( int i = 0; i < td.cumoutsets[td.ndata]*td.nout; i++ ) logZ += log(td.acc[i]) / 2.0;
			}
			logZ += nweights * log(alpha[0]) / 2.0;
			
			/*if( nlayers > 2 )
			{
				float logG = -FLT_MAX;
				for( int i = 1; i < nlayers-1; i++ )
				{
					logG = LogSumExp(logG, lognfact(nnodes[i]) + nnodes[i] * log(2.0));
					if( recurrent )
					{
						int k = pow(float(nnodes[i]-1), 2);
						logG = LogSumExp(logG, lognfact(k) + k * log(2.0));
					}
				}
				logZ += logG;
			}*/
				
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(&logZ, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
		
			return 0;
		}
		
		int CorrectClass(TrainingData &td, PredictedData &pd)
		{
			std::cerr << "should not be calling this function for regression net.\n";
			abort();
		}
};

class FeedForwardClassNetwork : public NeuralNetwork
{
	public:
		FeedForwardClassNetwork()
		{
			size_t _nlayers = 0;
			size_t *_nnodes = NULL;
			Init(_nlayers, _nnodes, false, false);
			loglikef = 1;
		}

		~FeedForwardClassNetwork()
		{
		}

		FeedForwardClassNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, bool _norbias)
		{
			Init(_nlayers, _nnodes, _recurrent, _norbias);
			loglikef = 1;
		}

		FeedForwardClassNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent)
		{
			Init(_nlayers, _nnodes, _recurrent, false);
			loglikef = 1;
		}

		FeedForwardClassNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, bool _norbias, int *_linear)
		{
			Init(_nlayers, _nnodes, _recurrent, _norbias, _linear);
			loglikef = 1;
		}

		FeedForwardClassNetwork(size_t _nlayers, size_t *_nnodes, bool _recurrent, int *_linear)
		{
			Init(_nlayers, _nnodes, _recurrent, false, _linear);
			loglikef = 1;
		}

		FeedForwardClassNetwork(const NeuralNetwork &n)
		{
			Clone(n);
			loglikef = 1;
		}
		
		int forward(TrainingData &td, PredictedData &pd, int n, bool incerr)
		{
			incerr = false;
			/*calculate the NN output*/
			
			int q  = -1, r = td.cuminsets[n]*totnnodes;
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				int p = td.cuminsets[n]*totnnodes + t*totnnodes; // starting index of non-recurrent nodes
				int rp = td.cuminsets[td.ndata]*totnnodes + n*totrnodes; // starting index of recurrent nodes
				int nrw = nnweights; // starting index of recurrent weights
				if( recurrent && t > 0 ) nrw += totrnodes;
				
				for( int i = 0; i < nlayers; i++ )
				{
					// set up the RNN initial state
					if( recurrent && !norbias && t == 0 && i > 0 && i < nlayers-1 )
					{
						for( int j = 0; j < rnodes[i]; j++ )
						{
							pd.in[rp] = 0E0;
							pd.out[rp] = 1E0;
							rp++;
						}
					}
					
					for( int j = 0; j < nnodes[i]; j++ )
					{
						q++;
						pd.in[r+q] = 0E0;
						
						if( i < nlayers-1 && j == nnodes[i]-1 )
						{
							// node with the bias
							pd.out[r+q] = 1E0;
						}
						else
						{
							if( i > 0 )
							{
								pd.in[r+q] = 0E0;
								
								// contribution from non-recurrent connections
								for( int k = 0; k < nnodes[i-1]; k++ )
									pd.in[r+q] += w[i-1][j][k] * pd.out[p+k];
									
								// contribution from recurrent connections
								if( recurrent && i < nlayers-1 && j < nnodes[i]-1 )
								{
									if( t == 0 )
									{
										if( !norbias )
										{
											pd.in[r+q] += weights[nrw];
											nrw++;
										}
									}
									else
									{
										for( int k = 0; k < nnodes[i]-1; k++ )
										{
											pd.in[r+q] += weights[nrw] * pd.out[r+q-j-totnnodes+k];
											nrw++;
										}
									}
								}
								
								//pd.out[r+q] = linear[i] ? pd.in[r+q] : sigmoid(pd.in[r+q], 1);
								pd.out[r+q] = activation(linear[i], pd.in[r+q], 1);
								
								if( i == nlayers-1 )
								{
									if( j == td.nout-1 )
									{
										float out[td.nout];
										int s = r+(t+1)*totnnodes-td.nout;
										size_t g = 0;
										for( int h = 0; h < td.ncat; h++ )
										{
											for( int k = g; k < g+td.nclasses[h]; k++ )
											{
												out[k] = 0E0;
												for( int m = g; m < g+td.nclasses[h]; m++ )
													out[k] += exp( pd.out[s+m] - pd.out[s+k] );
											}
											
											g += td.nclasses[h];
										}
										for( int k = 0; k < td.nout; k++ ) pd.out[s+k] = 1E0 / out[k];
									}
								}
							}
							else
							{
								pd.out[r+q] = td.inputs[td.cuminsets[n]*td.nin+t*td.nin+j];
							}
						}
					}
					
					if( i > 0 ) p += nnodes[i-1];
				}
			}
			
			return 0;
		}
		
		int forwardOne(int ntime, float *in, float *out)
		{
			/*calculate the NN output*/ 
			
			float pd_in[ntime*totnnodes+totrnodes], pd_out[ntime*totnnodes+totrnodes];
			int q  = -1;
			for( int t = 0; t < ntime; t++ )
			{
				int p = t*totnnodes; // starting index of non-recurrent nodes
				int rp = ntime*totnnodes; // starting index of recurrent nodes
				int nrw = nnweights; // starting index of recurrent weights
				if( recurrent && t > 0 ) nrw += totrnodes;
				
				for( int i = 0; i < nlayers; i++ )
				{
					// set up the RNN initial state
					if( recurrent && !norbias && t == 0 && i > 0 && i < nlayers-1 )
					{
						for( int j = 0; j < rnodes[i]; j++ )
						{
							pd_in[rp] = 0E0;
							pd_out[rp] = 1E0;
							rp++;
						}
					}
					
					for( int j = 0; j < nnodes[i]; j++ )
					{
						q++;
						pd_in[q] = 0E0;
						
						if( i < nlayers-1 && j == nnodes[i]-1 )
						{
							// node with the bias
							pd_out[q] = 1E0;
						}
						else
						{
							if( i > 0 )
							{
								pd_in[q] = 0E0;
								
								// contribution from non-recurrent connections
								for( int k = 0; k < nnodes[i-1]; k++ )
									pd_in[q] += w[i-1][j][k] * pd_out[p+k];
									
								// contribution from recurrent connections
								if( recurrent && i < nlayers-1 && j < nnodes[i]-1 )
								{
									if( t == 0 )
									{
										if( !norbias )
										{
											pd_in[q] += weights[nrw];
											nrw++;
										}
									}
									else
									{
										for( int k = 0; k < nnodes[i]-1; k++ )
										{
											pd_in[q] += weights[nrw] * pd_out[q-j-totnnodes+k];
											nrw++;
										}
									}
								}
								
								//pd_out[q] = linear[i] ? pd_in[q] : sigmoid(pd_in[q], 1);
								pd_out[q] = activation(linear[i], pd_in[q], 1);
								
								if( i == nlayers-1 )
								{
									if( j == nnodes[i]-1 )
									{
										float temp[nnodes[i]];
										int r = (t+1)*totnnodes-nnodes[i];
										for( int k = 0; k < nnodes[i]; k++ )
										{
											temp[k] = 0E0;
											for( int m = 0; m < nnodes[i]; m++ )
												temp[k] += exp( pd_out[r+m] - pd_out[r+k] );
										}
										for( int k = 0; k < nnodes[i]; k++ )
										{
											pd_out[r+k] = 1E0 / temp[k];
											out[k] = pd_out[r+k];
										}
									}
								}
							}
							else
							{
								pd_out[q] = in[j];
							}
						}
					}
					
					if( i > 0 ) p += nnodes[i-1];
				}
			}
		
			return 0;
		}
		
		int logLike(TrainingData &td, PredictedData &pd, float &logL)
		{
			bool doforward = true;
			logLike(td, pd, logL, doforward);
			return 0;
		}
		
		int logLike(TrainingData &td, PredictedData &pd, float &logL, bool doforward)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
					
			logL = 0E0;
			//for( int i = 0; i < td.nout; i++ ) pd.errsqr[i] = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				if( doforward ) forward(td, pd, n, false);
				
				int q = td.cumoutsets[n]*td.nout;
				int r = (td.cuminsets[n]+td.ntimeignore[n])*totnnodes+totnnodes-td.nout;
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					for( int i = 0; i < td.nout; i++ )
					{
						logL += td.outputs[q] * log(pd.out[r+i]);
						//pd.errsqr[i] += pow(pd.out[r+i] - td.outputs[q], 2.0);
						
						q++;
					}
					r += totnnodes;
				}
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(&logL, 1, MPI_FLOAT, 0, myid*2, MPI_COMM_WORLD);
					//MPI_Send(pd.errsqr, td.nout, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						//float loglike, esqr[td.nout];
						float loglike;
						MPI_Status status;
						MPI_Recv(&loglike, 1, MPI_FLOAT, i, i*2, MPI_COMM_WORLD, &status);
						//MPI_Recv(esqr, td.nout, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						
						logL += loglike;
						//for( int j = 0; j < td.nout; j++ ) pd.errsqr[j] += esqr[j];
					}
				}
				
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(&logL, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
				//MPI_Bcast(pd.errsqr, td.nout, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
#endif
			//for( int i = 0; i < td.nout; i++ ) pd.errsqr[i] /= float(td.cumoutsets[td.ndata]);
			
			return 0;
		}
		
		float HHPscore(TrainingData &td, PredictedData &pd)
		{
			return 0E0;
		}
		
		float HHPscore(TrainingData &td, PredictedData &pd, bool doforward)
		{
			return 0E0;
		}
		
		int logC(TrainingData &td, float &logC)
		{
			logC = (float) (td.cumoutsets[td.ndata]*td.nout) * log(2.0*M_PI) / -2.0;
			for( size_t i = 0; i < td.cumoutsets[td.ndata]*td.nout; i++ ) logC += log(td.acc[i]);
		
			return 0;
		}
		
		int backward(float *u, TrainingData &td, PredictedData &pd, int n, bool incerr, float *grad)
		{
			incerr = false;
			
			float dEdx[td.ntime[n]*totnnodes+totrnodes], dEdy[td.ntime[n]*totnnodes+totrnodes];
			
			for( int i = 0; i < nweights; i++ ) grad[i] = 0E0;
			
			int q = td.ntime[n]*totnnodes;
			int rq = td.ntime[n]*totnnodes+totrnodes;
			for( int t = td.ntime[n]-1; t >= 0; t-- )
			{
				int nrw = nweights;
				for( int i = nlayers-1; i >= 0; i-- )
				{
					q -= nnodes[i];
					if( recurrent && i < nlayers-1 && i > 0 ) nrw -= (int) pow(float(nnodes[i]-1), 2);
					int nrwc = nrw;
					
					for( int j = 0; j < nnodes[i]; j++ )
					{
						if( i == nlayers-1 )
						{
							if( t < td.ntimeignore[n] )
								dEdy[q+j] = 0E0;
							else
								dEdy[q+j] = u[(t-td.ntimeignore[n])*td.nout+j];
							//dEdx[q+j] = linear[i] ? dEdy[q+j] : sigmoid(pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
							dEdx[q+j] = activation(linear[i], pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
						}
						else
						{
							dEdy[q+j] = 0E0;
							
							// contribution from non-recurrent connections
							int kend = i == nlayers-2 ? nnodes[i+1] : nnodes[i+1]-1;
							for( int k = 0; k < kend; k++ )
								dEdy[q+j] += w[i][k][j] * dEdx[q+nnodes[i]+k];
							
							// contribution from recurrent connections	
							if( recurrent && i > 0 && i < nlayers-1 && t < td.ntime[n]-1 && j < nnodes[i]-1 )
							{
								nrwc = nrw + j;
								for( int k = 0; k < nnodes[i]-1; k++ )
								{
									dEdy[q+j] += weights[nrwc] * dEdx[q+totnnodes+k];
									nrwc += nnodes[i]-1;
								}
							}
							
							if( i == 0 || j == nnodes[i]-1 )
								dEdx[q+j] = 0E0;		// nodes with the bias
							else
								//dEdx[q+j] = linear[i] ? dEdy[q+j] : sigmoid(pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
								dEdx[q+j] = activation(linear[i], pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
						}
					}
					
					if( recurrent && !norbias && t == 0 && i > 0 && i < nlayers-1 )
					{
						rq -= rnodes[i];
						int nrwc = nnweights;
						for( size_t k = 0; k < i-1; k++ ) nrwc += rnodes[k];
						
						for( int j = 0; j < rnodes[i]; j++ )
						{
							dEdy[rq+j] = weights[nrwc] * dEdx[q+j];
							dEdx[rq+j] = 0E0;
							nrwc++;
						}
					}
				}
			}
			
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				int q = t*totnnodes + nnodes[0]; // index of first node in the hidden layer
				int r = td.cuminsets[n]*totnnodes + t*totnnodes;
				int p = -1;
				int nrw = nnweights-1;
				if( recurrent && t > 0 ) nrw += totrnodes;
				
				for( int i = 1; i < nlayers; i++ )
				{
					if( recurrent && i < nlayers-1 )
					{
						for( int j = 0; j < nnodes[i]-1; j++ )
						{
							if( t == 0 )
							{
								if( !norbias )
								{
									nrw++;
									grad[nrw] = dEdx[q+j];
								}
							}
							else
							{
								for( int k = 0; k < nnodes[i]-1; k++ )
								{
									nrw++;
									grad[nrw] += pd.out[r+nnodes[i-1]-totnnodes+k] * dEdx[q+j];
								}
							}
						}
					}
					
					int jend = i == nlayers-1 ? nnodes[i] : nnodes[i]-1;
					for( int j = 0; j < jend; j++ )
					{
						for( int k = 0; k < nnodes[i-1]; k++ )
						{
							p++;
							grad[p] += pd.out[r+k] * dEdx[q+j];
						}
					}
					q += nnodes[i];
					r += nnodes[i-1];
				}
			}
		
			return 0;
		}
		
		int backward(float *ul, float *us, TrainingData &td, PredictedData &pd, int n, bool incerr, float *gradl, float *grads)
		{
			incerr = false;
			
			float dEdx[td.ntime[n]*totnnodes+totrnodes], dEdy[td.ntime[n]*totnnodes+totrnodes];
			
			for( int i = 0; i < nweights; i++ ) gradl[i] = 0E0;
			
			int q = td.ntime[n]*totnnodes;
			int rq = td.ntime[n]*totnnodes+totrnodes;
			for( int t = td.ntime[n]-1; t >= 0; t-- )
			{
				int nrw = nweights;
				for( int i = nlayers-1; i >= 0; i-- )
				{
					q -= nnodes[i];
					if( recurrent && i < nlayers-1 && i > 0 ) nrw -= (int) pow(float(nnodes[i]-1), 2);
					int nrwc = nrw;
					
					for( int j = 0; j < nnodes[i]; j++ )
					{
						if( i == nlayers-1 )
						{
							if( t < td.ntimeignore[n] )
								dEdy[q+j] = 0E0;
							else
								dEdy[q+j] = ul[(t-td.ntimeignore[n])*td.nout+j];
							//dEdx[q+j] = linear[i] ? dEdy[q+j] : sigmoid(pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
							dEdx[q+j] = activation(linear[i], pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
						}
						else
						{
							dEdy[q+j] = 0E0;
							
							// contribution from non-recurrent connections
							int kend = i == nlayers-2 ? nnodes[i+1] : nnodes[i+1]-1;
							for( int k = 0; k < kend; k++ )
								dEdy[q+j] += w[i][k][j] * dEdx[q+nnodes[i]+k];
							
							// contribution from recurrent connections	
							if( recurrent && i > 0 && i < nlayers-1 && t < td.ntime[n]-1 && j < nnodes[i]-1 )
							{
								nrwc = nrw + j;
								for( int k = 0; k < nnodes[i]-1; k++ )
								{
									dEdy[q+j] += weights[nrwc] * dEdx[q+totnnodes+k];
									nrwc += nnodes[i]-1;
								}
							}
							
							if( i == 0 || j == nnodes[i]-1 )
								dEdx[q+j] = 0E0;		// nodes with the bias
							else
								//dEdx[q+j] = linear[i] ? dEdy[q+j] : sigmoid(pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
								dEdx[q+j] = activation(linear[i], pd.in[td.cuminsets[n]*totnnodes+q+j], 2) * dEdy[q+j];
						}
					}
					
					if( recurrent && !norbias && t == 0 && i > 0 && i < nlayers-1 )
					{
						rq -= rnodes[i];
						int nrwc = nnweights;
						for( size_t k = 0; k < i-1; k++ ) nrwc += rnodes[k];
						
						for( int j = 0; j < rnodes[i]; j++ )
						{
							dEdy[rq+j] = weights[nrwc] * dEdx[q+j];
							dEdx[rq+j] = 0E0;
							nrwc++;
						}
					}
				}
			}
			
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				int q = t*totnnodes + nnodes[0]; // index of first node in the hidden layer
				int r = td.cuminsets[n]*totnnodes + t*totnnodes;
				int p = -1;
				int nrw = nnweights-1;
				if( recurrent && t > 0 ) nrw += totrnodes;
				
				for( int i = 1; i < nlayers; i++ )
				{
					if( recurrent && i < nlayers-1 )
					{
						for( int j = 0; j < nnodes[i]-1; j++ )
						{
							if( t == 0 )
							{
								if( !norbias )
								{
									nrw++;
									gradl[nrw] = dEdx[q+j];
								}
							}
							else
							{
								for( int k = 0; k < nnodes[i]-1; k++ )
								{
									nrw++;
									gradl[nrw] += pd.out[r+nnodes[i-1]-totnnodes+k] * dEdx[q+j];
								}
							}
						}
					}
					
					int jend = i == nlayers-1 ? nnodes[i] : nnodes[i]-1;
					for( int j = 0; j < jend; j++ )
					{
						for( int k = 0; k < nnodes[i-1]; k++ )
						{
							p++;
							gradl[p] += pd.out[r+k] * dEdx[q+j];
						}
					}
					q += nnodes[i];
					r += nnodes[i-1];
				}
			}
		
			return 0;
		}
		
		int grad(TrainingData &td, PredictedData &pd, float *grad)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			float g[nweights];
			for( int i = 0; i < nweights; i++ ) grad[i] = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				float e[(td.ntime[n]-td.ntimeignore[n])*td.nout]; // dE/dy where E is the error function
				int r = td.cuminsets[n]*totnnodes+(td.ntimeignore[n]+1)*totnnodes-td.nout;
				size_t k = 0;
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					for( int i = 0; i < td.nout; i++ )
					{
						e[k] = td.outputs[td.cumoutsets[n]*td.nout+k] - pd.out[r+i];
						k++;
					}
					r += totnnodes;
				}
				backward(e, td, pd, n, true, g);
				for( int i = 0; i < nweights; i++ ) if( !setzero[i] ) grad[i] += g[i];
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(grad, nweights, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float gradient[nweights];
						MPI_Status status;
						MPI_Recv(gradient, nweights, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						for( int j = 0; j < nweights; j++ ) grad[j] += gradient[j];
					}
				}
				
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(grad, nweights, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
#endif
		
			return 0;
		}
		
		int Rforward(float *Cube, TrainingData &td, PredictedData &pd, int n, bool incerr, float *Rv)
		{
			incerr = false;
			
			float Rin[td.ntime[n]*totnnodes], Rout[td.ntime[n]*totnnodes];
			for( int i = 0; i < td.ntime[n]*totnnodes; i++ ) Rin[i] = Rout[i] = 0E0;
			
			/*calculate the R derivative*/
			
			int q  = -1, r = td.cuminsets[n]*totnnodes;
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				int p = td.cuminsets[n]*totnnodes + t*totnnodes;
				int nrw = nnweights; // starting index of recurrent weights
				if( recurrent && t > 0 ) nrw += totrnodes;
				int s = -1; // starting index of the vector
				
				for( int i = 0; i < nlayers; i++ )
				{
					for( int j = 0; j < nnodes[i]; j++ )
					{
						q++;
						
						if( i > 0 && !( i < nlayers-1 && j == nnodes[i]-1 ) ) // not a bias or input layer node
						{
							// contribution from non-recurrent connections
							for( int k = 0; k < nnodes[i-1]; k++ )
							{
								s++;
								Rin[q] += w[i-1][j][k] * Rout[p-r+k] + Cube[s] * pd.out[p+k];
							}
								
							// contribution from recurrent connections
							if( recurrent && i < nlayers-1 && j < nnodes[i]-1 )
							{
								if( t == 0 )
								{
									if( !norbias )
									{
										Rin[q] += Cube[nrw];
										nrw++;
									}
								}
								else
								{
									for( int k = 0; k < nnodes[i]-1; k++ )
									{
										Rin[q] += weights[nrw] * Rout[q-j-totnnodes+k] + Cube[nrw] * pd.out[r+q-j-totnnodes+k];
										nrw++;
									}
								}
							}
							
							//Rout[q] = linear[i] ? Rin[q] : Rin[q] * sigmoid(pd.in[r+q], 2);
							Rout[q] = activation(linear[i], pd.in[r+q], 2) * Rin[q];
						}
					}
					if( i > 0 ) p += nnodes[i-1];
				}
			}
			
			for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
			{
				int q = (t+1)*totnnodes-td.nout;
				for( int i = 0; i < td.nout; i++ )
					Rv[(t-td.ntimeignore[n])*td.nout+i] = Rout[q+i];
			}
			
			return 0;
		}
		
		int Rforward(float *Cube, TrainingData &td, PredictedData &pd, int n, bool incerr, float *Rv, float *Sv)
		{
			incerr = false;
			
			float Rin[td.ntime[n]*totnnodes], Rout[td.ntime[n]*totnnodes];
			for( int i = 0; i < td.ntime[n]*totnnodes; i++ ) Rin[i] = Rout[i] = 0E0;
			
			/*calculate the R derivative*/
			
			int q  = -1, r = td.cuminsets[n]*totnnodes;
			for( int t = 0; t < td.ntime[n]; t++ )
			{
				int p = td.cuminsets[n]*totnnodes + t*totnnodes;
				int nrw = nnweights; // starting index of recurrent weights
				if( recurrent && t > 0 ) nrw += totrnodes;
				int s = -1; // starting index of the vector
				
				for( int i = 0; i < nlayers; i++ )
				{
					for( int j = 0; j < nnodes[i]; j++ )
					{
						q++;
						
						if( i > 0 && !( i < nlayers-1 && j == nnodes[i]-1 ) ) // not a bias or input layer node
						{
							// contribution from non-recurrent connections
							for( int k = 0; k < nnodes[i-1]; k++ )
							{
								s++;
								Rin[q] += w[i-1][j][k] * Rout[p-r+k] + Cube[s] * pd.out[p+k];
							}
								
							// contribution from recurrent connections
							if( recurrent && i < nlayers-1 && j < nnodes[i]-1 )
							{
								if( t == 0 )
								{
									if( !norbias )
									{
										Rin[q] += Cube[nrw];
										nrw++;
									}
								}
								else
								{
									for( int k = 0; k < nnodes[i]-1; k++ )
									{
										Rin[q] += weights[nrw] * Rout[q-j-totnnodes+k] + Cube[nrw] * pd.out[r+q-j-totnnodes+k];
										nrw++;
									}
								}
							}
							
							//Rout[q] = linear[i] ? Rin[q] : Rin[q] * sigmoid(pd.in[r+q], 2);
							Rout[q] = activation(linear[i], pd.in[r+q], 2) * Rin[q];
						}
					}
					if( i > 0 ) p += nnodes[i-1];
				}
			}
			
			for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
			{
				int q = (t+1)*totnnodes-td.nout;
				for( int i = 0; i < td.nout; i++ )
					Rv[(t-td.ntimeignore[n])*td.nout+i] = Rout[q+i];
			}
			
			return 0;
		}
		
		int Av(float *v, TrainingData &td, PredictedData &pd, float *Av)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			
			float Av1[nweights];
			for( int i = 0; i < nweights; i++ )
			{
				Av[i] = Av1[i] = 0E0;
				if( setzero[i] ) v[i] = 0E0;
			}
			
			for( int n = nstart; n < nend; n++ )
			{
				float Rv[(td.ntime[n]-td.ntimeignore[n])*td.nout];
				for( int i = 0; i < (td.ntime[n]-td.ntimeignore[n])*td.nout; i++ ) Rv[i] = 0E0;
				
				// calculate R.v
				Rforward(v, td, pd, n, true, Rv);
				// calculate R^{t}.R.v
				backward(Rv, td, pd, n, true, Av1);
				for( int i = 0; i < nweights; i++ ) if( !setzero[i] ) Av[i] += Av1[i];
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(Av, nweights, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float Adotv[nweights];
						MPI_Status status;
						MPI_Recv(Adotv, nweights, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						for( int j = 0; j < nweights; j++ ) Av[j] += Adotv[j];
					}
				}
				
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(Av, nweights, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
#endif

			return 0;
		}
		
		int Av(float *v, TrainingData &td, PredictedData &pd, float alpha, float mu, float *Av)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			
			float Av1[nweights];
			for( int i = 0; i < nweights; i++ )
			{
				Av[i] = Av1[i] = 0E0;
				if( setzero[i] ) v[i] = 0E0;
			}
			
			for( int n = nstart; n < nend; n++ )
			{
				float Rv[(td.ntime[n]-td.ntimeignore[n])*td.nout];
				for( int i = 0; i < (td.ntime[n]-td.ntimeignore[n])*td.nout; i++ ) Rv[i] = 0E0;
				
				// calculate R.v
				Rforward(v, td, pd, n, true, Rv);
				// calculate R^{t}.R.v
				backward(Rv, td, pd, n, true, Av1);
				for( int i = 0; i < nweights; i++ ) if( !setzero[i] ) Av[i] += Av1[i];
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(Av, nweights, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float Adotv[nweights];
						MPI_Status status;
						MPI_Recv(Adotv, nweights, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						for( int j = 0; j < nweights; j++ ) Av[j] += Adotv[j];
					}
				}
				
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(Av, nweights, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
#endif

			return 0;
		}
		
		int correlations(TrainingData &td, PredictedData &pd)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			double sumxy[td.nout], sumx[td.nout], sumx2[td.nout], sumy[td.nout], sumy2[td.nout];
			
			for( int i = 0; i < td.nout; i++ ) pd.corr[i] = sumxy[i] = sumx[i] = sumx2[i] = sumy[i] = sumy2[i] = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				int p = td.cuminsets[n]*totnnodes;
				int l = td.cumoutsets[n]*td.nout;
				
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					int q = (t+1)*totnnodes-td.nout;
					
					for( int i = 0; i < td.nout; i++ )
					{
						sumx[i] += td.outputs[l];
						sumx2[i] += pow(td.outputs[l], 2.0);
						sumy[i] += pd.out[p+q+i];
						sumy2[i] += pow(pd.out[p+q+i], 2.0);
						sumxy[i] += td.outputs[l] * pd.out[p+q+i];
						
						l++;
					}
				}
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
		
				if( myid != 0 )
				{
					MPI_Send(sumx, td.nout, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
					MPI_Send(sumx2, td.nout, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
					MPI_Send(sumy, td.nout, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
					MPI_Send(sumy2, td.nout, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
					MPI_Send(sumxy, td.nout, MPI_DOUBLE, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						double sx[td.nout], sx2[td.nout], sy[td.nout], sy2[td.nout], sxy[td.nout];
						MPI_Status status;
						MPI_Recv(sx, td.nout, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
						MPI_Recv(sx2, td.nout, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
						MPI_Recv(sy, td.nout, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
						MPI_Recv(sy2, td.nout, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
						MPI_Recv(sxy, td.nout, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status);
	
						for( int j = 0; j < td.nout; j++ )
						{
							sumx[j] += sx[j];
							sumx2[j] += sx2[j];
							sumy[j] += sy[j];
							sumy2[j] += sy2[j];
							sumxy[j] += sxy[j];
						}
					}
				}
			}
#endif
			
			if( myid == 0 )
			{
				for( int i = 0; i < td.nout; i++ )
					pd.corr[i] = (float) ( td.cumoutsets[td.ndata] * sumxy[i] - sumx[i] * sumy[i] ) / ( sqrt( td.cumoutsets[td.ndata] * sumx2[i] - pow(sumx[i], 2.0) ) * sqrt( td.cumoutsets[td.ndata] * sumy2[i] - pow(sumy[i], 2.0) ) );
			}
			
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(pd.corr, td.nout, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
				
			return 0;
		}
		
		int ErrorSquared(TrainingData &td, PredictedData &pd)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			for( int i = 0; i < td.nout; i++ ) pd.errsqr[i] = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				int p = td.cuminsets[n]*totnnodes;
				int l = td.cumoutsets[n]*td.nout;
				
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					int q = (t+1)*totnnodes-td.nout;
					
					for( int i = 0; i < td.nout; i++ )
					{
						pd.errsqr[i] += pow(td.outputs[l] - pd.out[p+q+i], 2.0);
						
						l++;
					}
				}
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
		
				if( myid != 0 )
				{
					MPI_Send(pd.errsqr, td.nout, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float esqr[td.nout];
						MPI_Status status;
						MPI_Recv(esqr, td.nout, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
	
						for( int j = 0; j < td.nout; j++ ) pd.errsqr[j] += esqr[j];
					}
				}
			}
#endif
			
			if( myid == 0 )
			{
				for( int i = 0; i < td.nout; i++ )
					pd.errsqr[i] /= float(td.cumoutsets[td.ndata]);
			}
			
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(pd.errsqr, td.nout, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
				
			return 0;
		}
		
		int logZ(float *alpha, float logdetB, TrainingData &td, PredictedData &pd, float &logZ)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			logPrior(alpha, logZ);
			
			float logL = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				int q = td.cumoutsets[n]*td.nout;
				int r = (td.cuminsets[n]+td.ntimeignore[n])*totnnodes+totnnodes-td.nout;
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					for( int i = 0; i < td.nout; i++ )
					{
						logL += td.outputs[q] * log(pd.out[r+i]);
						q++;
					}
					r += totnnodes;
				}
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(&logL, 1, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float loglike;
						MPI_Status status;
						MPI_Recv(&loglike, 1, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						logL += loglike;
					}
				}
			}
#endif
			
			logZ += logL;
			
			logZ += -logdetB/2.0 - td.cumoutsets[td.ndata]*td.nout * log(2.0*M_PI) / 2.0;
			for( int i = 0; i < td.cumoutsets[td.ndata]*td.nout; i++ ) logZ += log(td.acc[i]) / 2.0;
			logZ += nweights * log(alpha[0]) / 2.0;
			
			/*if( nlayers > 2 )
			{
				float logG = -FLT_MAX;
				for( int i = 1; i < nlayers-1; i++ )
				{
					logG = LogSumExp(logG, lognfact(nnodes[i]) + nnodes[i] * log(2.0));
					if( recurrent )
					{
						int k = pow(float(nnodes[i]-1), 2);
						logG = LogSumExp(logG, lognfact(k) + k * log(2.0));
					}
				}
				logZ += logG;
			}*/
				
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Bcast(&logZ, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
		
			return 0;
		}
		
		int CorrectClass(TrainingData &td, PredictedData &pd)
		{
			int nstart, nend;
			getnstartend(td.ndata, nstart, nend);
			
			for( int j = 0; j < td.ncat; j++ ) pd.predrate[j] = 0E0;
			
			for( int n = nstart; n < nend; n++ )
			{
				int p = td.cuminsets[n]*totnnodes;
				int l = td.cumoutsets[n]*td.nout;
				
				for( int t = td.ntimeignore[n]; t < td.ntime[n]; t++ )
				{
					int q = (t+1)*totnnodes-td.nout;
					
					for( int i = 0; i < td.ncat; i++ )
					{
						float maxprob = pd.out[p+q];
						int k = 0;
						q++;
						
						for( int j = 1; j < td.nclasses[i]; j++ )
						{
							if( pd.out[p+q] > maxprob )
							{
								maxprob = pd.out[p+q];
								k = j;
							}
							q++;
						}
						pd.predrate[i] += (int) td.outputs[l+k];
						l += td.nclasses[i];
					}
				}
			}
			
#ifdef PARALLEL
			if( pflag )
			{
				MPI_Barrier(MPI_COMM_WORLD);
				
				if( myid != 0 )
				{
					MPI_Send(pd.predrate, pd.ncat, MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
				}
				else
				{
					for( int i = 1; i < ncpus; i++ )
					{
						float c[pd.ncat];
						MPI_Status status;
						MPI_Recv(c, pd.ncat, MPI_FLOAT, i, i, MPI_COMM_WORLD, &status);
						for( int j = 0; j < td.ncat; j++ ) pd.predrate[j] += c[j];
					}
				}
				
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Bcast(pd.predrate, pd.ncat, MPI_FLOAT, 0, MPI_COMM_WORLD);
			}
#endif
			for( int j = 0; j < td.ncat; j++ ) pd.predrate[j] /= float(td.cumoutsets[td.ndata]);
			
			return 0;
		}
};

class RBM : public NeuralNetwork
{
	public:
		size_t			ndata;		// total no. of data elements
		size_t			nepoch;		// no. of training epochs
		size_t			batchsize;	// no. of training examples in each batch
		size_t			nbatch;		// no. of batches
		float			epsilon_w;	// learning rate for weights
		float			epsilon_bv;	// learning rate for biases of input units
		float			epsilon_bh;	// learning rate for biases of hidden units
		float			cost_w;		// weight cost
		float			p_i;		// initial momentum
		float			p_f;		// final momentum
		float			*visbias;	// biases on the input units
		float		*batchposhidprob;	// hidden probabilities, to be used as input data for subsequent layers
	
		RBM()
		{
			size_t _nlayers = 0;
			size_t *_nnodes = NULL;
			Init(_nlayers, _nnodes, false, false);
			visbias = batchposhidprob = NULL;
			ndata = 0;
			
		}
		
		RBM(size_t *_nnodes, int _linear, size_t _ndata, int _nepoch)
		{
			size_t _nlayers = 2;
			bool _recurrent = false, _norbias = false;
			//bool _linear1[] = {true, _linear};
			int _linear1[] = {0, _linear};
			Init(_nlayers, _nnodes, _recurrent, _norbias, _linear1);
			ndata = _ndata;
			visbias = new float [nnodes[0]-1];
			
			// calculate the no. of batches & allocate batchposhidprob
			batchsize = 100;
			batchsize = ndata < batchsize ? 1 : batchsize;
			nbatch = ceil(float(ndata)/float(batchsize));
			batchposhidprob = new float [ndata*nnodes[1]];
			
			nepoch = _nepoch <= 0 ? 10 : _nepoch;
			
			if( linear[1]==0 )
				epsilon_w = epsilon_bv = epsilon_bh = 0.001;
			else
				epsilon_w = epsilon_bv = epsilon_bh = 0.1;

			cost_w = 0.0002;
			p_i = 0.5;
			p_f = 0.9;
		}
		
		RBM(size_t *_nnodes, int _linear, size_t _ndata, size_t _batchsize, size_t _nepoch, float _epsilon_w, float _epsilon_bv, float _epsilon_bh, float _cost_w, float _p_i, float _p_f)
		{
			size_t _nlayers = 2;
			bool _recurrent = false, _norbias = false;
			//bool _linear1[] = {true, _linear};
			int _linear1[] = {0, _linear};
			Init(_nlayers, _nnodes, _recurrent, _norbias, _linear1);
			ndata = _ndata;
			visbias = new float [nnodes[0]-1];
			
			batchsize = _batchsize;
			// calculate the no. of batches & allocate batchposhidprob
			batchsize = ndata < batchsize ? 1 : batchsize;
			nbatch = ceil(float(ndata)/float(batchsize));
			batchposhidprob = new float [ndata*nnodes[1]];
			
			nepoch = _nepoch;
			epsilon_w = _epsilon_w;
			epsilon_bv = _epsilon_bv;
			epsilon_bh = _epsilon_bh;
			cost_w = _cost_w;
			p_i = _p_i;
			p_f = _p_f;
		}

		~RBM()
		{
			if( nlayers > 0 ) delete [] batchposhidprob, visbias;
		}

		RBM(const RBM &rbm)
		{
			Clone(rbm);
			visbias = new float [nnodes[0]-1];
			for( size_t i = 0; i < nnodes[0]-1; i++ ) visbias[i] = rbm.visbias[i];
			
			ndata = rbm.ndata;
			batchsize = rbm.batchsize;
			nbatch = rbm.nbatch;
			nepoch = rbm.nepoch;
			epsilon_w = rbm.epsilon_w;
			epsilon_bv = rbm.epsilon_bv;
			epsilon_bh = rbm.epsilon_bh;
			cost_w = rbm.cost_w;
			p_i = rbm.p_i;
			p_f = rbm.p_f;
			
			batchposhidprob = new float [ndata*nnodes[1]];
			for( size_t i = 0; i < ndata*nnodes[1]; i++ ) batchposhidprob[i] = rbm.batchposhidprob[i];
		}
		
		void weightstoarrays(float **vishid, float *hidbias)
		{
			size_t k = 0;
			
			for( size_t i = 0; i < nnodes[1]; i++ )
			{
				for( size_t j = 0; j < nnodes[0]; j++ )
				{
					if( j != nnodes[0]-1 )
						vishid[j][i] = weights[k];
					else
						hidbias[i] = weights[k];
						
					k++;
				}
			}
		}
		
		void arraystoweights(float **vishid, float *hidbias)
		{
			size_t k = 0;
			
			for( size_t i = 0; i < nnodes[1]; i++ )
			{
				for( size_t j = 0; j < nnodes[0]; j++ )
				{
					if( j != nnodes[0]-1 )
						weights[k] = vishid[j][i];
					else
						weights[k] = hidbias[i];
						
					k++;
				}
			}
			
			arrangeweights();
		}
		
		// set weights drawn from a normal distribution with means of in-hid, hid-bias, in-bias being 
		// mean[0], mean[1], mean[2] respectively and the standard deviations being sigma[0], sigma[1],
		// sigma[2] respectively.
		int setnormalweights(float *mean, float *sigma, long seed)
		{
			size_t nhid = nnodes[1];
			size_t nin = nnodes[0]-1;
			
			float **vishid, *hidbias;
			hidbias = new float [nhid];
			vishid = new float* [nin];
			for( size_t i = 0; i < nin; i++ ) vishid[i] = new float [nhid];
			
			// set the in-hid weights
			for( size_t i = 0; i < nin; i++ )
				for( size_t j = 0; j < nhid; j++ )
					vishid[i][j] = sigma[0] > 0E0 ? mean[0] + gasdev(&seed)*sigma[0] : mean[0];
			
			// set the hid-bias
			for( size_t i = 0; i < nhid; i++ ) hidbias[i] = sigma[1] > 0E0 ? mean[1] + gasdev(&seed)*sigma[1] : mean[1];
			
			// set the in-bias
			for( size_t i = 0; i < nin; i++ ) visbias[i] = sigma[2] > 0E0 ? mean[2] + gasdev(&seed)*sigma[2] : mean[2];
			
			arraystoweights(vishid, hidbias);
			
			for( size_t i = 0; i < nin; i++ ) vishid[i];
			delete [] vishid, hidbias;
			
			return 0;
		}

		void pretrain(float *trainingdata, long seed)
		{
		
			size_t nhid = nnodes[1];
			size_t nin = nnodes[0]-1;
			
			std::cout << "Pretraining Layer with RBM: " << nin << "-" << nhid << "\n";
			
			float *vishid = (float *) malloc(nin * nhid * sizeof(float));
			float *vishidinc = (float *) malloc(nin * nhid * sizeof(float));
			float *posprod = (float *) malloc(nin * nhid * sizeof(float));
			float *negprod = (float *) malloc(nin * nhid * sizeof(float));

			float *hidbias = (float *) malloc(nhid * sizeof(float));
			float *visbiasM = (float *) malloc(nin * sizeof(float));
			float *hidbiasinc = (float *) malloc(nhid * sizeof(float));
			float *visbiasinc = (float *) malloc(nin * sizeof(float));
			float *poshidact = (float *) malloc(nhid * sizeof(float));
			float *neghidact = (float *) malloc(nhid * sizeof(float));
			float *posvisact = (float *) malloc(nin * sizeof(float));
			float *negvisact = (float *) malloc(nin * sizeof(float));

			// copy the weights, hidden & visible biases to their respective matrix & vectors
			{
				float **w = new float* [nin]; for( size_t i = 0; i < nin; i++ ) w[i] = new float [nhid];
				float *b = new float [nhid];
				weightstoarrays(w, b);
				
				for (size_t i = 0; i < nin; i++)
				{
					visbiasM[i] = visbias[i];
					for (size_t j = 0; j < nhid; j++)
					{
						vishid[i * nhid + j] = w[i][j];
						hidbias[j] = b[j];
					}
				}
				
				for( size_t i = 0; i < nin; i++ ) delete [] w[i];
				delete[] w, b;
			}
			
			
			// start of training
			for( size_t epoch = 0; epoch < nepoch; epoch++ )
			{
				std::cout << "epoch " << epoch+1 << "\n";
				
				float errsum = 0E0;
				
				for( size_t batch = 0; batch < nbatch; batch++ )
				{
					size_t bsize = batch == nbatch-1 ? ndata-(nbatch-1)*batchsize : batchsize;
					
					
					float *negdata = (float *) malloc(bsize * nin * sizeof(float));
					float *poshidprob = (float *) malloc(bsize * nhid * sizeof(float));
					float *neghidprob = (float *) malloc(bsize * nhid * sizeof(float));
					float *poshidstates = (float *) malloc(bsize * nhid * sizeof(float));

					//std::cout << "epoch " << epoch+1 << " batch " << batch+1 << "\r";
					
					/**************** get the data for this batch ****************/
					
					float *data = (float *) malloc(bsize * nin * sizeof(float));
					for (size_t i=0; i<bsize*nin; i++)
						data[i] = trainingdata[batch * batchsize * nin + i];
					
					/**************** start of the positive phase ****************/
					
					for (size_t i=0; i<bsize; i++)
						for (size_t j=0; j<nhid; j++)
							poshidprob[i * nhid + j] = hidbias[j];
					cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,bsize,nhid,nin,1.0,data,nin,vishid,nhid,1.0,poshidprob,nhid);
					
					for( size_t i = 0; i < bsize; i++ )
					{
						for( size_t j = 0; j < nhid; j++ )
						{
							poshidprob[i * nhid + j] = activation(linear[1], poshidprob[i * nhid + j], 3);
							batchposhidprob[batch*batchsize*nhid+i*nhid+j] = poshidprob[i * nhid + j];
						}
					}
					
					cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nin,nhid,bsize,1.0,data,nin,poshidprob,nhid,0.0,posprod,nhid);
					
					for( size_t i = 0; i < nhid; i++ )
					{
						float a = 0E0;
						for( size_t j = 0; j < bsize; j++ ) a += poshidprob[j * nhid + i];
						poshidact[i] = a;
					}
					
					for( size_t i = 0; i < nin; i++ )
					{
						float a = 0E0;
						for( size_t j = 0; j < bsize; j++ ) a += data[j * nin + i];
						posvisact[i] = a;
					}
					
					/**************** end of positive phase ****************/
					
					
					
					for( size_t i = 0; i < bsize; i++ )
						for( size_t j = 0; j < nhid; j++ )
							if( linear[1]==act_linear )
								poshidstates[i * nhid + j] = poshidprob[i * nhid + j] + gasdev(&seed);
							else if( linear[1]==act_relu || linear[1]==act_softplus )
								poshidstates[i * nhid + j] = activation(linear[1], poshidprob[i * nhid + j] + gasdev(&seed) * sqrt(sigmoid(poshidprob[i * nhid + j],1)), 1);
							else if( linear[1]==act_sigmoid )
								poshidstates[i * nhid + j] = poshidprob[i * nhid + j] > ran2(&seed) ? 1E0 : 0E0;
							else
								poshidstates[i * nhid + j] = poshidprob[i * nhid + j] > ran2(&seed) ? 1E0 : -1E0;
					
					
					/**************** start of negative phase ****************/
					
					for (size_t i=0; i<bsize; i++)
						for (size_t j=0; j<nin; j++)
							negdata[i * nin + j] = visbiasM[j];
					cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,bsize,nin,nhid,1.0,poshidstates,nhid,vishid,nhid,1.0,negdata,nin);
					
					for( size_t i = 0; i < bsize*nin; i++ )
						negdata[i] = activation(linear[1],negdata[i],1);
					
					for (size_t i=0; i<bsize; i++)
						for (size_t j=0; j<nhid; j++)
							neghidprob[i * nhid + j] = hidbias[j];
					cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,bsize,nhid,nin,1.0,negdata,nin,vishid,nhid,1.0,neghidprob,nhid);
					
					for (size_t i=0; i<bsize*nhid; i++)
						neghidprob[i] = activation(linear[1],neghidprob[i],3);
					
					cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,nin,nhid,bsize,1.0,negdata,nin,neghidprob,nhid,0.0,negprod,nhid);
					
					for( size_t i = 0; i < nhid; i++ )
					{
						float a = 0E0;
						for( size_t j = 0; j < bsize; j++ ) a += neghidprob[j * nhid + i];
						neghidact[i] = a;
					}
					
					for( size_t i = 0; i < nin; i++ )
					{
						float a = 0E0;
						for( size_t j = 0; j < bsize; j++ ) a += negdata[j * nin + i];
						negvisact[i] = a;
					}
					
					/**************** end of negative phase ****************/
					
					
					float err = 0E0;
					for (size_t i=0; i<bsize*nin; i++)
						err += pow(data[i] - negdata[i],2.0);
					errsum += err;
					
					float p = epoch > 5 ? p_f : p_i;
					
					
					
					/**************** update weights & biases ****************/
					
					// updating weights
					
					for (size_t i=0; i<nin*nhid; i++)
					{
						float tmp;
						tmp = (posprod[i] - negprod[i]) * epsilon_w / float(bsize);
						vishidinc[i] = vishidinc[i] * p + tmp;
						tmp = vishid[i] * epsilon_w * cost_w;
						vishidinc[i] -= tmp;
						vishid[i] += vishidinc[i];
					}
					
					// updating visible biases
					
					for (size_t i=0; i<nin; i++)
					{
						float tmp;
						tmp = (posvisact[i] - negvisact[i]) * epsilon_bv / float(bsize);
						visbiasinc[i] = visbiasinc[i] * p + tmp;
						visbiasM[i] += visbiasinc[i];
					}
					
					// updating hidden biases
					
					for (size_t i=0; i<nhid; i++)
					{
						float tmp;
						tmp = (poshidact[i] - neghidact[i]) * epsilon_bh / float(bsize);
						hidbiasinc[i] = hidbiasinc[i] * p + tmp;
						hidbias[i] += hidbiasinc[i];
					}
					
					free(negdata);
					free(poshidprob);
					free(neghidprob);
					free(poshidstates);
					free(data);
				}
				std::cout << "epoch " << epoch+1 << " error " << errsum << "\n";
			}
			
			// copy the learned weights, hidden & visible back to class members
			{
				float **w = new float* [nin]; for( size_t i = 0; i < nin; i++ ) w[i] = new float [nhid];
				float *b = new float [nhid];
				
				for (size_t i = 0; i < nin; i++)
				{
					visbias[i] = visbiasM[i];
					for (size_t j = 0; j < nhid; j++)
					{
						w[i][j] = vishid[i * nhid + j];
						b[j] = hidbias[j];
					}
				}
				
				arraystoweights(w, b);
				
				for( size_t i = 0; i < nin; i++ ) delete [] w[i];
				delete[] w, b;
			}
			
			free(vishid);
			free(vishidinc);
			free(posprod);
			free(negprod);

			free(hidbias);
			free(visbiasM);
			free(hidbiasinc);
			free(visbiasinc);
			free(poshidact);
			free(neghidact);
			free(posvisact);
			free(negvisact);
		}
		
		int correlations(TrainingData &td, PredictedData &pd)
		{
			return 0;
		}
		
		int CorrectClass(TrainingData &td, PredictedData &pd)
		{
			return 0;
		}
		
		int ErrorSquared(TrainingData &td, PredictedData &pd)
		{
			return 0;
		}
		
		int logZ(float *alpha, float logdetB, TrainingData &td, PredictedData &pd, float &logZ)
		{
			return 0;
		}
		
		int forward(TrainingData &td, PredictedData &pd, int n, bool incerr)
		{
			return 0;
		}
		
		int forwardOne(int ntime, float *in, float *out)
		{
			return 0;
		}
		
		int logLike(TrainingData &td, PredictedData &pd, float &logL, bool doforward)
		{
			return 0;
		}
		
		int logLike(TrainingData &td, PredictedData &pd, float &logL)
		{
			return 0;
		}
		
		int logC(TrainingData &td, float &logC)
		{
			return 0;
		}
		
		int backward(float *u, TrainingData &td, PredictedData &pd, int n, bool incerr, float *grad)
		{
			return 0;
		}
		
		int backward(float *ul, float *us, TrainingData &td, PredictedData &pd, int n, bool incerr, float *gradl, float *grads)
		{
			return 0;
		}
		
		int grad(TrainingData &td, PredictedData &pd, float *grad)
		{
			return 0;
		}
		
		int Rforward(float *Cube, TrainingData &td, PredictedData &pd, int n, bool incerr, float *Rv)
		{
			return 0;
		}
		
		int Rforward(float *Cube, TrainingData &td, PredictedData &pd, int n, bool incerr, float *Rv, float *Sv)
		{
			return 0;
		}
		
		int Av(float *v, TrainingData &td, PredictedData &pd, float *Av)
		{
			return 0;
		}
		
		int Av(float *v, TrainingData &td, PredictedData &pd, float alpha, float mu, float *Av)
		{
			return 0;
		}
		
		float HHPscore(TrainingData &td, PredictedData &pd, bool doforward)
		{
			return 0E0;
		}
		
		float HHPscore(TrainingData &td, PredictedData &pd)
		{
			return 0E0;
		}
};



#endif
