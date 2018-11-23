#ifndef __PREDICTEDDATA_H__
#define __PREDICTEDDATA_H__ 1

class PredictedData
{
	public:
		size_t					ndata;		// I	Number of data lines
		size_t					nin;		// I	Number of input nodes = width of input table
		size_t					nout;      	// I	Number of output nodes = width of output table
		size_t					ninsets;	// I	Number of input sets
		size_t					noutsets;	// I	Number of output sets
		size_t 					totnnodes;	// O	total no. of non-recurrent nodes
		size_t 					totrnodes;	// O	total no. of recurrent nodes
		
		// information that affects classnet likelihood calculations
		size_t					ncat;		// I no. of categories
		
		
		float* 					in;		// O	node inputs for the weights in nn
		float* 					out;		// O	node outputs for the weights in nn
		float* 					y;		// O	node outputs for the weights in nn, includes the error contribution
		float* 					Chi;		// O	(y-t)/sigma
		float* 					corr;		// O	correlation (error rate) for regression (classification) outputs
		float* 					errsqr;		// O	sum_i (y_i - t_i)^2
		float* 					predrate;	// O	fraction of correct predictions, relevant for classification nets only

		PredictedData()
		{
			ndata = nin = nout = ninsets = noutsets = totnnodes = totrnodes = ncat = 0;
		}
		
		~PredictedData()
		{
			clear();
		}

		PredictedData(size_t _totnnodes, size_t _totrnodes, size_t _ndata, size_t _nin, size_t _nout, size_t _ninsets, size_t _noutsets)
		{
			size_t _ncat = 0;
			resize(_totnnodes, _totrnodes, _ndata, _nin, _nout, _ninsets, _noutsets, _ncat);
		}

		PredictedData(size_t _totnnodes, size_t _totrnodes, size_t _ndata, size_t _nin, size_t _nout, size_t _ninsets, size_t _noutsets, size_t _ncat)
		{
			resize(_totnnodes, _totrnodes, _ndata, _nin, _nout, _ninsets, _noutsets, _ncat);
		}

		void resize(size_t _totnnodes, size_t _totrnodes, size_t _ndata, size_t _nin, size_t _nout, size_t _ninsets, size_t _noutsets, size_t _ncat)
		{
			totnnodes = _totnnodes;
			totrnodes = _totrnodes;
			ndata = _ndata;
			nin = _nin;
			nout = _nout;
			ninsets = _ninsets;
			noutsets = _noutsets;
			ncat = _ncat;
			
			in = new float [ninsets*totnnodes+ndata*totrnodes];
			out = new float [ninsets*totnnodes+ndata*totrnodes];
			y = new float [ninsets*totnnodes+ndata*totrnodes];
			Chi = new float [noutsets*nout];
			corr = new float [nout];
			errsqr = new float [nout];
			predrate = new float [ncat];
		}
		
		void clear()
		{
			delete [] in, out, y, Chi, corr, errsqr, predrate;
		}

		PredictedData &operator = (const PredictedData &pd)
		{
			Clone(pd);
			return *this;
		}

		void Clone(const PredictedData &pd)
		{	
			resize(pd.totnnodes, pd.totrnodes, pd.ndata, pd.nin, pd.nout, pd.ninsets, pd.noutsets, pd.ncat);
			for(size_t i = 0; i < ninsets*totnnodes+ndata*totrnodes; i++)
			{
				in[i] = pd.in[i];
				out[i] = pd.out[i];
				y[i] = pd.y[i];
			}
			for(size_t i = 0; i < noutsets*nout; i++) Chi[i] = pd.Chi[i];
			for(size_t i = 0; i < nout; i++)
			{
				corr[i] = pd.corr[i];
				errsqr[i] = pd.errsqr[i];
			}
			for(size_t i = 0; i < ncat; i++) predrate[i] = pd.predrate[i];
		}	
};



#endif
