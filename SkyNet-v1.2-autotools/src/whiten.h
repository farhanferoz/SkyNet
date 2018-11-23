#ifndef __WHITEN_H__
#define __WHITEN_H__ 1

#include <numeric>
#include <limits>
#include <vector>
#include <iomanip>
#include <map>
#include <algorithm>
#include <stdlib.h>

// this file defines three transforms: 

// axisAlignedTransform : translates and rescales each axis independantly 
// histEqualTransform: Farhan's histogram equaliser
// histEqualTransform2: Charlies's histogram equaliser

// NB: neither of the last two rescale accuracies properly

// after whitening the covarience matrix will have WSIGMA^2
// along its diagonal (probably)
#define WSIGMA 1.0

static inline std::string crudeTime()
{
#ifdef WIN32
	return "FIXME";
#else
	time_t ffs = time(0);
	return ctime(&ffs);
#endif
};

class whiteTransform 
{
	public:
		
		void read(const std::string &fn)
		{
			std::ifstream fin(fn.c_str());
			read(fin);
		}

		void write(const std::string &fn)
		{
			std::ofstream fout(fn.c_str());
			write(fout);
		}
		
		virtual int noutput() const = 0;
		virtual void clear(size_t) = 0;
		virtual void read(std::istream &os) = 0;
		virtual void write(std::ostream &os) const = 0;
		virtual void findFromData(float *data, size_t ndata, size_t nout, float) = 0;
		virtual void apply(const float *in, float *out, size_t n) const = 0;
		virtual void inverse(const float *in, float *out, size_t n) const = 0;
		virtual void applyScalingOnly(const float* positions, const float* in, float* out, size_t n) const = 0;
		virtual void inverseScalingOnly(const float* positions, const float* in, float* out, size_t n) const = 0;
		virtual void inversePredictedOutputs(const float *in, float *out, size_t nout, size_t ndata,
											 size_t totnnodes, size_t *cuminsets, size_t *cumoutsets,
											 size_t *ntimeignore, size_t *ntime) const = 0;
};

static inline std::ostream &operator << (std::ostream &os, const whiteTransform &a)
{
	a.write(os);
	return os;
}

class axisAlignedTransform : public whiteTransform
{
	public:
		int type;			// type = 1 => set mean = 0 & stdev = 0.5
							// type = 2 => set min = 0 & max = 1
							// type = 3 => set min = -1 & max = 1
		bool indep;			// are all input/output values independent, if they are then a separate scale/off set is calculated for each
		std::vector<float> scale;
		std::vector<float> off;

		axisAlignedTransform()
		{
			type = 1;
			indep = true;
		}
		
		~axisAlignedTransform()
		{
			scale.clear();
			off.clear();
		}

		axisAlignedTransform(size_t _type, bool _indep)
		{
			type = _type;
			indep = _indep;
		}

		axisAlignedTransform(size_t _type, bool _indep, size_t size)
		{
			type = _type;
			indep = _indep;
			clear(size);
		}

		void clear(size_t nout)
		{
			off = std::vector<float>(nout, 0E0);
			scale = std::vector<float>(nout, 1E0);
		}

		int noutput() const { return off.size(); };
			
		void read(std::istream &os)
		{
			size_t size = 0;
			os >> size;
			clear(size);
			
			//for(size_t i = 0; i < scale.size(); i++) os >> scale[i];
			//for(size_t i = 0; i < off.size(); i++) os >> off[i];
			for(size_t i = 0; i < scale.size(); i++)
			{
				os >> scale[i];
				os >> off[i];
			}
		}
		
		void write(std::ostream &os) const 
		{
			os << off.size() << "\n";
			std::ios_base::fmtflags originalFormat = os.flags();
			os << std::scientific << std::setprecision(10);
			
			for(size_t j = 0; j < scale.size(); j++)
			{
				os << scale[j] << " " << off[j] << "\n";
			}		
			os.flags(originalFormat);
		}
		
		void findFromData(float *data, size_t ndata, size_t nout, float)
		{
			if( type == 1 )
			{
				off = std::vector<float>(nout, 0E0);
				scale = std::vector<float>(nout, 0E0);
			}
			else
			{
				off = std::vector<float>(nout, FLT_MAX);
				scale = std::vector<float>(nout, -FLT_MAX);
			}

			int nsamp = 0;
			for (size_t i=0; i<ndata; i++)
			{
				nsamp = indep ? nsamp+1 : nsamp+nout;
				for (int j=0; j<nout; j++)
				{
					float d = data[i*nout+j];
					size_t k = indep ? j : 0;
					
					if( type == 1 )
					{
						off[k] += d;
						scale[k] += d*d;
					}
					else
					{
						if( d < off[k] ) off[k] = d;
						if( d > scale[k] ) scale[k] = d;
					}
				}
			}

			for(int j=0; j<(indep?nout:1); j++)
			{
				if( type == 1 )
				{
					off[j] /= nsamp; 
					scale[j] = sqrt((scale[j] - off[j]*off[j]*nsamp)/(nsamp - 1))/WSIGMA;
				}
				else if( type ==2 )
				{
					scale[j] -= off[j];
				}
				else
				{
					float tmpmax,tmpmin;
					tmpmax = scale[j];
					tmpmin = off[j];
					scale[j] = (tmpmax-tmpmin)/2E0;
					off[j] = (tmpmax+tmpmin)/2E0;
				}
				
				//if( !finite(scale[j]) || scale[j] <= 0E0 ) scale[j] = 1E0;
				//if (!finite(scale[j])) scale[j] = 0E0;
				if( fabs(scale[j]) > std::numeric_limits<float>::max() || scale[j] <= 0E0 ) scale[j] = 1E0;
				if( fabs(scale[j]) > std::numeric_limits<float>::max() ) scale[j] = 0E0;
				
				//assert(finite(scale[j]));
				assert(fabs(scale[j]) <= std::numeric_limits<float>::max());
				assert(scale[j] >= 0E0);
			}
			
			if( !indep )
			{
				for (int j=1; j<nout; j++)
				{
					off[j] = off[0];
					scale[j] = scale[0];
				}
			}
		}

		// apply transform found above 		
		void apply(const float *in, float *out, size_t n) const
		{
			//out.resize(in.size());
			assert(off.size());
			assert(scale.size() == off.size());
			
			for (size_t i=0, p=0; i<n/off.size(); i++)
			for (size_t j=0; j<off.size(); j++, p++) 
			{
				if (scale[j] != 0E0)
					out[p] = float(in[p])/scale[j] - off[j]/scale[j];
				else
					out[p] = off[j];
			}
		}

		// remove transform found above
		void inverse(const float *in, float *out, size_t n) const
		{
			//out.resize(in.size());
			assert(off.size());
			assert(scale.size() == off.size());

			for (size_t i=0, p=0; i<n/off.size(); i++)
			for (size_t j=0; j<off.size(); j++, p++) out[p] = in[p]*scale[j] + off[j];
		}

		// newdata = iscale *(data - mu)

		// sig -> sig/scale
		// (1/sig) -> (1/sig) * scale
		// acc -> acc * scale

		// apply scale part of transform found above
		void applyScalingOnly(const float*, const float* in, float* out, size_t n) const
		{
			//out.resize(in.size());

			for (size_t i=0, p=0; i<n/off.size(); i++)
			{
				for (size_t j=0; j<off.size(); j++, p++) 
				{
					if (scale[j] != 0E0)
						out[p] = in[p] * scale[j];
					else 
						out[p] = in[p];
				}
			}
		}

		// inverse scale part of transform found above
		void inverseScalingOnly(const float*, const float* in, float* out, size_t n) const
		{
			//out.resize(in.size());

			for (size_t i=0, p=0; i<n/off.size(); i++)
			{
				for (size_t j=0; j<off.size(); j++, p++) 
				{
					if (scale[j] != 0E0)
						out[p] = in[p] / scale[j];
					else 
						out[p] = in[p];
				}
			}
		}
		
		// inverse transform on predicted data outputs
		void inversePredictedOutputs(const float *in, float *out, size_t nout, size_t ndata,
									 size_t totnnodes, size_t *cuminsets, size_t *cumoutsets,
									 size_t *ntimeignore, size_t *ntime) const
		{
			for( int n = 0; n < ndata; n++ )
			{
				int p = cuminsets[n]*totnnodes;
				int l = cumoutsets[n]*nout;
				
				for( int t = ntimeignore[n]; t < ntime[n]; t++ )
				{
					int q = (t+1)*totnnodes-nout;
					
					for( int i = 0; i < nout; i++ )
					{
						out[p+q+i] = in[p+q+i]*scale[i] + off[i];
						l++;
					}
				}
			}
		}
};

static inline std::ostream &operator << (std::ostream &os, const axisAlignedTransform &a)
{
	os << "[ ";
	for(size_t i = 0; i < a.off.size(); i++)
		os << "(" << a.off[i] << ", " << a.scale[i] << ") ";
	os << "]";
	return os;
}

class histEqualTransform : public whiteTransform
{
	public:
		int nout, ndata;
		float **hist_x, **hist_y;

		histEqualTransform() {};

		~histEqualTransform()
		{
			for(size_t i = 0; i < nout; i++ ) delete [] hist_x[i], hist_y[i];
			delete [] hist_x, hist_y;
		}

		histEqualTransform(size_t nOut, size_t nData)
		{
			nout = nOut;
			ndata = nData;
			clear(nout);
		}

		void clear(size_t nout)
		{
			hist_x = new float*[nout];
			hist_y = new float*[nout];
			
			for(size_t i = 0; i < nout; i++ )
			{
				hist_x[i] = new float[ndata];
				hist_y[i] = new float[ndata];
			}
		}

		int noutput() const { return nout; };
			
		void read(std::istream &os)
		{
			os >> nout >> ndata;
			clear(nout);
			
			for(int i = 0; i < nout; i++)
			{
				for(int j = 0; j < ndata; j++)
				{
					os >> hist_x[i][j] >> hist_y[i][j];
				}
			}
		}
		
		void write(std::ostream &os) const 
		{
			os << nout << "\t" << ndata << "\n";
			std::ios_base::fmtflags originalFormat = os.flags();
			os << std::scientific << std::setprecision(10);
			
			for(int i = 0; i < nout; i++)
			{
				for(int j = 0; j < ndata; j++)
				{
					os << hist_x[i][j] << "\t" << hist_y[i][j] << "\n";
				}
			}
			os.flags(originalFormat);
		}
		
		void findFromData(float *data, size_t _ndata, size_t nOut, float)
		{
			nout = nOut;
			ndata = _ndata;
			clear(nout);
			
			for(int i = 0; i < nout; i++)
			{
				// collect & sort the data values
				for(int j = 0; j < ndata; j++)
				{
					int k;
					for(k = 0; k < j; k++)
					{
						if( data[i + nout * j] <= hist_x[i][k] ) break;
					}
					
					for(int m = j; m > k; m--)
					{
						hist_x[i][m] = hist_x[i][m - 1];
					}
					
					hist_x[i][k] = data[i + nout * j];
				}
				
				// calculate the cumulative probabilities
				for(int j = 0; j < ndata; j++)
				{
					hist_y[i][j] = (float) j / ( (float) ( ndata - 1 ) ) - 0.5E0;
					
					for(int k = j - 1; k >= 0; k--)
					{
						if( hist_x[i][k] == hist_x[i][j] ) hist_y[i][k] = hist_y[i][j];
					}
				}
			}
		}

		// apply transform found above 
		void apply(const float *in, float *out, size_t n) const
		{
			//out.resize(in.size());
			
			for( int i = 0; i < nout; i++ )
			{
				for(size_t j = 0; j < n / nout; j++ )
				{
					for( int k = 0; k < ndata; k++ )
					{
						if( in[i + nout * j] < hist_x[i][k] )
						{
							if( k - 1 >= 0 )
								out[i + nout * j] = hist_y[i][k - 1];
							else
								out[i + nout * j] = hist_y[i][0];
							break;
						}
						else if( in[i + nout * j] == hist_x[i][k] || k == ndata - 1 )
						{
							out[i + nout * j] = hist_y[i][k];
							break;
						}
					}
				}
			}
		}

		// remove transform found above
		void inverse(const float *in, float *out, size_t n) const
		{
			//out.resize(in.size());
			
			for( int i = 0; i < nout; i++ )
			{
				for(size_t j = 0; j < n / nout; j++ )
				{
					for( int k = 0; k < ndata; k++ )
					{
						if( in[i + nout * j] < hist_y[i][k] )
						{
							if( k - 1 >= 0 )
								out[i + nout * j] = hist_x[i][k - 1];
							else
								out[i + nout * j] = hist_x[i][0];
							break;
						}
						else if( in[i + nout * j] == hist_y[i][k] || k == ndata - 1 )
						{
							out[i + nout * j] = hist_x[i][k];
							break;
						}
					}
				}
			}
		}
		
		void applyScalingOnly(const float*, const float* in, float* out, size_t n) const
		{
			std::cerr << "DEBUG FIXME: returning bogus result\n";
			//out.resize(in.size());
			for( size_t i = 0; i < n; i++ ) out[i] = 1E0;
			//std::fill(out.begin(), out.end(), 1.0f);
		}
		
		void applinverseScalingOnly(const float*, const float* in, float* out, size_t n) const
		{
			std::cerr << "DEBUG FIXME: returning bogus result\n";
			//out.resize(in.size());
			for( size_t i = 0; i < n; i++ ) out[i] = 1.0;
			//std::fill(out.begin(), out.end(), 1.0f);
		}
		
		void inverseScalingOnly(const float*, const float* in, float* out, size_t n) const
		{
			std::cerr << "DEBUG FIXME: returning bogus result\n";
			//out.resize(in.size());
			for( size_t i = 0; i < n; i++ ) out[i] = 1.0;
			//std::fill(out.begin(), out.end(), 1.0f);
		}
		
		void inversePredictedOutputs(const float *in, float *out, size_t nout, size_t ndata,
									 size_t totnnodes, size_t *cuminsets, size_t *cumoutsets,
									 size_t *ntimeignore, size_t *ntime) const
		{
		}
};

class histEqualTransform2 : public whiteTransform
{
	public:
		std::vector<std::multimap<float, float> > forward;
		std::vector<std::multimap<float, float> > backward;
		
		~histEqualTransform2()
		{
			forward.clear();
			backward.clear();
		}

		void clear(size_t nout)
		{
			forward.clear();
			backward.clear();
			forward.resize(nout);
			backward.resize(nout);
		}

		int noutput() const { return forward.size(); };
		int ninput() const { return forward.size(); };
			
		void read(std::istream &os)
		{
			abort();
		}
		
		void write(std::ostream &os) const 
		{
			os << forward.size() << "\n";
			for(size_t i = 0; i < forward.size(); i++)
			{
				const std::multimap<float, float> &c = forward[i];
				os << c.size() << " ";
				for(std::multimap<float, float>::const_iterator s = c.begin(); 
					s != c.end(); s++)
					os << (*s).first << " " << (*s).second << " ";
				os << "\n";
			}
		}
		
		void findFromData(float *data, size_t ndata, size_t nOut, float)
		{
			clear(nOut);
			
			for(int i = 0; i < nOut; i++)
			{
				std::vector<float> d;
				for(size_t j = 0; j < ndata; j++ )
					d.push_back(data[j*nOut + i]);
				std::sort(d.begin(), d.end());
				
				size_t np = std::min(size_t(10), d.size());
				for(size_t j = 0; j < np; j++)
				{
					size_t p = j*d.size()/np + d.size()/(2*np);
					forward[i].insert(std::make_pair(d[p], float(p)/float(d.size()) - 0.5f)); 
					backward[i].insert(std::make_pair(float(p)/float(d.size()) - 0.5f, d[p])); 
				}
			}
		}
		
		float mapit(const std::multimap<float, float> &cum, const float &in, bool reverse) const
		{
			if (!cum.size()) return INFINITY;
			
			// Finds the first element whose key is > in. 			
			std::multimap<float, float>::const_iterator f = cum.upper_bound(in);

			if (f == cum.begin())
			{
				// underflow
				float x0 = (*f).first;
				float y0 = (*f).second;

				assert(in <= x0);

				++f;
				if (f == cum.end()) return -1.0;

				float x1 = (*f).first;
				float y1 = (*f).second;

				if (!reverse)
				{
					if (x0 == x1) return -1.0;
						
					float lam = ((y1-y0)/(x1-x0)) / (0.5 + y0);
					
					return -0.5 + (0.5+y0)*exp(lam*(in - x0));
				}
				else
				{
					if (y0 == y1) return -1.0;
						
					float lam = ((x1-x0)/(y1-y0)) / (0.5 + x0);
					
					// in = -0.5 + (0.5+x0)*exp(lam*(out - y0));
					// log((in + 0.5)/(0.5+x0)) = lam*(out - y0)
					// log((in + 0.5)/(0.5+x0))/lam + y0 = out
					
					return log((in + 0.5)/(0.5+x0))/lam + y0;
				}				
			}
			else if (f == cum.end())
			{
				// overflow
				std::multimap<float, float>::const_reverse_iterator f = cum.rbegin();
				
				float x0 = (*f).first;
				float y0 = (*f).second;
				
				assert(x0 <= in);
				
				++f;
				if (f == cum.rend()) return 1.0;

				float x1 = (*f).first;
				float y1 = (*f).second;
				
				if (!reverse)
				{
					if (x0 == x1) return 1.0;
					
					float lam = ((y0-y1)/(x0-x1)) / (0.5 - y0);
					
					return 0.5 - (0.5-y0)*exp(lam*(x0 - in));
				}
				else
				{
					if (y0 == y1) return 1.0;
					
					float lam = ((x0-x1)/(y0-y1)) / (0.5 - x0);
					
					return y0 - log((0.5-in)/(0.5-x0))/lam;
				}
			}
			else
			{
				std::multimap<float, float>::const_iterator p = f;
				p--;
				if ((*p).first != (*f).first)
				{
					assert((*f).first > in);
					assert((*p).first <= in);
					assert((*f).first > (*p).first);
					assert((*f).second > (*p).second);
					
					// linear map 
					return (*p).second + 
						((*f).second - (*p).second) * (in - (*p).first) / ((*f).first - (*p).first);
				}
				else
					return ((*f).second + (*p).second) / 2.0;				
			}
		}

		// apply transform found above 
		void apply(const float* in, float* out, size_t n) const
		{
			//out.resize(in.size());
			size_t nout = forward.size();
			size_t ndata = n/nout;
			
			for(size_t i = 0; i < nout; i++ )
			{
				const std::multimap<float, float> &cum = forward[i];
				for(size_t j = 0; j < ndata; j++)
					out[nout*j + i] = mapit(cum, in[nout*j + i], false);
			}
		}

		// remove transform found above
		void inverse(const float *in, float *out, size_t n) const
		{
			//out.resize(in.size());
			size_t nout = forward.size();
			size_t ndata = n/nout;
			
			for(size_t i = 0; i < nout; i++ )
			{
				const std::multimap<float, float> &cum = backward[i];
				for(size_t j = 0; j < ndata; j++)
					out[nout*j + i] = mapit(cum, in[nout*j + i], true);
			}
		}
		
		void applyScalingOnly(const float*, const float* in, float* out, size_t n) const
		{
			std::cerr << "DEBUG FIXME: returning bogus result\n";
			//out.resize(in.size());
			for( size_t i = 0; i < n; i++ ) out[i] = 1.0;
			//std::fill(out.begin(), out.end(), 1.0f);
		}
		
		void inverseScalingOnly(const float*, const float* in, float* out, size_t n) const
		{
			std::cerr << "DEBUG FIXME: returning bogus result\n";
			//out.resize(in.size());
			for( size_t i = 0; i < n; i++ ) out[i] = 1.0;
			//std::fill(out.begin(), out.end(), 1.0f);
		}
		
		void inversePredictedOutputs(const float *in, float *out, size_t nout, size_t ndata,
									 size_t totnnodes, size_t *cuminsets, size_t *cumoutsets,
									 size_t *ntimeignore, size_t *ntime) const
		{
		}
};


#endif
