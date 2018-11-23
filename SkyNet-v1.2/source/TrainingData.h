#ifndef __TRININGDATA_H__
#define __TRININGDATA_H__ 1

#ifdef PARALLEL
#include <mpi.h>
#endif

class TrainingData
{
	public:
		int			myid;		// MPI processor ID
		int			ncpus;		// MPI total no. of processors
		size_t			ndata;		// I no. of data lines = height of both tables
		size_t			nin;		// I no. of input nodes = width of input table
		size_t			nout;      	// I no. of output nodes = width of output table
		bool			autoencoder;	// if true then outputs (cumoutsets) points to inputs (cuminsets)

		float*			outputs; 	// I training data targets (also called data)
		float*			inputs;  	// I training data inputs (also called x)
		size_t*			ntime;		// I no. of time elements
		size_t*			ntimeignore;	// I no. of output time elements to ignore
		size_t*			cuminsets;	// I cumulative no. of input sets
		size_t*			cumoutsets;	// I cumulative no. of out sets
		
		// input groupings
		size_t			nincat;		// I no. of categories
		size_t*			ninclasses;	// I no. of classes in each categories
		
		// information that affects classnet likelihood calculations
		size_t			ncat;		// I no. of categories
		size_t*			nclasses;	// I no. of classes in each categories

		// information that affects likelihood calculations
		float*			acc;     	// I

		// Gaussian Likelihood: acc[i] = 1/sigma[i]
		// Classification Likelihood: acc[i] = empty
		// Classification Prize Likelihood: acc[i] = prize earned for selecting that class
		// Tophat Likelihood: acc[i] = 1/d[i] where the insensitive region is data +/- d
		
//		std::vector<int> train;		// I mask vector 1=target is defined, 0 = fill in target

		TrainingData()
		{
			nincat = ncat = ndata = nin = nout = 0;
			autoencoder = false;
			resize(0, 0, 0, 0, 0, 0, 0);
		}

		~TrainingData()
		{
			clear();
		}

		TrainingData(bool classnet, int _nin, int _nout, int _ndata, std::vector <float> &_inputs, std::vector <float> &_outputs, std::vector <size_t> &_ntime, std::vector <size_t> &_ntimeignore)
		{
			bool _autoencoder = false;
			Init(classnet, _nin, _nout, _ndata, _inputs, _outputs, _ntime, _ntimeignore, _autoencoder);
		}

		TrainingData(bool classnet, int _nin, int _nout, int _ndata, std::vector <float> &_inputs, std::vector <float> &_outputs, std::vector <float> &_acc, std::vector <size_t> &_ntime, std::vector <size_t> &_ntimeignore)
		{
			bool _autoencoder = false;
			Init(classnet, _nin, _nout, _ndata, _inputs, _outputs, _acc, _ntime, _ntimeignore, _autoencoder);
		}

		void Init(bool classnet, int _nin, int _nout, int _ndata, std::vector <float> &_inputs, std::vector <float> &_outputs, std::vector <size_t> &_ntime, std::vector <size_t> &_ntimeignore, bool _autoencoder)
		{
			autoencoder = _autoencoder;
			nincat = ncat = ndata = nin = nout = 0;
			std::vector <size_t> _cuminsets, _cumoutsets, _nclasses, _ninclasses;
			if( classnet ) _nclasses.push_back(_nout);
			_ninclasses.push_back(_nin);
			size_t nins = 0, nouts = 0;
			
			for( int i = 0; i < _ndata; i++ )
			{
				_cuminsets.push_back(nins);
				nins += _ntime[i];
				_cumoutsets.push_back(nouts);
				nouts += _ntime[i] - _ntimeignore[i];
			}
			_cuminsets.push_back(nins);
			_cumoutsets.push_back(nouts);
			//std::cerr << "Read " << _ndata << " data elements.\n";
			
			size_t insize, outsize, classsize, inclasssize;
			insize = _inputs.size();
			outsize = _outputs.size();
			classsize = _nclasses.size();
			inclasssize = _ninclasses.size();

			resize(_nin, _nout, _ndata, insize, outsize, classsize, inclasssize);
			
			for( size_t i = 0; i < insize; i++ ) inputs[i] = _inputs[i];
			if( !autoencoder )
			{
				for( size_t i = 0; i < outsize; i++ ) outputs[i] = _outputs[i];
			}
			for( size_t i = 0; i < ndata; i++ )
			{
				ntime[i] = _ntime[i];
				ntimeignore[i] = _ntimeignore[i];
			}
			for( size_t i = 0; i < ndata+1; i++ )
			{
				cuminsets[i] = _cuminsets[i];
				if( !autoencoder ) cumoutsets[i] = _cumoutsets[i];
			}
			for( size_t i = 0; i < classsize; i++ ) nclasses[i] = _nclasses[i];
			for( size_t i = 0; i < inclasssize; i++ ) ninclasses[i] = _ninclasses[i];
		}

		void Init(bool classnet, int _nin, int _nout, int _ndata, std::vector <float> &_inputs, std::vector <float> &_outputs, std::vector <float> &_acc, std::vector <size_t> &_ntime, std::vector <size_t> &_ntimeignore, bool _autoencoder)
		{
			Init(classnet, _nin, _nout, _ndata, _inputs, _outputs, _ntime, _ntimeignore, _autoencoder);
			if( !classnet )
			{
				for( size_t i = 0; i < _outputs.size(); i++ ) acc[i] = _acc[i];
			}
		}

		TrainingData(std::string filename, std::map<char, int> &charmap)
		{
			autoencoder = false;
			GenTrainingDataFromText(filename, charmap);
		}

		TrainingData(std::string infile, std::map<char, int> &charmap, std::string mapfile)
		{
			autoencoder = false;
			GenTextMap(infile, charmap, mapfile);
			GenTrainingDataFromText(infile, charmap);
		}

		TrainingData(std::string filename, bool classnet, bool readacc)
		{
			bool _autoencoder = false;
			int _nin = 0, _nout = 0;
			GenTrainingData(filename, classnet, _nin, _nout, _autoencoder, readacc);
		}

		TrainingData(std::string filename, bool classnet, bool _autoencoder, bool readacc)
		{
			int _nin = 0, _nout = 0;
			GenTrainingData(filename, classnet, _nin, _nout, _autoencoder, readacc);
		}

		TrainingData(std::string filename, bool classnet, int _nin, int _nout, bool readacc)
		{
			bool _autoencoder = false;
			GenTrainingData(filename, classnet, _nin, _nout, _autoencoder, readacc);
		}

		TrainingData(std::string filename, bool classnet, int _nin, int _nout, bool _autoencoder, bool readacc)
		{
			GenTrainingData(filename, classnet, _nin, _nout, _autoencoder, readacc);
		}

		void GenTrainingData(std::string filename, bool classnet, int _nin, int _nout, bool _autoencoder, bool readacc)
		{
			myid = 0;
			ncpus = 1;
#ifdef PARALLEL
			MPI_Comm_rank(MPI_COMM_WORLD,&myid);
			MPI_Comm_size(MPI_COMM_WORLD, &ncpus);
#endif
			
			autoencoder = _autoencoder;
			nincat = ncat = ndata = nin = nout = 0;
			size_t _ndata = 0;
			std::vector <float> _inputs, _outputs, _acc;
			std::vector <size_t> _ntime, _ntimeignore, _cuminsets, _cumoutsets, _nclasses, _ninclasses;
			size_t nins = 0, nouts = 0;
			
			if( myid == 0 )
			{
				std::ifstream fin(filename.c_str());
				if( fin.fail() ) throw std::runtime_error("Can not open the file.\n");
				std::string line, item;
				
				// get the input specification
				getline(fin, line);
				std::istringstream linestream(line);
				if( _nin == 0 )
				{
					for(;;)
					{
						if( !getline(linestream, item, ',') )
						{
							for( size_t i = 0; i < _ninclasses.size(); i++ ) _nin += _ninclasses[i];
							break;
						}
						_ninclasses.push_back(atoi(item.c_str()));
					}
				}
				
				// get the output specification
				getline(fin, line);
				linestream.clear();
				linestream.str(line);
				if( classnet )
				{
					if( _nout == 0 )
					{
						for(;;)
						{
							if( !getline(linestream, item, ',') )
							{
								for( size_t i = 0; i < _nclasses.size(); i++ ) _nout += _nclasses[i];
								break;
							}
							_nclasses.push_back(atoi(item.c_str()));
						}
					}
					else
					{
						_nclasses.push_back(_nout);
					}
				}
				else
				{
					getline(linestream, item, ',');
					if( _nout == 0 ) _nout = atoi(item.c_str());
					if( autoencoder ) _nout = _nin;
				}
				
				while(getline(fin, line))
				{
					if( line == "" ) continue;
					size_t j = 0;
					linestream.clear();
					linestream.str(line);
					for( ;; )
					{
						if( !getline(linestream, item, ',') )
						{
							if( j%_nin != 0 ) throw std::runtime_error("no. of inputs not a multiple of nin.\n");
							_ntime.push_back(j/_nin);
							_cuminsets.push_back(nins);
							nins += j/_nin;
							break;
						}
						j++;
						_inputs.push_back(atof(item.c_str()));
					}
					
					if( !getline(fin, line) ) throw std::runtime_error("no output corresponding to an input set");
					if( autoencoder )
					{
						_ntimeignore.push_back(0);
						_ndata++;
					}
					else
					{
						linestream.clear();
						linestream.str(line);
						j = 0;
						size_t nc = 0;
						for( ;; )
						{
							if( !getline(linestream, item, ',') )
							{
								if( j == 0 || j%_nout != 0 ) throw std::runtime_error("no. of outputs not a multiple of nout.\n");
								_ntimeignore.push_back(_ntime.back() - j/_nout);
								_ndata++;
								_cumoutsets.push_back(nouts);
								nouts += j/_nout;
								nc = 0;
								break;
							}
							
							if( classnet )
							{
								int ival = atoi(item.c_str());
								if( ival < 0 || ival >= _nclasses[nc] ) throw std::runtime_error("invalid class member.\n");
								for( size_t i = 0; i < _nclasses[nc]; i++ )
								{
									j++;
									if( i == ival )
										_outputs.push_back(1.0);
									else
										_outputs.push_back(0.0);
								}
								nc++;
								if( nc >= _nclasses.size() ) nc = 0;
							}
							else
							{
								j++;
								_outputs.push_back(atof(item.c_str()));
							}
						}
					}
					
					// read the inverse standard deviations
					if( readacc && !classnet )
					{
						if( !getline(fin, line) ) throw std::runtime_error("no acc corresponding to an input set");
						linestream.clear();
						linestream.str(line);
						j = 0;
						for( ;; )
						{
							if( !getline(linestream, item, ',') )
							{
								if( j == 0 || j%_nout != 0 ) throw std::runtime_error("no. of acc not a multiple of nout.\n");
								break;
							}
							
							j++;
							_acc.push_back(atof(item.c_str()));
						}
					}
				}
				fin.close();
				_cuminsets.push_back(nins);
				if( !autoencoder ) _cumoutsets.push_back(nouts);
				//std::cerr << "Read " << _ndata << " data elements.\n";
			}
			
			size_t insize, outsize, classsize, inclasssize;
			if( myid == 0 )
			{
				insize = _inputs.size();
				outsize = autoencoder ? insize : _outputs.size();
				classsize = _nclasses.size();
				inclasssize = _ninclasses.size();
			}
			
			int mpiint;
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
			
			mpiint = (int) _nin;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			_nin = (size_t) mpiint;
			
			mpiint = (int) _nout;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			_nout = (size_t) mpiint;
			
			mpiint = (int) _ndata;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			_ndata = (size_t) mpiint;
			
			mpiint = (int) insize;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			insize = (size_t) mpiint;
			
			mpiint = (int) outsize;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			outsize = (size_t) mpiint;
			
			mpiint = (int) classsize;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			classsize = (size_t) mpiint;
			
			mpiint = (int) inclasssize;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			inclasssize = (size_t) mpiint;
			
			MPI_Barrier(MPI_COMM_WORLD);
#endif

			resize(_nin, _nout, _ndata, insize, outsize, classsize, inclasssize);
			
			for( size_t i = 0; i < insize; i++ )
			{
				if( myid == 0 ) inputs[i] = _inputs[i];
#ifdef PARALLEL
				MPI_Bcast(&inputs[i], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
			}
			if( !autoencoder )
			{
				for( size_t i = 0; i < outsize; i++ )
				{
					if( myid == 0 ) outputs[i] = _outputs[i];
#ifdef PARALLEL
					MPI_Bcast(&outputs[i], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
				}
			}
			if( readacc && !classnet )
			{
				for( size_t i = 0; i < outsize; i++ )
				{
					if( myid == 0 ) acc[i] = _acc[i];
#ifdef PARALLEL
					MPI_Bcast(&acc[i], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
				}
			}
			for( size_t i = 0; i < _ndata; i++ )
			{
				if( myid == 0 )
				{
					ntime[i] = _ntime[i];
					ntimeignore[i] = _ntimeignore[i];
				}
#ifdef PARALLEL
				mpiint = (int) ntime[i];
				MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
				ntime[i] = (size_t) mpiint;
				
				mpiint = (int) ntimeignore[i];
				MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
				ntimeignore[i] = (size_t) mpiint;
#endif
			}
			for( size_t i = 0; i < _ndata+1; i++ )
			{
				if( myid == 0 )
				{
					cuminsets[i] = _cuminsets[i];
					if( !autoencoder ) cumoutsets[i] = _cumoutsets[i];
				}
#ifdef PARALLEL
				mpiint = (int) cuminsets[i];
				MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
				cuminsets[i] = (size_t) mpiint;
				
				if( !autoencoder )
				{
					mpiint = (int) cumoutsets[i];
					MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
					cumoutsets[i] = (size_t) mpiint;
				}
#endif
			}
			for( size_t i = 0; i < classsize; i++ )
			{
				if( myid == 0 ) nclasses[i] = _nclasses[i];
#ifdef PARALLEL
				mpiint = (int) nclasses[i];
				MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
				nclasses[i] = (size_t) mpiint;
#endif
			}
			for( size_t i = 0; i < inclasssize; i++ )
			{
				if( myid == 0 ) ninclasses[i] = _ninclasses[i];
#ifdef PARALLEL
				mpiint = (int) ninclasses[i];
				MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
				ninclasses[i] = (size_t) mpiint;
#endif
			}
		}

		void GenTextMap(std::string filename, std::map<char, int> &charmap)
		{
			charmap.clear();
			std::ifstream fin(filename.c_str());
			if( fin.fail() ) throw std::runtime_error("Can not open the file.\n");
			std::string line, item;
			getline(fin, line);
				
			char c;
			for(;;)
			{
				if( !fin.get(c) ) break;
				if( c == '\n' ) continue;
				if( charmap[c] == 0 ) charmap[c] = charmap.size();
			}
			fin.close();
		}

		void GenTextMap(std::string infile, std::map<char, int> &charmap, std::string mapfile)
		{
			myid = 0;
#ifdef PARALLEL
			MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#endif
			if( myid == 0 )
			{
				GenTextMap(infile, charmap);
				
				std::ofstream fout(mapfile.c_str());
				if( fout.fail() ) throw std::runtime_error("Can not open the map file.\n");
				std::map<char, int>::iterator it;
				for( it = charmap.begin(); it != charmap.end(); it++ )
					fout << (*it).first << "\t" << (*it).second << "\n";
				fout.close();
			}
		}

		void ReadTextMap(std::string mapfile, std::map<char, int> &charmap)
		{
			myid = 0;
#ifdef PARALLEL
			MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#endif
			if( myid == 0 )
			{
				std::ifstream fin(mapfile.c_str());
				if( fin.fail() ) throw std::runtime_error("Can not open the map file.\n");
				char c;
				int i;
				do
				{
					std::string line, item;
					getline(fin, line);
					std::istringstream linestream(line);
					getline(linestream, item, '\t');
					if( item.size() == 0 ) break;
					c = item[0];
					getline(linestream, item, '\n');
					i = atoi(item.c_str());
					charmap[c] = i;
				}
				while( !fin.eof() );
				fin.close();
			}
		}

		void GenTrainingDataFromText(std::string filename, std::map<char, int> &charmap)
		{
			myid = 0;
			ncpus = 1;
#ifdef PARALLEL
			MPI_Comm_rank(MPI_COMM_WORLD,&myid);
			MPI_Comm_size(MPI_COMM_WORLD, &ncpus);
#endif
			
			nincat = ncat = ndata = nin = nout = 0;
			size_t _ndata = 0;
			int _nin = 0, _nout = 0;
			std::vector <char> inseq;
			std::vector <size_t> nseq;
			std::vector <float> _inputs, _outputs;
			std::vector <size_t> _ntime, _ntimeignore, _cuminsets, _cumoutsets, _nclasses;
			size_t nins = 0, nouts = 0;
			int minseqlen, nignore;
			int quotecount = 0;
			
			if( myid == 0 )
			{
				std::ifstream fin(filename.c_str());
				if( fin.fail() ) throw std::runtime_error("Can not open the file.\n");
				std::string line, item;
				getline(fin, line);
				std::istringstream linestream(line);
				getline(linestream, item, ','); minseqlen = atoi(item.c_str());
				getline(linestream, item, ','); nignore = atoi(item.c_str());
				
				char c;
				std::vector <char> seq;
				for(;;)
				{
					if( !fin.get(c) ) break;
					if( seq.size() == 0 && ( c == ' ' || c == '\n' ) )
					{
						continue;
					}
					else if( seq.size() >= minseqlen-1 && ( ( c == '.' && quotecount == 0 ) || ( c == '"' && seq.back() == '.' && quotecount == 1 ) ) )
					{
						seq.push_back(c);
						for( size_t i = 0; i < seq.size(); i++ ) inseq.push_back(seq[i]);
						nseq.push_back(seq.size());
						seq.clear();
						quotecount = 0;
					}
					else if( c == '\n' )
					{
						seq.clear();
						quotecount = 0;
					}
					else
					{
						if( charmap.count(c) > 0 )
						{
							seq.push_back(c);
							if( c == '"' )
							{
								quotecount++;
								if( quotecount == 2 ) quotecount == 0;
							}
						}
						else
						{
							seq.clear();
							quotecount = 0;
						}
					}
				}
				fin.close();
				
				_nin = _nout = charmap.size();
				_ndata = nseq.size();
				_nclasses.push_back(_nout);
				
				int m = -1;
				for( size_t i = 0; i < nseq.size(); i++ )
				{
					int ntin = (nseq[i]-1);
					int ntout = (nseq[i]-1-nignore);
					_ntime.push_back(ntin);
					_ntimeignore.push_back(nignore);
					_cuminsets.push_back(nins);
					_cumoutsets.push_back(nouts);
					nins += ntin;
					nouts += ntout;
					
					for( size_t j = 0; j < nseq[i]; j++ )
					{
						m++;
						c = inseq[m];
						int cl = charmap.find(c)->second;
						
						if( j < nseq[i]-1 )
						{
							for( size_t k = 1; k <= _nin; k++ )
							{
								if( k == cl )
									_inputs.push_back(1.0);
								else
									_inputs.push_back(0.0);
							}
						}
						
						if( j > nignore )
						{
							for( size_t k = 1; k <= _nout; k++ )
							{
								if( k == cl )
									_outputs.push_back(1.0);
								else
									_outputs.push_back(0.0);
							}
						}
					}
				}
				_cuminsets.push_back(nins);
				_cumoutsets.push_back(nouts);
				//std::cerr << "Read " << _ndata << " data elements.\n";
			}
			
			size_t insize, outsize, classsize;
			size_t inclasssize = 1;
			if( myid == 0 )
			{
				insize = _inputs.size();
				outsize = _outputs.size();
				classsize = _nclasses.size();
			}
			
			int mpiint;
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
			
			mpiint = (int) _nin;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			_nin = (size_t) mpiint;
			
			mpiint = (int) _nout;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			_nout = (size_t) mpiint;
			
			mpiint = (int) _ndata;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			_ndata = (size_t) mpiint;
			
			mpiint = (int) insize;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			insize = (size_t) mpiint;
			
			mpiint = (int) outsize;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			outsize = (size_t) mpiint;
			
			mpiint = (int) classsize;
			MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
			classsize = (size_t) mpiint;
			
			MPI_Barrier(MPI_COMM_WORLD);
#endif

			resize(_nin, _nout, _ndata, insize, outsize, classsize, inclasssize);
			
			for( size_t i = 0; i < insize; i++ )
			{
				if( myid == 0 ) inputs[i] = _inputs[i];
#ifdef PARALLEL
				MPI_Bcast(&inputs[i], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
			}
			for( size_t i = 0; i < outsize; i++ )
			{
				if( myid == 0 ) outputs[i] = _outputs[i];
#ifdef PARALLEL
				MPI_Bcast(&outputs[i], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
			}
			for( size_t i = 0; i < _ndata; i++ )
			{
				if( myid == 0 )
				{
					ntime[i] = _ntime[i];
					ntimeignore[i] = _ntimeignore[i];
				}
#ifdef PARALLEL
				mpiint = (int) ntime[i];
				MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
				ntime[i] = (size_t) mpiint;
				
				mpiint = (int) ntimeignore[i];
				MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
				ntimeignore[i] = (size_t) mpiint;
#endif
			}
			for( size_t i = 0; i < _ndata+1; i++ )
			{
				if( myid == 0 )
				{
					cuminsets[i] = _cuminsets[i];
					cumoutsets[i] = _cumoutsets[i];
				}
#ifdef PARALLEL
				mpiint = (int) cuminsets[i];
				MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
				cuminsets[i] = (size_t) mpiint;
				
				mpiint = (int) cumoutsets[i];
				MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
				cumoutsets[i] = (size_t) mpiint;
#endif
			}
			for( size_t i = 0; i < classsize; i++ )
			{
				if( myid == 0 ) nclasses[i] = _nclasses[i];
#ifdef PARALLEL
				mpiint = (int) nclasses[i];
				MPI_Bcast(&mpiint, 1, MPI_INT, 0, MPI_COMM_WORLD);
				nclasses[i] = (size_t) mpiint;
#endif
			}
			ninclasses[0] = nin;
		}

		void resize(size_t _nin, size_t _nout, size_t _ndata, size_t insize, size_t outsize, size_t _ncat, size_t _nincat)
		{
			clear();
			
			ndata = _ndata;
			nin = _nin;
			nout = _nout;
			ncat = _ncat;
			nincat = _nincat;
			
			if( insize > 0 ) inputs = new float [insize];
			if( outsize > 0 )
			{
				outputs = autoencoder ? inputs : new float [outsize];
				acc = new float [outsize];
			}
			if( ndata > 0 )
			{
				ntime = new size_t [ndata];
				ntimeignore = new size_t [ndata];
				cuminsets = new size_t [ndata+1];
				cumoutsets = autoencoder ? cuminsets : new size_t [ndata+1];
			}
			if( ncat > 0 ) nclasses = new size_t [ncat];
			if( nincat > 0 ) ninclasses = new size_t [nincat];
		}

		void clear()
		{
			if( nin > 0 ) delete [] inputs;
			if( nout > 0 )
			{
				delete [] acc;
				if( !autoencoder ) delete [] outputs;
			}
			if( ndata > 0 )
			{
				delete [] ntime, ntimeignore, cuminsets;
				if( !autoencoder ) delete [] cumoutsets;
			}
			if( ncat > 0 ) delete [] nclasses;
			if( nincat > 0 ) delete [] ninclasses;
			
			nincat = ncat = ndata = nin = nout = 0;
		}

		TrainingData &operator = (const TrainingData &td)
		{
			Clone(td);
			return *this;
		}

		void Clone(const TrainingData &td)
		{
			autoencoder = td.autoencoder;
			size_t insize = td.cuminsets[td.ndata]*td.nin;
			size_t outsize = td.cumoutsets[td.ndata]*td.nout;
			resize(td.nin, td.nout, td.ndata, insize, outsize, td.ncat, td.nincat);
			for( size_t i = 0; i < insize; i++ ) inputs[i] = td.inputs[i];
			for( size_t i = 0; i < outsize; i++ )
			{
				if( !autoencoder ) outputs[i] = td.outputs[i];
				acc[i] = td.acc[i];
			}
			for( size_t i = 0; i < ndata; i++ )
			{
				ntime[i] = td.ntime[i];
				ntimeignore[i] = td.ntimeignore[i];
			}
			for( size_t i = 0; i < ndata+1; i++ )
			{
				cuminsets[i] = td.cuminsets[i];
				if( !autoencoder ) cumoutsets[i] = td.cumoutsets[i];
			}
			for( size_t i = 0; i < ncat; i++ ) nclasses[i] = td.nclasses[i];
			for( size_t i = 0; i < nincat; i++ ) ninclasses[i] = td.ninclasses[i];
		}

		void dump(std::ostream &os)
		{
			/*os << "ndata = " << ndata << "\n";
			os << "ntime = " << ntime << "\n";
			os << "outputs.size = " << ndata*ntime*nout << "\n";
			os << "nin = " << nin << "\n";
			os << "nout = " << nout << "\n";
			os << "acc = " << acc << "\n";
			os << "outputs = " << outputs << "\n";
			os << "inputs = " << inputs << "\n";
			os << "train = " << train << "\n";*/
		}
		
		// generate a new data set by whitening both input and output, also  return the two affineTransforms that were used
		void generateWhiteData(whiteTransform *itrans, whiteTransform *otrans, TrainingData &out, float slimit = -1) const
		{
			//std::cerr << "itrans=" << *itrans << "\n";
			//std::cerr << "otrans=" << *otrans << "\n";
			
			size_t insize = cuminsets[ndata]*nin;
			size_t outsize = cumoutsets[ndata]*nout;
			
			out.Clone(*this);
			
			itrans->findFromData(inputs, cuminsets[ndata], nin, slimit);
			if( !autoencoder )
				otrans->findFromData(outputs, cumoutsets[ndata], nout, slimit);
			else
				otrans = itrans;

			//std::cerr << "itrans=" << *itrans << "\n";
			//std::cerr << "otrans=" << *otrans << "\n";

			//std::cerr << "RESIZE " << itrans->noutput() << " " <<  otrans->noutput() << "\n";

			itrans->apply(inputs, out.inputs, insize);
			if( !autoencoder ) otrans->apply(outputs, out.outputs, outsize);
			otrans->applyScalingOnly(inputs, acc, out.acc, outsize);
		}

		// generate a new data set by whitening only the input (if flag=1) or output (if flag = 2), also  return the two affineTransforms that were used
		void generateWhiteData(whiteTransform *trans, TrainingData &out, int flag, float slimit = -1)
		{
			if( flag == 1 )
			{
				trans->findFromData(inputs, cuminsets[ndata], nin, slimit);
			
				size_t insize = cuminsets[ndata]*nin;

				trans->apply(inputs, out.inputs, insize);
			}
			else if( flag == 2 )
			{
				trans->findFromData(outputs, cumoutsets[ndata], nout, slimit);
			
				size_t outsize = cumoutsets[ndata]*nout;
			
				trans->apply(outputs, out.outputs, outsize);
				trans->applyScalingOnly(inputs, acc, out.acc, outsize);
			}
		}		
};



#endif
