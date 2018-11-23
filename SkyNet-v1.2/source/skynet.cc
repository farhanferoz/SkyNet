#include "NNopt.h"
#include "RBMpretrain.h"

double TrainNetwork(char *inputfile, char *inroot, char *outroot, size_t *nlayerspass, size_t *nnodespass, bool resume, bool printerror) {
	int myid = 0, ncpus = 1;
#ifdef PARALLEL
 	MPI_Comm_size(MPI_COMM_WORLD,&ncpus);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#endif
	
	//size_t nhid=1;				// nodes in the hidden layer
	std::vector <size_t> nhid;
	int i,j,n;
	float tol=1.e-3;			// tolerance for convergence
	float rate=0.1,ratemin=0.01;
	float frac=1.;
	bool doprint = true;
	bool printseparate = true;
	int stopcounter=100;
	//int stopcounter=500;
	//int bouncecheck[] = {1, 10};		// bouncecheck[1] = no. of past iterations to check the bouncing behaviour on
	int bouncecheck[] = {1, 15};		// bouncecheck[1] = no. of past iterations to check the bouncing behaviour on
						// bouncecheck[0] = no. of diverging iterations out of last bouncecheck[1] to signify convergence
	bool evidence=false;
	int printfreq=1;
	float aim=1.0;
	int maxniter=1000000;			// max no. of iterations allowed
	int nin = 0, nout = 0;
	
	bool classnet = false;			// whether it's a classification net or not
	//bool resume = true;			// Should be left to true so resuming is done when possible
	bool resuming = false;			// Don't touch!
	bool resetalpha = false;
	bool resetsigma = false;
	bool histmaxent = false;
	bool recurrent = true;
	int whitenin = 1;
	int whitenout = 1;
	int stopf = 4;				// stopping criterion based on, 1 = validation correlation, 2 = validation logL, 3 = validation logP, 4 = validation error squared
	int hhpl = 1;				// 1 = regular logL, 2 = hhps logL
	bool hhps = false;
	bool text = false;			// textual inputs & outputs?
	
	// Options you should set as you want
	bool prior = true;			// Use prior or not
	bool noise = false;			// Use noise re-scaling or not
	bool wnoise = true;			// If the noise level is set on whitened data (false sets on unwhitened)
	bool norbias = false;			// start with a blank state for RNN?
	float sigma = 1.0;			// the noise level on the whitened or unwhitened data, depending on wnoise
	int useSD = 0;				// use structural damping
	float mu = 0.0;				// structural damping coefficient
	bool vdata=true;			// use validation data
	bool autoencoder=false;			// train an autoencoder?
	bool pretrain=false;			// pretrain network?
	int nepoch=0;
	bool indep=true;			// are input/output values in the data independent?
	float loglrange;			// dummy variable, not used
	float randweights=0.0;			// add a random offset to the saved weights when resuming
	int lnsrch = 0;				// perform line search for optimal distance to move in new direction (default=1.0)
	
	bool fixseed = false;
	int fixedseed = 123;
	int verbose=3;
	bool readacc=false;
	
	char linearlayers[100];
	linearlayers[0]='\0';
	float stdev = 0.1;
	ReadInputFile2(inputfile,nhid,&classnet,&frac,&prior,&noise,&wnoise,&sigma,&rate,&printfreq,
		      &fixseed,&fixedseed,&evidence,&histmaxent,&recurrent,&whitenin,&whitenout,&stopf,&hhps,&hhpl,&maxniter,
		      &nin,&nout,&norbias,&useSD,&text,&vdata,linearlayers,&resetalpha,&resetsigma,&autoencoder,&pretrain,&nepoch,&indep,
		      &ratemin,&loglrange,&randweights,&verbose,&readacc,&stdev,&lnsrch);
	
	if (classnet) readacc=false;
	if (readacc) noise=wnoise=false;
	if (recurrent) pretrain=false;
	//if (pretrain) whitenin=whitenout=2;
	if (text) recurrent=classnet=true;
	if (useSD==1) mu=1E-4;
	if (recurrent) stopcounter=500;
	if (histmaxent) noise=false;		// cannot do noise scaling with historical maximum entropy, only classic
	if (classnet) whitenout=0;
	if (whitenout!=1 && whitenout!=2 && whitenout!=3) wnoise=false;		// cannot set noise on whitened outputs if not whitening the outputs!
	if (hhpl==2 || hhpl==3) hhps=true;
	if (classnet) hhps=false;
	if (classnet) autoencoder=false;
	if (!hhps) hhpl=1;
	if (autoencoder) whitenout=whitenin;
	bool doforward = whitenin != 0 || whitenout != 0 ? true : false;
	int whiteninORIG=whitenin;
	int whitenoutORIG=whitenout;
	
	// setup filenames
	std::string root     = inroot;
	std::string root2    = outroot;
	std::string datafile = concatenate(root,"train.txt");
	std::string testfile = concatenate(root,"test.txt");
	std::string dataout  = concatenate(root2,"train_pred.txt");
	std::string testout  = concatenate(root2,"test_pred.txt");
	std::string dumpfile = concatenate(root2,"network.txt");
	std::string simpfile = concatenate(root2,"networksimple.txt");
	std::string AEencoderfile = concatenate(root2,"encoder.txt");
	std::string AEdecoderfile = concatenate(root2,"decoder.txt");
	std::string mapfile  = concatenate(root2,"charmap.txt");
	
	// test for resume file
	FILE *res;
	if (resume && (res=fopen(dumpfile.c_str(),"r"))!=NULL) {
		fclose(res);
		resuming=true;
	}
	
	// initialize the training data
	TrainingData td,vd;
	if(text)
	{
		std::map <char, int> charmap;
		td = TrainingData(datafile,charmap,mapfile);
		if( vdata ) vd = TrainingData(testfile,charmap);
	}
	else
	{
		td = TrainingData(datafile,classnet,nin,nout,autoencoder,readacc);
		if( vdata ) vd = TrainingData(testfile,classnet,nin,nout,autoencoder,readacc);
	}
	
	if( vd.ndata==0 ) vdata=false;

	if(hhpl==2 || hhpl==3) {
		for( size_t i = 0; i < td.cumoutsets[td.ndata]*td.nout; i++ ) td.outputs[i] = log(td.outputs[i]+1.0);
		if( vdata ) for( size_t i = 0; i < vd.cumoutsets[vd.ndata]*vd.nout; i++ ) vd.outputs[i] = log(vd.outputs[i]+1.0);
		if(hhpl==3) hhpl=1;
	}
	
	if (td.cuminsets[td.ndata]==td.ndata) recurrent=false;
	
	// create acc vectors and fill with dummy value
	if (!classnet && !readacc) {
		for( size_t i = 0; i < td.cumoutsets[td.ndata]*td.nout; i++ ) td.acc[i] = 1.0/sigma;
		if( vdata ) for( size_t i = 0; i < vd.cumoutsets[vd.ndata]*vd.nout; i++ ) vd.acc[i] = 1.0/sigma;
	}
	
	// whiten the data and set the noise
	//trans_t itrans, otrans;
	whiteTransform *itrans, *otrans;
	
	if(whitenin==1)
		itrans = new axisAlignedTransform(1, indep);
	else if(whitenin==2) {
		itrans = new axisAlignedTransform(2, indep);
		whitenin=1;
	} else if(whitenin==3) {
		itrans = new axisAlignedTransform(3, indep);
		whitenin=1;
	} else if(whitenin==4)
		itrans = new histEqualTransform();
	
	if(whitenout==1)
		otrans = new axisAlignedTransform(1, indep);
	else if(whitenout==2) {
		otrans = new axisAlignedTransform(2, indep);
		whitenout=1;
	} else if (whitenout==3) {
		otrans = new axisAlignedTransform(3, indep);
		whitenout=1;
	} else if(whitenout==3)
		otrans = new histEqualTransform();
		
	std::vector <float> wmean(td.nout, 0.0), wsigma(td.nout, 0.0);
	if (whitenout==4) {
		for( size_t i = 0; i < td.cumoutsets[td.ndata]; i++ ) {
			for( size_t j = 0; j < td.nout; j++ ) {
				td.outputs[i*td.nout+j] = td.outputs[i*td.nout+j];
				wmean[j] += td.outputs[i*td.nout+j];
				wsigma[j] += pow(td.outputs[i*td.nout+j], 2.0);
			}
		}
	}
	
	if (whitenin!=0) {
		td.generateWhiteData(itrans, td, 1);
		if( vdata ) itrans->apply(vd.inputs,vd.inputs,vd.cuminsets[vd.ndata]*vd.nin);
	} 
	if (whitenout!=0) {
		if( !autoencoder )
		{
			td.generateWhiteData(otrans, td, 2);
			if( vdata ) otrans->apply(vd.outputs,vd.outputs,vd.cumoutsets[vd.ndata]*vd.nout);
		}
		else
		{
			otrans=itrans;
			otrans->applyScalingOnly(td.inputs, td.acc, td.acc, td.cumoutsets[td.ndata]*td.nout);
			if( vdata ) otrans->applyScalingOnly(vd.inputs, vd.acc, vd.acc, vd.cumoutsets[vd.ndata]*vd.nout);
		}
			
		if (!classnet && !readacc && wnoise) {
			for (i=0;i<td.cumoutsets[td.ndata]*td.nout;i++) td.acc[i]=1./sigma;
			if( vdata ) for (i=0;i<vd.cumoutsets[vd.ndata]*vd.nout;i++) vd.acc[i]=1./sigma;
		}
	}
	
	if (whitenin==4) {
		whiteninORIG=whitenin;
		whitenin=0;
	}
	
	if (whitenout==4) {
		if (noise) {
			for( size_t j = 0; j < td.nout; j++ )
				wsigma[j] = sqrt(fmax(0.0, wsigma[j]/td.cumoutsets[td.ndata] - pow(wmean[j]/td.cumoutsets[td.ndata], 2.0)));
				
			for( size_t i = 0; i < td.cumoutsets[td.ndata]; i++ )
				for( size_t j = 0; j < td.nout; j++ )
					td.acc[i*td.nout+j] = wsigma[j];
		}
		if( noise && vdata ) {
			for( size_t i = 0; i < vd.cumoutsets[vd.ndata]; i++ )
				for( size_t j = 0; j < vd.nout; j++ )
					vd.acc[i*vd.nout+j] = wsigma[j];
		}
		whitenout=0;
	}
	
	// Set up the mini-batch
	int tndstore=(int)td.ndata;
	if (frac>1.)
		frac=1.;
	else if (frac<0.)
		frac=0.1;
	td.ndata=(int)(frac*(float)td.ndata);
	
	// initialize the NN
	if(td.nincat>1) nhid.insert(nhid.begin(),td.nincat);
	size_t nlayers = nhid.size()+2;
	size_t* nnodes = new size_t [nlayers];
	nnodes[0] = td.nin+1;			// nodes (including biases) in the input layer
	//if( nhid > 0 ) nnodes[1] = nhid+1;	// nodes (including biases) in the hidden layer
	for( size_t i = 0; i < nhid.size(); i++ ) nnodes[i+1] = nhid[i]+1;
	nnodes[nlayers-1] = td.nout;		// nodes in the output layer
	
	*nlayerspass = nlayers;
	for (i=0;i<nlayers;i++) nnodespass[i]=nnodes[i];
	
	if( autoencoder && td.nin != td.nout ) {		
		fprintf(stderr,"nin is not equal to nout, can not run autoencoder on this data-set. Aborting!");
		abort();
	}
	if( autoencoder && pretrain )
	{
		if( nhid.size()%2 == 0 ) pretrain = false;
		if( pretrain )
		{
			for( size_t i = 1; i < nlayers/2; i++ )
			{
				if( nnodes[i] != nnodes[nlayers-1-i] )
				{
					pretrain = false;
					break;
				}
			}
		}
	}
	
	// set layer type (linear/non-linear)
	//bool linear[nlayers];
	int linear[nlayers];
	if( strlen(linearlayers) == 0 )
	{
		// set input & output layers as linear & rest as non-linear (sigmoid)
		//linear[0] = true;
		//linear[nlayers-1] = autoencoder && pretrain ? false : true;
		//for( size_t i = 1; i < nlayers-1; i++ ) linear[i] = false;
		//if( autoencoder && pretrain ) linear[nlayers/2] = true;
		linear[0] = 0;
		linear[nlayers-1] = autoencoder && pretrain ? 1 : 0;
		for( size_t i = 1; i < nlayers-1; i++ ) linear[i] = 1;
		if( autoencoder && pretrain ) linear[nlayers/2] = 0;
	}
	else
	{
		if( strlen(linearlayers) != nlayers-1 )
		{
			//fprintf(stderr,"You need to specify layer types for %d layers (%d hidden and 1 output). No characters other than [TtFf] are allowed.\n",int(nlayers-1),int(nlayers-2));
			fprintf(stderr,"You need to specify layer types for %d layers (%d hidden and 1 output). Only integers {0,1,2,3,4,5} are allowed.\n",int(nlayers-1),int(nlayers-2));
			abort();
		}
		//linear[0] = true;	// input layer is always linear
		linear[0] = 0;
		for( size_t i = 0; i < nlayers-1; i++ )
		{
			char c = linearlayers[i];
			/*if( c == 'F' || c == 'f' )
			{
				linear[i+1] = false;
			}
			else if( c == 'T' || c == 't' )
			{
				linear[i+1] = true;
			}
			else
			{
				fprintf(stderr,"Invalid character in linear_layers. No characters other than [TtFf] are allowed.\n");
				abort();
			}*/
			if( c == '0' )
				linear[i+1] = 0;
			else if( c == '1' )
				linear[i+1] = 1;
			else if( c == '2' )
				linear[i+1] = 2;
			else if( c == '3' )
				linear[i+1] = 3;
			else if( c == '4' )
				linear[i+1] = 4;
			else if( c == '5' )
				linear[i+1] = 5;
			else 
			{
				fprintf(stderr,"Invalid selection in activation. Only integers {0,1,2,3,4,5} are allowed.\n");
				abort();
			}
		}
	}

	NeuralNetwork *nn;
	if (classnet)
		nn = new FeedForwardClassNetwork(nlayers,nnodes,recurrent,norbias,linear);
	else
		nn = new FeedForwardNeuralNetwork(nlayers,nnodes,recurrent,norbias,hhpl,linear);
	nn->SetZero(td.nincat,td.ninclasses);
	
	// Write out otrans if output layer is non-linear
	//if (whitenout==1 && !nn->linear[nn->nlayers-1])
	if (whitenout==1 && nn->linear[nn->nlayers-1]>0)
	{
		std::string otransfname  = concatenate(root2,"otrans.txt");
		std::ofstream otransfile(otransfname.c_str());
		otrans->write(otransfile);
		otransfile.close();
	}
	
	// initialize the PredictedData object
	PredictedData pd(nn->totnnodes, nn->totrnodes, td.ndata, td.nin, td.nout, td.cuminsets[td.ndata], td.cumoutsets[td.ndata], td.ncat);
	PredictedData pvd;
	if( vdata ) pvd=PredictedData(nn->totnnodes, nn->totrnodes, vd.ndata, vd.nin, vd.nout, vd.cuminsets[vd.ndata], vd.cumoutsets[vd.ndata], vd.ncat);
	
	int Npar = nn->nparams();
	
	if (myid==0 && verbose>0) {
		PrintIntro(classnet,resuming,prior,nn,(int)td.ndata,(int)vd.ndata,frac,tndstore,noise,wnoise,histmaxent,
			   recurrent,whitenin,whitenout,stopf);
	}

	/*bool lintest=false;
	for( size_t i = 0; i < nlayers; i++ )
		if( linear[i] > 1 ) lintest=true;
	if( lintest && pretrain )
		fprintf(stderr,"Warning! Pretraining is not designed for activation functions other than linear (0) and sigmoid (1).\n\n");*/
	
	// for all
	unsigned int iseed;
	if( fixseed )
		iseed = fixedseed;
	else
		iseed = (unsigned int)time(NULL);
	srand(iseed);
	long seed = rand();
	seed += 123321;
	seed *= -1;
	int nalphas=1,niter=0,neval=0,dotrials=1,counter=0,accepted,bestniter=0,counterT=0;
	float logZ,oldP,lnew=-FLT_MAX,temp=0.,VCorr=0.,ratio=0.,lnOmega=-FLT_MAX,Verrsqr=0.;
	float newVlogL=-FLT_MAX,oldVlogP,TlogC=0.,VlogC=0.,logPr;
	float newVScore=-FLT_MAX,oldVScore,temp2=0.,temp3=0.;
	float Omega,Ovar,scoreT,scoreV,score;
	float *x=(float *)malloc(Npar*sizeof(float));
	float *alphas=(float *)malloc(nalphas*sizeof(float));
	float *gamma=(float *)malloc(nalphas*sizeof(float));
	float *beta=(float *)malloc(td.nout*sizeof(float));
	float *best=(float *)malloc((Npar+2*nalphas+td.nout+2)*sizeof(float));
	
	NN_args args;
	args.np=Npar;
	args.td=&td;
	args.pd=&pd;
	args.nn=nn;
	args.alphas=alphas;
	args.prior=prior;
	args.noise=noise;
	args.nalphas=nalphas;
	args.Bcode=true;
	args.lnsrch=lnsrch;
	
	NN_args vargs;
	if( vdata ) {
		vargs.np=Npar;
		vargs.td=&vd;
		vargs.pd=&pvd;
		vargs.nn=nn;
		vargs.alphas=alphas;
		vargs.prior=prior;
		vargs.noise=noise;
		vargs.nalphas=nalphas;
		vargs.Bcode=true;
		vargs.lnsrch=lnsrch;
	}
	
	AlpOmTable Save;
	Save.Nsize=10;
	Save.Ntable=0;
	Save.X=(float *)malloc(Save.Nsize*sizeof(float));
	Save.Y=(float *)malloc(Save.Nsize*sizeof(float));
	Save.V=(float *)malloc(Save.Nsize*sizeof(float));
	
	// Initialise the starting point
	if (resuming) {
		if (myid==0 && verbose>0) printf("Reading in saved weights.\n");
		nn->read(dumpfile,&rate,alphas,beta);
		nn->SetZero(td.nincat,td.ninclasses);
		if (!classnet && !resetsigma && !readacc) {
			for (n=0;n<td.ndata;n++)
				for (i=0;i<td.ntime[n]-td.ntimeignore[n];i++)
					for (j=0;j<td.nout;j++)
						td.acc[td.cumoutsets[n]*td.nout+i*td.nout+j]=beta[j];
			if( vdata ) {
				for (n=0;n<vd.ndata;n++)
					for (i=0;i<vd.ntime[n]-vd.ntimeignore[n];i++)
						for (j=0;j<vd.nout;j++)
							vd.acc[vd.cumoutsets[n]*vd.nout+i*vd.nout+j]=beta[j];
			}
		}
		
		if (whitenin==1) nn->whitenWeights(itrans,1);
		if (nn->linear[nn->nlayers-1]==0 && whitenout==1) {
			otrans->applyScalingOnly(td.inputs,td.acc,td.acc,td.cumoutsets[td.ndata]*td.nout);
			if( vdata ) otrans->applyScalingOnly(vd.inputs,vd.acc,vd.acc,vd.cumoutsets[vd.ndata]*vd.nout);
			nn->whitenWeights(otrans,2);
		}

		nn->getweights(x);
		
		if (randweights!=0.0) {
			if (myid==0) {
				if (verbose>0) printf("Adding a random offset to the saved weights ~N(0,%g).\n",randweights);
				for (i=0;i<Npar;i++) {
					x[i]+=gasdev(&seed)*randweights;
				}
			}
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
	    		MPI_Bcast(x, Npar, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
			nn->setweights(x);
		}
	} else {
		//stdev = pretrain ? 0.1E0 : 0.01E0;
		//stdev=0.1;
		if( !pretrain )
		{
			if (myid==0 && verbose>0) printf("Setting random weights ~N(0,%g).\n",stdev);
			for(;;)
			{
				if (myid==0) {
					for (i=0;i<Npar;i++) x[i]=gasdev(&seed)*stdev;
					/*if (nn->nlayers>2) {
						int k=-1;
						for (i=0;i<nn->nnodes[1]-1;i++) {
							for (j=0;j<nn->nnodes[0]-1;j++) {
								k++;
								x[k]=gasdev(&seed);
							}
							k++;
						}
					}*/
				}
#ifdef PARALLEL
				MPI_Barrier(MPI_COMM_WORLD);
	    			MPI_Bcast(x, Npar, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
				lnew=logLike(x,&args);
				if(hhpl==2) {
					if(whitenin==1) nn->unwhitenWeights(itrans,1);
					if(whitenout==1) nn->unwhitenWeights(otrans,2);
					lnew=nn->HHPscore(td,pd,doforward);
					if(whitenin==1) nn->whitenWeights(itrans,1);
					if(whitenout==1) nn->whitenWeights(otrans,2);
					std::cerr << -lnew << "\n";
					if(-lnew<0.484) break;
				}
				else
					break;
			}
		}
		else
		{
			if( myid == 0 )
			{
				if (verbose>0) printf("Pre-training the network's hidden layers.\n");
				pretrainnet(nn, td, autoencoder, nepoch, stdev, seed);
				nn->getweights(x);
			}
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
	    		MPI_Bcast(x, Npar, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
				
			lnew=logLike(x,&args);
		}
	}
	
	if (myid==0 && verbose>0) printf("Beginning H-F optimizaton.\n\n");
	
	// Calculate some initial values
	if (resuming && !resetalpha) {
		FindGamma(&args,alphas,gamma,&dotrials,&temp,myid,doprint,useSD,mu,&seed,verbose);
	} else {
		InitialiseAlpha(x,&args,&Save,rate,&Omega,&Ovar,myid,verbose);
		for (i=0;i<nalphas;i++) gamma[i]=0.;
	}

	if(hhps) {
		if(whitenin==1) nn->unwhitenWeights(itrans,1);
		if(whitenout==1) nn->unwhitenWeights(otrans,2);
		if( vdata )
			newVScore=nn->HHPscore(vd,pvd,doforward);
		else
			newVScore=nn->HHPscore(td,pd,doforward);
		if(whitenin==1) nn->whitenWeights(itrans,1);
		if(whitenout==1) nn->whitenWeights(otrans,2);
	}
	
	if (!classnet) {
		if( noise ) nn->logC(td,TlogC);
		if( vdata ) {
			for (n=0;n<vd.ndata;n++)
				for (i=0;i<vd.ntime[n]-vd.ntimeignore[n];i++)
					for (j=0;j<vd.nout;j++)
						vd.acc[vd.cumoutsets[n]*vd.nout+i*vd.nout+j]=td.acc[j];
						
			if( noise ) nn->logC(vd,VlogC);
		} else {
			VlogC=TlogC;
		}
	}
	
	// Calculate values for starting point
	lnew=logLike(x,&args);
	nn->logPrior(alphas,logPr);
	if( vdata ) {
		newVlogL=logLike(x,&vargs);
		nn->correlations(vd,pvd);
		nn->ErrorSquared(vd,pvd);
		size_t j = 0;
		for (VCorr=0.,Verrsqr=0.,i=0;i<vd.nout;i++)
		{
			if(!std::isnan(pvd.corr[i]))
			{
				VCorr+=pvd.corr[i];
				j++;
			}
			Verrsqr+=pvd.errsqr[i];
		}
		VCorr/=(float)j;
		Verrsqr/=(float)vd.nout;
	} else {
		newVlogL=lnew;
		nn->correlations(td,pd);
		nn->ErrorSquared(td,pd);
		size_t j = 0;
		for (VCorr=0.,Verrsqr=0.,i=0;i<td.nout;i++)
		{
			if(!std::isnan(pd.corr[i]))
			{
				VCorr+=pd.corr[i];
				j++;
			}
			Verrsqr+=pd.errsqr[i];
		}
		VCorr/=(float)j;
		Verrsqr/=(float)td.nout;
	}
	if(std::isnan(VCorr) || std::isinf(VCorr)) VCorr = 0.0;
	
	// Save as best point
	best[Npar+2*nalphas+td.nout+1]=rate;
	if (hhps) {
		best[Npar+2*nalphas+td.nout]=newVScore;
	} else {
		if (stopf==1)
			best[Npar+2*nalphas+td.nout]=VCorr;
		else if (stopf==2)
			best[Npar+2*nalphas+td.nout]=newVlogL+VlogC;
		else if (stopf==3)
			best[Npar+2*nalphas+td.nout]=newVlogL+VlogC+logPr;
		else if (stopf==4)
			best[Npar+2*nalphas+td.nout]=-Verrsqr;
	}
	for (i=0;i<Npar;i++) best[i]=x[i];
	for (i=0;i<nalphas;i++) best[Npar+i]=alphas[i];
	for (i=0;i<nalphas;i++) best[Npar+nalphas+i]=gamma[i];
	if (!classnet) for (i=0;i<td.nout;i++) best[Npar+2*nalphas+i]=beta[i];
	
	float bestT=lnew+TlogC;
	
	if (myid==0 && verbose>1) {
		printf("Starting optimization from a point with the following values:\n");
		printf("Training logP=%g and logL=%g and Chisq=%g\n",lnew+TlogC+logPr,lnew+TlogC,-2.0*lnew);
		if (vdata) printf("Validation logP=%g and logL=%g and Chisq=%g\n",newVlogL+VlogC+logPr,newVlogL+VlogC,-2.0*newVlogL);
		if (hhps) {
			printf("HHP score = %g\n",newVScore);
		} else {
			printf("Correlation = %g and Error squared = %g\n",VCorr,Verrsqr);
		}
		printf("Starting value of convergence criterion = %g\n",best[Npar+2*nalphas+td.nout]);
		printf("\n");
	}
	
	if (resuming) AdjustAlpha(x,alphas,gamma,&args,classnet,&ratio,&dotrials,niter,rate,&Omega,&Ovar,&Save,otrans,myid,doprint,aim,histmaxent,whitenin,whitenout,useSD,mu,&seed,verbose);
	
	bool alphaupdated=true;
	float newV,oldV,newT,oldT;
	int nbounce[] = {0, 0};
	bool bounce[bouncecheck[1]];
	for(i=0;i<bouncecheck[1];i++) bounce[i]=0;
	
	// Loop over iterations
	do {
		if (maxniter==0) break;
		
		niter++;
		if (niter%printfreq==0) doprint=true;
		else doprint=false;
		
		if(alphaupdated) {
			if (!classnet) {
				if( noise ) nn->logC(td,TlogC);
				if( vdata ) {
					for (n=0;n<vd.ndata;n++)
						for (i=0;i<vd.ntime[n]-vd.ntimeignore[n];i++)
							for (j=0;j<vd.nout;j++)
								vd.acc[vd.cumoutsets[n]*vd.nout+i*vd.nout+j]=td.acc[j];
					
					if( noise ) nn->logC(vd,VlogC);
				} else {
					VlogC=TlogC;
				}
			}
			
			lnew=logLike(x,&args);
			nn->logPrior(alphas,logPr);
			if( vdata ) {
				nn->logLike(vd,pvd,newVlogL);
				nn->correlations(vd,pvd);
				nn->ErrorSquared(vd,pvd);
				size_t j = 0;
				for (VCorr=0.,Verrsqr=0.,i=0;i<vd.nout;i++)
				{
					if(!std::isnan(pvd.corr[i]))
					{
						VCorr+=pvd.corr[i];
						j++;
					}
					Verrsqr+=pvd.errsqr[i];
				}
				VCorr/=(float)j;
				Verrsqr/=(float)vd.nout;
			} else {
				newVlogL=lnew;
				nn->correlations(td,pd);
				nn->ErrorSquared(td,pd);
				size_t j = 0;
				for (VCorr=0.,Verrsqr=0.,i=0;i<td.nout;i++)
				{
					if(!std::isnan(pd.corr[i]))
					{
						VCorr+=pd.corr[i];
						j++;
					}
					Verrsqr+=pd.errsqr[i];
				}
				VCorr/=(float)j;
				Verrsqr/=(float)td.nout;
			}
			if(std::isnan(VCorr) || std::isinf(VCorr)) VCorr = 0.0;
			
			oldP=lnew+TlogC;
			if(hhps) {
				oldVScore=newVScore;
			} else {
				if(stopf==1)
					oldVlogP=VCorr;
				else if(stopf==2)
					oldVlogP=newVlogL+VlogC;
				else if(stopf==3)
					oldVlogP=newVlogL+VlogC+logPr;
				else if(stopf==4)
					oldVlogP=-Verrsqr;
			}
			
			if(hhps) {
				if(whitenin==1) nn->unwhitenWeights(itrans,1);
				if(whitenout==1) nn->unwhitenWeights(otrans,2);
				if( vdata )
					newVScore=nn->HHPscore(vd,pvd,doforward);
				else
					newVScore=nn->HHPscore(td,pd,doforward);
				if(whitenin==1) nn->whitenWeights(itrans,1);
				if(whitenout==1) nn->whitenWeights(otrans,2);
			}
			
			if(useSD==1) mu=alphas[0]/30.0;
			
			//if (doprint && myid==0 && verbose>1) printf("Iteration %d started with logP=%g and logL=%g\n",niter,lnew+TlogC+logPr,lnew+TlogC);
		} else {
			//if (doprint && myid==0 && verbose>1) printf("Iteration %d started with logP=%g and logL=%g\n",niter,lnew+TlogC+logPr,lnew+TlogC);
			oldP=lnew+TlogC;
			if(hhps) {
				oldVScore=newVScore;
			} else {
				if(stopf==1)
					oldVlogP=VCorr;
				else if(stopf==2)
					oldVlogP=newVlogL+VlogC;
				else if(stopf==3)
					oldVlogP=newVlogL+VlogC+logPr;
				else if(stopf==4)
					oldVlogP=-Verrsqr;
			}
		}
		
		// perform optimization
		HessianFreeOpt(x,&args,&lnew,rate,useSD,mu);
		
		// get logL and logP for best-fit network returned
		//lnew=logLike(x,&args);  // unnecessary since done at end of HessianFreeOpt
		nn->logPrior(alphas,logPr);
		if (hhps) {
			scoreT=nn->HHPscore(td,pd);
			if( vdata ) {
				scoreV=nn->HHPscore(vd,pvd);
				score=sqrt(((float)td.ndata*scoreT*scoreT+(float)vd.ndata*scoreV*scoreV)/((float)td.ndata+(float)vd.ndata));
			}
			else {
				scoreV=scoreT;
				score=sqrt(((float)td.ndata*scoreT*scoreT)/((float)td.ndata));
			}
		}
		
		// Print max log-posterior value found
		//if (doprint && myid==0 && verbose>1) printf("Iteration %d returned max logP=%g and logL=%g and Chisq=%g\n",niter,lnew+TlogC+logPr,lnew+TlogC,-2.0*lnew);
		if (doprint && myid==0 && verbose>1) printf("Step %d returned:\nTraining   logP=%g and logL=%g and Chisq=%g\n",niter,lnew+TlogC+logPr,lnew+TlogC,-2.0*lnew);
		
		if(lnew+TlogC>bestT) {
			bestT=lnew+TlogC;
			counterT=0;
		} else {
			counterT++;
		}
		
		// Analyse correlations (and error rate for classnet)
		nn->correlations(td,pd);
		nn->ErrorSquared(td,pd);
		size_t j = 0;
		for (temp=0.,temp2=0.,i=0;i<td.nout;i++)
		{
			if(!std::isnan(pd.corr[i]))
			{
				temp+=pd.corr[i];
				j++;
			}
			temp2+=pd.errsqr[i];
		}	
		temp/=(float)j;
		temp2/=(float)td.nout;
		if( !vdata ) {
			VCorr=temp;
			if(std::isnan(VCorr) || std::isinf(VCorr)) VCorr = 0.0;
			Verrsqr=temp2;
			newVlogL=lnew;
		}
		if (classnet) {
			nn->CorrectClass(td,pd);
			temp3=0E0;
			for (i=0;i<td.ncat;i++) temp3+=pd.predrate[i];
			temp3/=float(td.ncat);
			if (doprint && myid==0 && verbose>1) {
				printf("correlation = %g\t error squared = %g\t",temp,temp2);
				printf("%g%% in correct class\n",temp3*100.);
				if (printseparate) {
					printf("Correct class for each category: ");
					for (i=0;i<td.ncat;i++) printf("%g%% ",pd.predrate[i]*100);
					printf("\n");
				}
			}
		} else {
			if (doprint && myid==0 && verbose>1) printf("combined correlation = %g\t error squared = %g\n",temp,temp2);
		}
		
		// Look at quality of fit for validation data
		if(hhps) {
			if(whitenin==1) nn->unwhitenWeights(itrans,1);
			if(whitenout==1) nn->unwhitenWeights(otrans,2);
			if( vdata )
				newVScore=nn->HHPscore(vd,pvd,doforward);
			else
				newVScore=nn->HHPscore(td,pd,doforward);
			if(whitenin==1) nn->whitenWeights(itrans,1);
			if(whitenout==1) nn->whitenWeights(otrans,2);
		}
		if( vdata ) {
			nn->logLike(vd,pvd,newVlogL);

			//if (doprint && myid==0 && verbose>1) printf("Validation data has logP=%g and logL=%g and Chisq=%g\n",newVlogL+VlogC+logPr,newVlogL+VlogC,-2.0*newVlogL);
			if (doprint && myid==0 && verbose>1) {
				printf("Validation logP=%g and logL=%g and Chisq=%g\n",newVlogL+VlogC+logPr,newVlogL+VlogC,-2.0*newVlogL);
				if (hhps) printf("Validation data has score=%g\n",-newVScore);
			}
			nn->correlations(vd,pvd);
			nn->ErrorSquared(vd,pvd);
			size_t j = 0;
			for (temp=0.,temp2=0.,i=0;i<vd.nout;i++)
			{
				if(!std::isnan(pvd.corr[i]))
				{
					temp+=pvd.corr[i];
					j++;
				}
				temp2+=pvd.errsqr[i];
			}
			temp/=(float)j;
			temp2/=(float)vd.nout;
			VCorr=temp;
			Verrsqr=temp2;
			if(std::isnan(VCorr) || std::isinf(VCorr)) VCorr = 0.0;
			if (classnet) {
				nn->CorrectClass(vd,pvd);
				temp3=0E0;
				for (i=0;i<td.ncat;i++) temp3+=pvd.predrate[i];
				temp3/=float(td.ncat);
				if (doprint && myid==0 && verbose>1) {
					printf("correlation = %g\t error squared = %g\t",temp,temp2);
					printf("%g%% in correct class\n",temp3*100.);
					if (printseparate) {
						printf("Correct class for each category: ");
						for (i=0;i<td.ncat;i++) printf("%g%% ",pvd.predrate[i]*100);
						printf("\n");
					}
				}
			} else {
				if (doprint && myid==0 && verbose>1) printf("combined correlation = %g\t error squared = %g\n",temp,temp2);
			}
		}
		
		if (doprint && myid==0 && verbose>1) {
			if (hhps) printf("hhps score is %1.6lf (%1.6lf,%1.6lf)\n",score,-scoreT,-scoreV);
		}
		
		// check if the new network is better on validation and save if it is
		if(hhps) {
			newV=newVScore;
			oldV=oldVScore;
		} else {
			if(stopf==1)
				newV=VCorr;
			else if(stopf==2)
				newV=newVlogL+VlogC;
			else if(stopf==3)
				newV=newVlogL+VlogC+logPr;
			else if(stopf==4)
				newV=-Verrsqr;
			oldV=oldVlogP;
		}
		newT=lnew+TlogC;
		oldT=oldP;
		
		// check if train & test datasets are diverging
		/*j=0;
		for(i=0;i<bouncecheck[1]-1;i++) {
			bounce[i]=bounce[i+1];
			j+=bounce[i];
		}
		if((newT>oldT && newV<oldV) || (newT<oldT && newV>oldV))
			bounce[bouncecheck[1]-1]=1;
		else
			bounce[bouncecheck[1]-1]=0;
		j+=bounce[bouncecheck[1]-1];
		if(j>=bouncecheck[0]) {
			printf("Converged because of divergence between predictions on train and test data-sets.\n");
			break;
		}*/
		
				
		// adjust the rate parameter
		nbounce[1]++;
		if(newV<oldV) nbounce[0]++;
		if(nbounce[1] >= bouncecheck[1] || nbounce[0] > bouncecheck[0])
		{
			if(nbounce[0] > bouncecheck[0])
			{
				rate = fmax(ratemin, rate/1.5);
				//if(useSD==1) mu *= 2;
			}
			else if(nbounce[1] >= bouncecheck[1] && nbounce[0] < bouncecheck[0])
			{
				rate = fmin(1.0, rate*1.5);
				//if(useSD==1) mu = fmax(mu/2.0, 1E-6);
			}
			
			nbounce[0] = nbounce[1] = 0;
		}
		if (doprint && myid==0 && verbose>2) {
			printf("rate       = %g\n",rate);
			if(useSD==1) printf("mu         = %g\n",mu);
		}
		
	
		if (newV>=oldV) {
			if (newV>best[Npar+2*nalphas+td.nout]) {
				counter=0;
				bestniter=niter;
				best[Npar+2*nalphas+td.nout+1]=rate;
				best[Npar+2*nalphas+td.nout]=newV;
				for (i=0;i<Npar;i++) best[i]=x[i];
				for (i=0;i<nalphas;i++) best[Npar+i]=alphas[i];
				for (i=0;i<nalphas;i++) best[Npar+nalphas+i]=gamma[i];
				for (i=0;i<td.nout;i++) best[Npar+2*nalphas+i]=td.acc[i];
			} else counter++;
		} else counter++;
		
		if (doprint) {
			if (myid==0) printf("Best value of convergence criterion = %g\n\n",best[Npar+2*nalphas+td.nout]);
			
			nn->setweights(best);
			if (whitenin==1) nn->unwhitenWeights(itrans,1);
			if (whitenout==1 && nn->linear[nn->nlayers-1]==0) {
				nn->unwhitenWeights(otrans,2);
				otrans->inverseScalingOnly(td.inputs,&best[Npar+2*nalphas],&best[Npar+2*nalphas],td.nout);
			}
			
			if (classnet) {
				nn->write(dumpfile,best[Npar+2*nalphas+td.nout+1],&best[Npar]);
				nn->writeSimple(simpfile,classnet);
			} else {
				nn->write(dumpfile,best[Npar+2*nalphas+td.nout+1],&best[Npar],&best[Npar+2*nalphas]);
				nn->writeSimple(simpfile,classnet);
			}
				
			if (autoencoder)
			{
				nn->write(AEencoderfile, true, 1);
				nn->write(AEdecoderfile, true, 2);
			}
			
			nn->setweights(x);
			if (whitenout==1) otrans->applyScalingOnly(td.inputs,&best[Npar+2*nalphas],&best[Npar+2*nalphas],td.nout);
		}
		
		alphaupdated=AdjustAlpha(x,alphas,gamma,&args,classnet,&ratio,&dotrials,niter,rate,&Omega,&Ovar,&Save,otrans,myid,doprint,aim,histmaxent,whitenin,whitenout,useSD,mu,&seed,verbose);
		
		if(niter>=maxniter) break;
		
	} while ((!resume && bestniter<10 && niter<1000) || (prior && counter<stopcounter) || (!prior && niter<1000 && counter<stopcounter && fabs(2.*(newV-oldV)/(newV+oldV))>tol));
	//} while (bestniter<10 || (prior && counter<stopcounter && counterT<stopcounter/2) || (!prior && niter<1000 && counter<stopcounter && counterT<stopcounter/2 && fabs(2.*(newV-oldV)/(newV+oldV))>tol));
	//} while (bestniter<10 || (prior && fabs(log(Omega))>tol && counter<stopcounter) || (!prior && niter<1000 && counter<stopcounter && fabs(2.*(newVlogL+VlogC-oldVlogP)/(newVlogL+VlogC+oldVlogP))>tol));
	
	doprint=true;

	if( myid == 0 ) printf("Network Training Complete.\n\n");
	
	// set x, alphas, and beta to the best again
	rate=best[Npar+2*nalphas+td.nout+1];
	for (i=0;i<Npar;i++) x[i]=best[i];
	nn->setweights(x);
	for (i=0;i<nalphas;i++) alphas[i]=best[Npar+i];
	for (i=0;i<nalphas;i++) gamma[i]=best[Npar+nalphas+i];
	if (!classnet) {
		for (n=0;n<td.ndata;n++)
			for (i=0;i<td.ntime[n]-td.ntimeignore[n];i++)
				for (j=0;j<td.nout;j++)
					td.acc[td.cumoutsets[n]*td.nout+i*td.nout+j]=best[Npar+2*nalphas+j];
		if( vdata ) {
			for (n=0;n<vd.ndata;n++)
				for (i=0;i<vd.ntime[n]-vd.ntimeignore[n];i++)
					for (j=0;j<vd.nout;j++)
						vd.acc[vd.cumoutsets[n]*vd.nout+i*vd.nout+j]=best[Npar+2*nalphas+j];
		}
	}
	
	if (myid==0 && dotrials==1 && verbose>0) printf("gamma average stddev/mean = %g\n",ratio/(float)niter);
	
	// Calculate the evidence of the best-fit network
	lnew=logLike(x,&args);
	if (!classnet && noise) nn->logC(td,TlogC);
	nn->logPrior(alphas,logPr);
	if( vdata ) {
		nn->logLike(vd,pvd,newVlogL);
		if (!classnet && noise) nn->logC(vd,VlogC);
	}
	if (prior && evidence) logZ=GetlogZ(x,&args,alphas,gamma,classnet,myid,useSD,mu,&seed,verbose);
	if (myid==0 && verbose>0) {
		printf("Found logP=%g, logL=%g, Chisq=%g",lnew+TlogC+logPr,lnew+TlogC,-2.0*lnew);
		if (prior && evidence) printf(", logZ=%g",logZ);
		printf("\n");
		if( vdata ) printf("      logP=%g, logL=%g, Chisq=%g\n",newVlogL+VlogC+logPr,newVlogL+VlogC,-2.0*newVlogL);
		printf("alpha = %g\n",alphas[0]);
	}
	
	float wt[nn->nweights],g[nn->nweights];
	nn->getweights(wt);
	float omicron=FindOmicron(wt,&args,rate,g);
		
	// Print predicted values for training and validation data
	if( myid == 0 )
	{
		PrintPredictions(dataout,whitenout,whitenin,itrans,otrans,&args,&args,omicron,classnet,printerror,autoencoder,verbose);
		if( vdata ) PrintPredictions(testout,whitenout,whitenin,itrans,otrans,&vargs,&args,omicron,classnet,printerror,autoencoder,verbose);
	}
	//printf("Unwhitening the network inputs.\n");
	// Unwhiten the network and data
	if(whitenin==1) {
		nn->unwhitenWeights(itrans,1);
		itrans->inverse(td.inputs,td.inputs,td.cuminsets[td.ndata]*td.nin);
		if( vdata ) itrans->inverse(vd.inputs,vd.inputs,vd.cuminsets[vd.ndata]*vd.nin);
	}
	//printf("Unwhitening the network outputs (rank %d).\n",myid);
	if(whitenout==1 && nn->linear[nn->nlayers-1]==0) {
		nn->unwhitenWeights(otrans,2);
		otrans->inverse(td.outputs,td.outputs,td.cuminsets[td.ndata]*td.nout);
		if( vdata ) otrans->inverse(vd.outputs,vd.outputs,vd.cuminsets[vd.ndata]*vd.nout);
		otrans->inverseScalingOnly(td.inputs,td.acc,td.acc,td.cumoutsets[td.ndata]*td.nout);
		if( vdata ) otrans->inverseScalingOnly(vd.inputs,vd.acc,vd.acc,vd.cumoutsets[vd.ndata]*vd.nout);
	}
	//printf("Unwhitening done. Calculating logL (rank %d).\n",myid);
	nn->getweights(x);
	temp=logLike(x,&args);
	if (vdata) temp=logLike(x,&vargs);
	//printf("logL done. Unwhitening predicted outputs if non-linear output layer (rank %d).\n",myid);
	// Unwhiten predicted data outputs if output layer is non-linear
	if(whitenout==1 && nn->linear[nn->nlayers-1]>0) {
		otrans->inversePredictedOutputs(pd.out,pd.out,td.nout,td.ndata,nn->totnnodes,td.cuminsets,td.cumoutsets,td.ntimeignore,td.ntime);
		if( vdata ) otrans->inversePredictedOutputs(pvd.out,pvd.out,vd.nout,vd.ndata,nn->totnnodes,vd.cuminsets,vd.cumoutsets,vd.ntimeignore,vd.ntime);
	}
	//printf("Calculating the correlations and error squared (rank %d).\n",myid);
	// Calculate the correlations and error squared
	nn->correlations(td,pd);
	if( vdata ) nn->correlations(vd,pvd);
	nn->ErrorSquared(td,pd);
	if (vdata) nn->ErrorSquared(vd,pvd);
	if (classnet) {
		nn->CorrectClass(td,pd);
		if( vdata ) nn->CorrectClass(vd,pvd);
	}
	//printf("Correlations and error squared calculated (rank %d).\n",myid);
	// Print  best network
	if (classnet) {
		nn->write(dumpfile,rate,alphas);
		//nn->writeSimple(simpfile,classnet);
	} else {
		nn->write(dumpfile,rate,alphas,&td.acc[td.cumoutsets[0]*td.nout]);
		//nn->writeSimple(simpfile,classnet);
	}
		
	if (autoencoder)
	{
		nn->write(AEencoderfile, true, 1);
		nn->write(AEdecoderfile, true, 2);
	}
	
	if(hhps) {
		if( vdata )
			newVScore=nn->HHPscore(vd,pvd,doforward);
		else
			newVScore=nn->HHPscore(td,pd,doforward);
	}
	if(hhps && myid==0 && verbose>0) printf("hhps score=%g\n",newVScore);
	
	if (myid==0 && verbose>0) printf("\n");
		
	// Print best-fit correlation (and error rate)
	if (classnet) {
		if (myid==0) {
			float esqr = 0.0; temp = 0.0;
			size_t j = 0;
			for (i=0;i<td.nout;i++)
			{
				if(!std::isnan(pd.corr[i]))
				{
					temp+=pd.corr[i];
					j++;
				}
				esqr+=pd.errsqr[i];
			}
			temp/=(float)j;
			esqr/=(float)td.nout;
			temp3=0E0;
			for (i=0;i<td.ncat;i++) temp3+=pd.predrate[i];
			temp3/=float(td.ncat);
			if (verbose>0) {
				printf("The training correlation is %g, error squared is %g, ",temp,esqr);
				printf("%g%% in correct class.\n",temp3*100.);
			}
			if (printseparate && verbose>0) {
				printf("Correct class for each category: ");
				for (i=0;i<td.ncat;i++) printf("%g%% ",pd.predrate[i]*100);
				printf("\n");
			}
			if( vdata )
			{
				esqr = 0.0; temp = 0.0;
				size_t j = 0;
				for (i=0;i<vd.nout;i++)
				{
					if(!std::isnan(pvd.corr[i]))
					{
						temp+=pvd.corr[i];
						j++;
					}
					esqr+=pvd.errsqr[i];
				}
				temp/=(float)j;
				esqr/=(float)vd.nout;
				temp3=0E0;
				for (i=0;i<td.ncat;i++) temp3+=pvd.predrate[i];
				temp3/=float(td.ncat);
				if (verbose>0) {
					printf("The validation correlation is %g, error squared is %g, ",temp,esqr);
					printf("%g%% in correct class.\n",temp3*100.);
				}
				if (printseparate && verbose>0) {
					printf("Correct class for each category: ");
					for (i=0;i<td.ncat;i++) printf("%g%% ",pvd.predrate[i]*100);
					printf("\n");
				}
			}
		}
	} else {
		size_t j = 0;
		for (temp=0.,temp2=0.,i=0;i<td.nout;i++)
		{
			if(!std::isnan(pd.corr[i]))
			{
				temp+=pd.corr[i];
				j++;
			}
			temp2+=pd.errsqr[i];
		}
		temp/=(float)j;
		temp2/=(float)td.nout;
		if (myid==0 && verbose>0) printf("The training correlation is %g, error squared is %g\n",temp,temp2);
		if( vdata ) {
			size_t j = 0;
			for (temp=0.,temp2=0.,i=0;i<vd.nout;i++)
			{
				if(!std::isnan(pvd.corr[i]))
				{
					temp+=pvd.corr[i];
					j++;
				}
				temp2+=pvd.errsqr[i];
			}
			temp/=(float)j;
			temp2/=(float)vd.nout;
			if (myid==0 && verbose>0) printf("The validation correlation is %g, error squared is %g\n",temp,temp2);
		}
	}
	
	FREEALLVARS
	if( whitenin>0 ) delete itrans;
	if( whitenout>0 && !autoencoder) delete otrans;
	delete nn;
	delete nnodes;
	fflush(stdout);fflush(stderr);
	return 0;
}
