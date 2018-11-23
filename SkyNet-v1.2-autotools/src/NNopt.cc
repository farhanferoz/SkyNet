#include "NNopt.h"
#include "linbcg.h"

std::string concatenate(std::string in1, std::string in2) {
	std::stringstream s;
	s << in1 << in2;
	return s.str();
}

std::string addnum(std::string in, int num) {
	std::stringstream s;
	s << in << num;
	return s.str();
}

void ErrorExit(std::string msg) {
	std::cerr << msg << "\n";abort();
}

float logLike(float* x, void *arg) {
	float logL=0.;
	NN_args *par = (NN_args *) arg;
	par->nn->setweights(x);
	par->nn->logLike(*(par->td), *(par->pd), logL);
	return logL;
}

void gradLike(float* grad, void *arg) {
	NN_args *par = (NN_args *) arg;
	par->nn->grad(*(par->td), *(par->pd), grad);
	
	/*float g[par->np];
	par->nn->grad_prior(par->alphas, g);
	for( size_t i = 0; i < par->np; i++ ) grad[i] += g[i];*/
}

float logPost(float *x, void *arg)
{
	float logP,logL;
	NN_args *par = (NN_args *) arg;
	logL=logLike(x,par);
	par->nn->logPrior(par->alphas,logP);
	return logL+logP;
}

float logPostMod(float *x, void *arg)
{
	float logP,logL;
	NN_args *par = (NN_args *) arg;
	logL=logLike(x,par);
	par->nn->logPrior(&(par->omicron),logP);
	return logL+logP;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  HessianFreeOpt
// 
// Purpose:   Uses the Hessian-free optimization method, implementing the
//            conjugate gradient from John Skilling (MemSys).
//-----------------------------------------------------------------------------
void HessianFreeOpt(float *x, NN_args *args, float *lnew, float r0, int useSD, float mu) {
	float temp;
	int i;
	
	float b[args->np], p[args->np];

	args->omicron = FindOmicron(x,args,r0,b);
	//gradLike(b,args);  // now being passed out by FindOmicron which does this calculation
	FindDirn(b,p,args,args->omicron,&temp,0,useSD,mu);

	if (args->lnsrch==1)
	{
		float dist,test=0.0;
		for (i=0;i<args->np;i++)
		{
			temp=fabs(p[i])/fmax(fabs(x[i]),1.0);
        	if (temp > test) test=temp;
		}
		float stepmin = FLT_EPSILON / test;

		// golden section search
		float fxn,xn[args->np];
		for (i=0;i<args->np;i++) xn[i]=x[i]+p[i];
		fxn = -logLike(xn,args);
		dist = goldenSectionSearch(args,x,p,fxn,stepmin,1.0,10.0,0.001);
		//printf("Moving distance %f\n",dist);
		for (i=0;i<args->np;i++) x[i]+=dist*p[i];
	}
	else if (args->lnsrch==2)
	{
		// linbcg search
		float dist,xnew[args->np],gp[args->np],stpmax=10.;
		bool check;
		//args->nn->grad_prior(args->alphas,gp);
		for (i=0;i<args->np;i++) gp[i] = b[i];
		linbcg_lnsrch(x,-logLike(x,args),gp,p,xnew,dist,temp,stpmax,check,args);
		if (check) fprintf(stderr,"Warning:  x is too close to xold\n");
		//printf("Moving distance %f\n",dist);
		for (i=0;i<args->np;i++) x[i] = xnew[i];
	}
	else
	{
		for (i=0;i<args->np;i++) x[i] += p[i];
	}
	
	*lnew = logLike(x,args);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  FindDirn
// 
// Purpose:   Perform the conjugate gradient optimization for the direction to
//            move in next.
//-----------------------------------------------------------------------------
int FindDirn(float *b, float *p, NN_args *args, float alpha, float *scalar, int what, int useSD, float mu) {
	int       Nmax   = (int)fmin(50,2*args->np);  // max number of transforms <= 2*Npar
	float*   Mock;             // vector for storing R.b
	float    utol   = utolg;   // tolerance, 0 = terminate on Nmax
	float*   v;                // Problem B  output vector f(A).b
	float*   w;                // Problem AB output vector f(A).b
	float    lower, upper;     // limits on b'.f(A).b
	float    hi;               // scalar for HiVector packaging
	sfnMatrix psfnMat[1];       // conjugate gradient & eigenstructure of A
	sMATRIX   A[1];             // transform matrix
	int       CALLvalue = 0;
	int       flag;             // function identifier
	int       i;
	
	// what={0,1,2,3} to be used for {finding next step, finding trace, finding determinant, finding prediction errors}
	switch (what) {
		case 0:
			flag=0;
			break;
		case 1:
			flag=4;
			break;
		case 2:
			flag=2;
			break;
		case 3:
			flag=0;
			break;
	}
	
	v=(float *)malloc(args->np*sizeof(float));
	w=(float *)malloc(args->np*sizeof(float));
	Mock=(float *)malloc(args->np*sizeof(float));
	
	// Assign sfnMatrix structure and setup user's matrix
	CALL( fnMatNew(psfnMat, args->np, Nmax) )
	A->Npar  = args->np;
	A->Mock  = Mock;
	A->alpha = alpha;
	A->args  = args;
	A->useSD = useSD;
	A->mu    = mu;
	
	// Use seed vector r to get subspace and eigenstructure defining P(A)
	CALL( i=fnMatPerform(psfnMat, A, b, alpha, utol) )
	//printf("%d transforms used\n",i);
	
	if (what==1 || what==2 || what==3) {
		 // Check bracketing scalars (could also check Lo and Hi vectors)
		lower = fnMatLoScalar(psfnMat, A, flag);
		upper = fnMatHiScalar(psfnMat, A, flag);
		//printf("\n %g  <=   b'.B^(-1).b   <= %g\n", lower, upper);
		*scalar = (upper+lower)/2.;
	} else {
		// Check Lo and Hi vectors (scalars would revert to those from original seed)
		fnMatLoVector(psfnMat, A, v, flag);
		//for( i = 0; i < Npar; ++i ) {printf("B^(-1).b  low vector %4d %g\n", i, v[i]);}
		fnMatHiVector(psfnMat, A, w, flag);
		SCALAR(&hi, A, w);
		VECTOR(w, A);
		hi = fnMat(A, 0.0, flag);
		//for( i = 0; i < Npar; ++i ) {printf("B^(-1).b from high vector %4d %g\n", i, hi * b[i] - w[i]);}
		for (i=0;i<args->np;i++) {p[i]=(v[i]+hi*b[i]-w[i])/2.;}
	}
	
	// Deallocate sfnMatrix structure
	fnMatOld(psfnMat);
	free(v);
	free(w);
	free(Mock);
	return 0;
	
	Exit:
	// Deallocate sfnMatrix structure
	fnMatOld(psfnMat);
	free(v);
	free(w);
	free(Mock);
	return CALLvalue;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  SCALAR
// 
// Purpose:   Transform's scalar     v'.A.v >= 0
//
//            Store somewhere in A (this implementation uses A->Mock)
//            whatever is needed to complete VECTOR's calculation of A.v,
//            such as just v itself
//                (which may need some repeat calculation in VECTOR)
//            or R.v if A = R'.R
//                (which is the traditional "OPUS/TROPUS" style)
//            or the final result A.v
//                (which wastes any extra calculation if VECTOR is not called).
//-----------------------------------------------------------------------------
int SCALAR(       //   O  0 = OK, -ve = error
float*  delta,   //   O  scalar v'.A.v
void*    ptrA,    // I O  & transform (called as void* from fnmatrix.c)
float*  v)       // I    input vector                                   [Ndim]
{
    sMATRIX* A     = (sMATRIX*)ptrA;
    int      Npar  = A->Npar;
    float*  Mock  = A->Mock;
    NN_args* par   = A->args;
    int      useSD = A->useSD;
    float   mu    = A->mu;
    int      i;
    if(useSD==1)
    	par->nn->Av(v, *(par->td), *(par->pd), 1.0, mu, Mock);
    else
    	par->nn->Av(v, *(par->td), *(par->pd), Mock);
    for( *delta=0.0,i = 0; i < Npar; ++i ) *delta += v[i] * Mock[i];
    return 0;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  VECTOR
// 
// Purpose:   Transform's vector     A.v
//
//            Use whatever the preceding SCALAR placed in matrix structure A.
//-----------------------------------------------------------------------------
int VECTOR(       //   O  0 = OK, -ve = error
float*  u,       //   O  A.v                                            [Ndim]
void*    ptrA)    // I    & transform (called as void* from fnmatrix.c)
{
    sMATRIX* A     = (sMATRIX*)ptrA;
    int      Npar  = A->Npar;
    float*  Mock  = A->Mock;
    int      i;
    for( i = 0; i < Npar; ++i ) u[i] = Mock[i];
    return 0;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMat
// 
// Purpose:   Function f(scalar x) defining required function f(A) of matrix
//
//            This implementation models simulatable functions of
//                        B = alpha * I + A 
//            either   f(A) = B^(-1)     (flag=0),
//            or       f(A) = B^(-1/2)   (flag=1),
//            or       f(A) = -log(B)    (flag=2),
//            or other functions as required.
//-----------------------------------------------------------------------------
float fnMat(     //   O  required function of scalar
void*    ptrA,    // I    definition of matrix
float   x,       // I    scalar operand = eigenvalue of modelled A
int      flag)    // I    function identifier
{
    sMATRIX* A = (sMATRIX*)ptrA;
    float   B = A->alpha + x;
    switch( flag )
    {
        case 0:
            return  1.0 / B;                         // direct inverse
        case 1:
            return  1.0 / sqrt(B);                   // inverse square root
        case 2:
            return  -log(B);                         // negative logarithm
        case 3:
            return  (pow(B, -0.001) - 1.0) / 0.001;  // approach to -log
        case 4:
	    //printf("eigenvalue passed in = %g\n",x);
	    return  x / B;			     // for trace calculation
	default:
            return  0.;                              // (alternatives...)
    }
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  AdjustAlpha
// 
// Purpose:   Adjusts the regularization constant(s) based on Equation 2.22
//            of David Mackay's thesis. Then adjusts the betas (noise scaling)
//            based on Equation 2.24 of David Mackay's thesis.
//-----------------------------------------------------------------------------
bool AdjustAlpha(float *x, float *alphas, float *gamma, NN_args *args, bool classnet, float *ratio, int *dotrials, int niter, 
		 float rz, float *Omega, float *Ovar, AlpOmTable *Save, whiteTransform *otrans, int myid, bool doprint, float aim, bool histmaxent,
		 int whitenin, int whitenout, int useSD, float mu, long *seed, int verbose) {
	float *gauss,*trAinv,stddev,*Mock,temp=0.,logdet=0.,gammasum=0.,c=1.;
	float r,alf1,alf2,y1,y2,y,gradL,sigma,logL=0.,logP=0.,S;
	int i,j,n,ntrial,nalphas,extra=0;
	bool Tcode,update=false;
	
	float beta[args->td->nout];
	
	if (args->nalphas>1) *dotrials=2;
	
	args->nn->logLike(*(args->td),*(args->pd),logL);
	args->nn->logPrior(alphas,logP);
	
	if (!classnet && args->noise) {
		c=fmin(1.0, sqrt(-2.*(logL+logP)/((float)(args->td->cumoutsets[args->td->ndata]*args->td->nout))));
		
		for (n=0;n<args->td->ndata;n++)
			for (i=0;i<args->td->ntime[n]-args->td->ntimeignore[n];i++)
				for (j=0;j<args->td->nout;j++)
					args->td->acc[args->td->cumoutsets[n]*args->td->nout+i*args->td->nout+j]/=c;
		for (j=0;j<args->td->nout;j++) beta[j]=args->td->acc[j];
		if (whitenout==1) otrans->inverseScalingOnly(args->td->inputs,beta,beta,args->td->nout);
		
		args->nn->logLike(*(args->td),*(args->pd),logL);
	}
	
	if (args->prior) {
		S=-logP/alphas[0];
		
		float gs[args->np];
		args->nn->grad(*(args->td),*(args->pd),gs);
		
		FindGamma(args,alphas,gamma,dotrials,ratio,myid,doprint,useSD,mu,seed,verbose);
		
		if (histmaxent) *Omega = (float)(args->td->ndata)*aim/(-2.*logL);
		else *Omega = gamma[0]*aim/(-2.*logP);
		
		*Ovar = OmegaVar(x,args,gs,&Tcode,false);
		
		//if (doprint && myid==0) printf("Writing alpha = %g, ln(Omega) = %g +/- %g\n",alphas[0],log(*Omega),*Ovar);
		WriteToTable(Save,alphas[0],*Omega,*Ovar);
		
		/*if (Tcode) printf("Tcode = true\t");
		else printf("Tcode = false\t");
		if (args->Bcode) printf("Bcode = true\n");
		else printf("Bcode = false\n");*/
		
		if (Tcode && args->Bcode) {
			ReadFromTable(Save,alphas[0],&y,&sigma);
			//if (doprint && myid==0) printf("Read alpha = %g, ln(Omega) = %g +/- %g\n",alphas[0],y,sigma);
			
			if (fabs(y)>=sigma) {
				
				for (gradL=0.,i=0;i<args->np;i++) gradL+=gs[i]*gs[i];
				
				r=1.+alphas[0]*sqrt((float)args->np)*rz/sqrt(gradL);
				
				if (alphas[0]*r>alphas[0]) alf1=alphas[0]*r;
				else alf1=alphas[0];
				if (alphas[0]/r<alphas[0]) alf2=alphas[0]/r;
				else alf2=alphas[0];
				
				//y1=log(gamma[0]*c*c/(2.*alf1*EW[0]));
				ReadFromTable(Save,alf1,&y1,&sigma);
				
				if (y1>0.) {
					alphas[0]=alf1;
				} else {
					//y2=log(gamma[0]*c*c/(2.*alf2*EW[0]));
					ReadFromTable(Save,alf2,&y2,&sigma);
					if (y2<=0.) {
						alphas[0]=alf2;
					} else {
						r=1.;
						do {
							r/=2.;
							alphas[0]=(alf1+alf2)/2.;
							//y=log(gamma[0]*c*c/(2.*alphas[0]*EW[0]));
							ReadFromTable(Save,alphas[0],&y,&sigma);
							if (y<=0.) {
								alf1=alphas[0];
								y1=y;
							} else {
								alf2=alphas[0];
								y2=y;
							}
						} while (r>FLT_EPSILON);
						if (y1-y2!=0.) {
							r=y1/(y1-y2);
							alphas[0]=alf1+(alf2-alf1)*r;
						}
					}
				}
				
				for (i=1;i<args->nalphas;i++) alphas[i]=alphas[0];
				update=true;
				//if (doprint && myid==0) printf("Alpha updated.\n");
			} else {
				//if (doprint && myid==0) printf("fabs(%g) < %g - no update.\n",y,sigma);
			}
		}
		
		if (doprint && myid==0 && verbose>2) {
			printf("alpha      = %g\n",alphas[0]);
			printf("Omega      = %g +/- %g\n",*Omega,*Ovar);
			printf("S          = %g\n",S);
		}
		
		if (!classnet && args->noise && !update) {
			for (n=0;n<args->td->ndata;n++)
				for (i=0;i<args->td->ntime[n]-args->td->ntimeignore[n];i++)
					for (j=0;j<args->td->nout;j++)
						args->td->acc[args->td->cumoutsets[n]*args->td->nout+i*args->td->nout+j]*=c;
			for (j=0;j<args->td->nout;j++) beta[j]=args->td->acc[j];
			if (whitenout==1) otrans->inverseScalingOnly(args->td->inputs,beta,beta,args->td->nout);
			
			args->nn->logLike(*(args->td),*(args->pd),logL);
		}
	}
	
	if (!classnet && args->noise && doprint && myid==0 && verbose>2) {
		printf("sigma      = {");
		for (i=0;i<args->td->nout-1;i++) printf("%g,",1./beta[i]);
		printf("%g}\n",1./beta[args->td->nout-1]);
	}
	
	
	if (doprint && myid==0 && ((!classnet && args->noise) || args->prior) && verbose>2) printf("\n");
	
	return update;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  GetlogZ
// 
// Purpose:   Calculates the log-evidence based on Equation 2.20 of David
//            Mackay's thesis. Many evaluations are done for different values
//            of log(|A|) returned by using different random vectors. Returns
//            the mean and prints the mean and standard deviation.
//-----------------------------------------------------------------------------
float GetlogZ(float *x, NN_args *args, float *alphas, float *gamma, bool classnet, int myid, int useSD, float mu, long *seed, int verbose) {
	float Z,temp,stddev,nlogdet;
	int i,j,ntrial,naccept=0;
	
	float gauss[args->np], Mock[args->np], beta[args->td->nout];
	
	ntrial=10;
	if (args->np<100) ntrial+=(100-args->np)/2;
	
	stddev=Z=0.;
	
	/*unsigned int iseed;
	if( fixseed )
		iseed = fixedseed;
	else
		iseed = (unsigned int)clock();
	srand(iseed);
	long seed = rand();
	seed += 123321;
	seed *= -1;*/
	
	if (!classnet)
		for (i=0;i<args->td->nout;i++) beta[i]=pow(args->td->acc[i],2.);
	
	for (i=0;i<ntrial;i++) {
		if( myid == 0 ) for (j=0;j<args->np;j++) gauss[j]=gasdev(seed);
#ifdef PARALLEL
		MPI_Barrier(MPI_COMM_WORLD);
    		MPI_Bcast(gauss, args->np, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
		FindDirn(gauss,Mock,args,alphas[0],&nlogdet,2,useSD,mu);
		args->nn->logZ(alphas,-nlogdet,*(args->td),*(args->pd),temp);
		//if (finite(temp)) {
		if (fabs(temp) <= std::numeric_limits<float>::max()) {
			Z+=temp;
			stddev+=temp*temp;
			naccept++;
		}
	}
	if (naccept!=0) {
		Z/=(float)naccept;
		stddev/=(float)naccept;
	} else {
		if (myid==0) fprintf(stderr,"No values accepted for logZ!\n");
	}
	stddev-=Z*Z;
	if (myid==0 && verbose>1) printf("log(Z) = %g +/- %g\n\n",Z,sqrt(stddev));
	
	return Z;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  InitialiseAlpha
// 
// Purpose:   Initialises the prior alpha according to the MemSys formula.
//-----------------------------------------------------------------------------
void InitialiseAlpha(float *x, NN_args *args, AlpOmTable *Save, float rz, float *Omega, float *Ovar, int myid, int verbose) {
	float gradL,logL;
	int i;
	
	args->nn->setweights(x);
	args->nn->logLike(*(args->td),*(args->pd),logL);
	
	if (args->prior) {
		float g[args->np];
		args->nn->grad(*(args->td),*(args->pd),g);
		for (gradL=0.,i=0;i<args->np;i++) gradL+=g[i]*g[i];
		
		*Omega=0.5;
		*Ovar=1.e20;
		WriteToTable(Save,1.e20,*Omega,*Ovar);
		
		args->alphas[0]=sqrt(gradL)/(sqrt((float)args->np)*rz);
		//args->alphas[0]=sqrt(gradL);
		
		if (myid==0 && verbose>2) printf("alpha  = %g\n\n",args->alphas[0]);
	} else {
		args->alphas[0]=0.;
		*Omega=0.5;
	}
	
	
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  FindOmicron
// 
// Purpose:   Finds the regularisation constant according to the MemSys formula.
//-----------------------------------------------------------------------------
float FindOmicron(float *x, NN_args *args, float r0, float *g) {
	float gg,d0,bmin,amin,omicron,temp;
	int i;
	
	//return args->alphas[0];
	
	//float g[args->np];
	
	args->nn->setweights(x);
	args->nn->logLike(*(args->td),*(args->pd),temp);
	args->nn->grad(*(args->td),*(args->pd),g);
	
	for (gg=0.,i=0;i<args->np;i++) gg+=g[i]*g[i];
	d0=gg/((float)args->np*r0*r0);
	bmin=sqrt(d0);
	
	amin=fmin(r0,1.);
	
	if (bmin>args->alphas[0]/amin) {
		omicron=bmin;
		args->Bcode=false;
	} else {
		omicron=args->alphas[0]/amin;
		args->Bcode=true;
	}
	
	//printf("omicron  = %g\n",omicron);
	
	return omicron;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  FindGamma
// 
// Purpose:   Finds the number of well-fit parameters from the prior
//	      regularisation constant and the trace of the inverse Hessian.
//-----------------------------------------------------------------------------
void FindGamma(NN_args *args, float *alphas, float *gamma, int *dotrials, float *ratio, int myid, bool doprint, int useSD, float mu, long *seed, int verbose) {
	int ntrial,i,j;
	float stddev,temp;
	
	/*unsigned int iseed;
	if( fixseed )
		iseed = fixedseed;
	else
		iseed = (unsigned int)clock();
	srand(iseed);
	long seed = rand();
	seed += 123321;
	seed *= -1;*/
	
	ntrial=3;
	
	if (!(args->prior)) *dotrials=-1;
	if (*dotrials!=-1 && ntrial>=args->np) *dotrials=2;
	
	float gauss[args->np], Mock[args->np], trAinv[args->nalphas];
	
	for (i=0;i<args->nalphas;i++) trAinv[i]=gamma[i]=0.;
	if (*dotrials==1) {
		stddev=0.;
		for (i=0;i<ntrial;i++) {
			if( myid == 0 ) for (j=0;j<args->np;j++) gauss[j]=gasdev(seed);
			//if( doprint && myid == 0 ) printf("gauss[0]=%lf\n",gauss[0]);
#ifdef PARALLEL
			MPI_Barrier(MPI_COMM_WORLD);
    		MPI_Bcast(gauss, args->np, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
			FindDirn(gauss,Mock,args,alphas[0],&temp,1,useSD,mu);
			trAinv[0]+=temp/(float)ntrial;
			stddev+=temp*temp/(float)ntrial;
		}
		stddev-=trAinv[0]*trAinv[0];
		trAinv[0]=fmax(trAinv[0],FLT_EPSILON);
		//if (doprint && myid==0) printf("Tr(A^-1)   = %g +/- %g\n",trAinv[0],sqrt(stddev));
		*ratio+=sqrt(stddev)/trAinv[0];
	}
	
	gamma[0]=trAinv[0];
	
	if (doprint && myid==0 && verbose>2) printf("gamma      = %g +/- %g\n",gamma[0],sqrt(stddev));
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  OmegaVar
// 
// Purpose:   Calculates the std dev of a new Omega esitmate
//-----------------------------------------------------------------------------
float OmegaVar(float *x, NN_args *args, float *gradL, bool *Tcode, bool init) {
	float ovar,test,SS,LL,SL,*gradS,utol=utolg;
	int i;
	
	if (init) {
		ovar=utol*utol/12.;
		*Tcode = true;
	} else {
		gradS=(float *)malloc(args->np*sizeof(float));
		
		args->nn->grad_prior(args->alphas,gradS);
		
		SS=LL=SL=0.;
		for (i=0;i<args->np;i++) {
			SS += gradS[i]*gradS[i];
			LL += gradL[i]*gradL[i];
			SL += gradS[i]*gradL[i];
		}
		
		test=1.+SL/fmax(sqrt(SS*LL),FLT_EPSILON);
		//printf("Test = %g\n",test);
		
		if (test<=1.)
			*Tcode=true;
		else
			*Tcode=false;
		
		ovar=test/fmax(1.-test,FLT_EPSILON)+utol*utol/12.;
	}
	
	return ovar;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  WriteToTable
// 
// Purpose:   Writes to the table of Omega(alpha) values.
//-----------------------------------------------------------------------------
void WriteToTable(AlpOmTable *Save, float alpha, float Omega, float var) {
	int i,J;
	float xnew,ynew,S,X,D;
	
	//printf("%g\t%g\t%g\n",alpha,Omega,var);
	xnew=log(alpha);
	ynew=log(Omega);
	
	// Check if the value is in the table already, update and return if it is
	for (i=0;i<Save->Ntable;i++) {
		if (xnew == Save->X[i]) {
			//if (doprint) printf("Updating table entry... var = %g -> ",Save->V[i]);
			S=var/(var+Save->V[i]);
			X=Save->V[i]/(var+Save->V[i]);
			Save->Y[i]=ynew*X+Save->Y[i]*S;
			Save->V[i]=var*X;
			//if (doprint) printf("%g\n",Save->V[i]);
			return;
		}
	}
	
	// If the table is full, eliminate the worst entry, return if new one is worst
	if (Save->Ntable >= Save->Nsize) {
		J=-1;
		S=var;
		for (i=0;i<Save->Ntable;i++) {
			X=xnew-Save->X[i];
			D=Save->V[i]+pow(X,4.);
			if (D>S) {
				S=D;
				J=i;
			}
		}
		if (J==-1) return;
		Save->Ntable--;
		for (i=J;i<Save->Ntable;i++) {
			Save->X[i]=Save->X[i+1];
			Save->Y[i]=Save->Y[i+1];
			Save->V[i]=Save->V[i+1];
		}
		//if (doprint) printf("Removed an entry from the table, Ntable = %d.\n",Save->Ntable);
	}
	
	// Write the new entry to the table
	J=0;
	for (i=0;i<Save->Ntable;i++)
		if (xnew>Save->X[i]) J=i+1;
	for (i=Save->Ntable-1;i>=J;i--) {
		Save->X[i+1]=Save->X[i];
		Save->Y[i+1]=Save->Y[i];
		Save->V[i+1]=Save->V[i];
	}
	Save->X[J]=xnew;
	Save->Y[J]=ynew;
	Save->V[J]=var;
	Save->Ntable++;
	//if (doprint) printf("Added new entry to the table, Ntable = %d.\n",Save->Ntable);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  ReadFromTable
// 
// Purpose:   Reads from the table of Omega(alpha) values.
//-----------------------------------------------------------------------------
void ReadFromTable(AlpOmTable *Save, float alpha, float *yval, float *sigma) {
	int i;
	float w,wt,wx,wxx,wy,wxy,X,Y,Xbar,Ybar;
	
	w=wt=wx=wxx=wy=wxy=0.;
	
	if (Save->Ntable==1) {
		*yval=Save->Y[0];
		*sigma=0.;
	} else {
		for (i=0;i<Save->Ntable;i++) {
			X=log(alpha)-Save->X[i];
			wt=1./(Save->V[i]+pow(X,4.));
			w+=wt;
			wx+=wt*X;
			wy+=wt*Save->Y[i];
		}
		Xbar=wx/w;
		Ybar=wy/w;
		for (i=0;i<Save->Ntable;i++) {
			X=log(alpha)-Save->X[i];
			wt=1./(Save->V[i]+pow(X,4.));
			X-=Xbar;
			Y=Save->Y[i]-Ybar;
			wxx+=wt*X*X;
			wxy+=wt*X*Y;
		}
		*yval=Ybar-Xbar*wxy/wxx;
		*sigma=1./sqrt(w);
	}
	//if (doprint) printf("Read %g\t%g\t%g\n",alpha,*yval,*sigma);
}

extern "C" void PrintIntro(bool classnet, bool resuming, bool prior, NeuralNetwork *nn, int tdata, int vdata, float frac, int tndstore,
		bool noise, bool wnoise, bool histmaxent, bool recurrent, int whitenin, int whitenout, int stopf) {
	printf("**********************************************************\n");
	printf(" SkyNet v1.2\n");
	printf(" Copyright F Feroz, P Graff, M P Hobson, A N Lasenby\n");
	printf(" Release January 2014\n");
	printf("----------------------------------------------------------\n");
	
	if (classnet) printf(" Setting up a classification network.\n");
	else printf(" Setting up a regression network.\n");
	
	if (prior) printf(" Optimizing over the log-posterior.\n");
	else printf(" Optimizing over the log-likelihood.\n");
	
	if (stopf==1) printf(" Converging using the correlation.\n");
	else if (stopf==2) printf(" Converging using the log-likelihood.\n");
	else if (stopf==3) printf(" Converging using the log-posterior.\n");
	else if (stopf==4) printf(" Converging using the error squared.\n");
	
	if (histmaxent) printf(" Using historic maximum entropy Omega=Ndata/(-2*Like)\n");
	else printf(" Using classic maximum entropy Omega=Good/(-2*Prior)\n");
	
	if (!classnet) {
		if (wnoise) printf(" Setting output noise on whitened data values.\n");
		else printf(" Setting output noise on original data values.\n");
		if (noise) printf("\t==> Using noise re-scaling.\n");
	}
	
	if (resuming) printf(" Resuming optimization from a previous saved state.\n");
	else printf(" Starting optimization from a random state.\n");
	
	if (nn->nlayers>=3) {
		printf(" %d inputs, ",(int)nn->nnodes[0]-1);
		printf("%d",(int)nn->nnodes[1]-1);
		for (int i=2;i<nn->nlayers-1;i++) printf("+%d",(int)nn->nnodes[i]-1);
		printf(" hidden nodes, %d outputs\n\t==> %d total weights.\n",(int)nn->nnodes[nn->nlayers-1],(int)nn->nparams());
	} else printf(" %d inputs, %d outputs\n\t==> %d total weights.\n",(int)nn->nnodes[0],(int)nn->nnodes[nn->nlayers-1],(int)nn->nparams());
	
	printf(" Read %d training and %d validation data points.\n",tndstore,vdata);
	printf(" Using %3.1lf%% of the training data.",100.*frac);
	
	if (frac<1) printf("\n\t==> %d points in the mini-batch.\n",tdata);
	else printf("\n");
	
	if (whitenin>0) printf(" Whitening the inputs before training.\n");
	if (whitenout>0) printf(" Whitening the outputs before training.\n");
	
	printf("**********************************************************\n\n");
}

void GetPredictions(NN_args *args, NN_args *refargs, int i, float omicron, bool classnet, std::vector <float> &predout, std::vector <float> &sigma, bool geterror)
{
	args->nn->pflag = false;
	args->nn->forward(*(args->td), *(args->pd), i, true);
	int q = 0;
	for (int h=args->td->ntimeignore[i];h<args->td->ntime[i];h++) {
		int k = args->td->cuminsets[i]*args->pd->totnnodes+(h+1)*args->pd->totnnodes-args->pd->nout;
		for( int j = 0; j < args->td->nout; j++ ) predout.push_back(args->pd->out[k+j]);
		if(!classnet && geterror) {
			float e[(args->td->ntime[i]-args->td->ntimeignore[i])*args->td->nout];
			for( int p = 0; p < (args->td->ntime[i]-args->td->ntimeignore[i])*args->td->nout; p++ ) e[p] = 0.0;
		
			for( int j = 0; j < args->td->nout; j++ ) {
				e[q]=1.0;
				q++;
				float g[args->nn->nweights], dummy[args->nn->nweights], x;
				if(!classnet)
				{
					args->nn->backward(e, *(args->td), *(args->pd), i, false, g);
					FindDirn(g, dummy, refargs, omicron, &x, 3, 0, 0.0);
					//sigma.push_back(sqrt(pow(1.0/args->wtd->acc[args->wtd->cumoutsets[i]*args->wtd->nout+h*args->wtd->nout+j],2.0)+x));
					sigma.push_back(sqrt(x));
				}
			}
		}
	}
	args->nn->pflag = true;
}

void PrintPredictions(std::string filename, int whitenout, int whitenin, whiteTransform *itrans, whiteTransform *otrans, NN_args *args, NN_args *refargs, float omicron,
		      bool classnet, bool printerror, bool autoencoder, int verbose) {
	int i,j,h;
	size_t k;
	FILE *of=fopen(filename.c_str(),"w");
	//fprintf(of,"%d\t%d\n",(int)td.nin,(int)td.nout);
	std::vector <float> predin, predout, actualout, sigma;
	
	if(whitenout==2) printerror=false;
	
	for( i = 0; i < args->td->ndata; i++ ) {
		std::vector <float> predout1, sigma1;
		GetPredictions(args, refargs, i, omicron, classnet, predout1, sigma1, printerror);
		for(h=0;h<predout1.size();h++) {
			predout.push_back(predout1[h]);
			if(!classnet && printerror) sigma.push_back(sigma1[h]);
		}
		for (h=args->td->ntimeignore[i];h<args->td->ntime[i];h++) {
			k = args->td->cuminsets[i]*args->pd->totnnodes+(h+1)*args->pd->totnnodes-args->pd->nout;
			for( j = 0; j < args->td->nin; j++ ) predin.push_back(args->td->inputs[args->td->cuminsets[i]*args->td->nin+h*args->td->nin+j]);
			for( j = 0; j < args->td->nout; j++ ) actualout.push_back(args->td->outputs[(args->td->cumoutsets[i]+h-args->td->ntimeignore[i])*args->td->nout+j]);
			//if(whitenout!=3 && !classnet && printerror) otrans->applyScalingOnly(args->td->inputs,sigma[i],sigma[i],args->td->nout);
		}
	}
	if (whitenin==1) itrans->inverse(&predin[0],&predin[0],args->td->ndata*args->td->nin);
	if (whitenout==1) {
		otrans->inverse(&actualout[0],&actualout[0],args->td->ndata*args->td->nout);
		otrans->inverse(&predout[0],&predout[0],args->td->ndata*args->td->nout);
	}
	
			
	float factor[args->td->nout];
	/*for( i = 0; i < args->wtd->nout; i++ ) {
		std::vector <float> errors;
		for( j = i; j < sigma.size(); j+=args->wtd->nout ) errors.push_back(sigma[j]);
		std::sort(errors.begin(),errors.end());
		float e=errors[errors.size()*0.683];
		
		float avge=0.0, avgs=0.0;
		for( j = i; j < sigma.size(); j+=args->wtd->nout ) {
			if(sigma[j]<=e) {
				avgs += sigma[j];
				avge += fabs(actualout[j]-predout[j]);
			}
		}
		
		factor[i] = avge/avgs;
	}*/
	
	float avgsqerr = 0;
	for( i = 0; i < actualout.size()/args->td->nout; i++ ) {
		for( j = 0; j < args->td->nin; j++ ) fprintf(of,"%lf\t",predin[i*args->td->nin+j]);
		if (!autoencoder) 
		  for( j = 0; j < args->td->nout; j++ )
		    fprintf(of,"%lf\t",actualout[i*args->td->nout+j]);
		for( j = 0; j < args->td->nout; j++ ) {
			fprintf(of,"%lf\t",predout[i*args->td->nout+j]);
			avgsqerr += pow(actualout[i*args->td->nout+j]-predout[i*args->td->nout+j], 2.0)/float(actualout.size());
		}
		//if(!classnet && printerror) for( j = 0; j < args->wtd->nout; j++ ) fprintf(of,"%lf\t",sigma[i*args->wtd->nout+j]*factor[j]);
		if(!classnet && printerror) for( j = 0; j < args->td->nout; j++ ) fprintf(of,"%lf\t",sigma[i*args->td->nout+j]);
		fprintf(of,"\n");
	}
	fclose(of);
	
	if (verbose>0) printf("average squared error = %f\n",avgsqerr);
}

void ReadInputFile1(char filename[], char inroot[], char outroot[], bool *resume) {
	char line[100],value[100];
	int temp;
	
	FILE *infile=fopen(filename,"r");
	
	if (infile==NULL) {
		fprintf(stderr,"No such input file exists!\n");
		exit(-1);
	}
	
	while (!feof(infile)) {
		fgets(line,100,infile);
		fgets(value,100,infile);
		
		if (strcmp(line,"#input_root\n")==0) {
			strcpy(inroot,value);
			inroot[strlen(inroot)-1]='\0';
		} else if (strcmp(line,"#output_root\n")==0) {
			strcpy(outroot,value);
			outroot[strlen(outroot)-1]='\0';
		} else if (strcmp(line,"#resume\n")==0) {
			temp=atoi(value);
			if (temp==0) *resume=false;
			else *resume=true;
		}
	}
	
	fclose(infile);
}

void ReadInputFile2(char filename[], std::vector <size_t> &nhid, bool *classnet, float *frac, bool *prior,
		   bool *noise, bool *wnoise, float *sigma, float *rate, int *printfreq, bool *fixseed, int *fixedseed, bool *evidence,
		   bool *histmaxent, bool *recurrent, int *whitenin, int *whitenout, int *stopf, bool *hhps, int *hhpl, int *maxniter, int *nin, int *nout, 
		   bool *norbias, int *useSD, bool *text, bool *vdata, char linearlayers[], bool *resetalpha, bool *resetsigma, bool *autoencoder, bool *pretrain,
		   int *nepoch, bool *indep, float *ratemin, float *logLRange, float *randweights, int *verbose, bool
		   *readacc, float *stdev, int *lnsrch) {
	char line[100],value[100];
	int temp;
	
	FILE *infile=fopen(filename,"r");
	
	if (infile==NULL) {
		fprintf(stderr,"No such input file exists!\n");
		exit(-1);
	}
	
	while (!feof(infile)) {
		fgets(line,100,infile);
		fgets(value,100,infile);

		if (strcmp(line,"#nhid\n")==0) {
			size_t nh = (size_t) atoi(value);
			if( nh > 0 ) nhid.push_back(nh);
		} else if (strcmp(line,"#classification_network\n")==0) {
			temp=atoi(value);
			if (temp==0) *classnet=false;
			else *classnet=true;
		} else if (strcmp(line,"#mini-batch_fraction\n")==0) {
			*frac=(float)atof(value);
		} else if (strcmp(line,"#prior\n")==0) {
			temp=atoi(value);
			if (temp==0) *prior=false;
			else *prior=true;
		} else if (strcmp(line,"#noise_scaling\n")==0) {
			temp=atoi(value);
			if (temp==0) *noise=false;
			else *noise=true;
		} else if (strcmp(line,"#set_whitened_noise\n")==0) {
			temp=atoi(value);
			if (temp==0) *wnoise=false;
			else *wnoise=true;
		} else if (strcmp(line,"#sigma\n")==0) {
			*sigma=(float)atof(value);
		} else if (strcmp(line,"#confidence_rate\n")==0) {
			*rate=(float)atof(value);
		} else if (strcmp(line,"#confidence_rate_minimum\n")==0) {
			*ratemin=(float)atof(value);
		} else if (strcmp(line,"#iteration_print_frequency\n")==0) {
			*printfreq=atoi(value);
		} else if (strcmp(line,"#startstdev\n")==0) {
			*stdev=atof(value);
		} else if (strcmp(line,"#fix_seed\n")==0) {
			temp=atoi(value);
			if (temp==0) *fixseed=false;
			else *fixseed=true;
		} else if (strcmp(line,"#fixed_seed\n")==0) {
			*fixedseed=atoi(value);
		} else if (strcmp(line,"#calculate_evidence\n")==0) {
			temp=atoi(value);
			if (temp==0) *evidence=false;
			else *evidence=true;
		} else if (strcmp(line,"#historic_maxent\n")==0) {
			temp=atoi(value);
			if (temp==0) *histmaxent=false;
			else *histmaxent=true;
		} else if (strcmp(line,"#recurrent\n")==0) {
			temp=atoi(value);
			if (temp==0) *recurrent=false;
			else *recurrent=true;
		} else if (strcmp(line,"#norbias\n")==0) {
			temp=atoi(value);
			if (temp==0) *norbias=false;
			else *norbias=true;
		} else if (strcmp(line,"#textual_data\n")==0) {
			temp=atoi(value);
			if (temp==0) *text=false;
			else *text=true;
		} else if (strcmp(line,"#validation_data\n")==0) {
			temp=atoi(value);
			if (temp==0) *vdata=false;
			else *vdata=true;
		} else if (strcmp(line,"#use_structural_damping\n")==0) {
			temp=atoi(value);
			if (temp==0) *useSD=0;
			else *useSD=1;
		} else if (strcmp(line,"#whitenin\n")==0) {
			*whitenin=atoi(value);
			if(*whitenin<0) *whitenin=0;
		} else if (strcmp(line,"#whitenout\n")==0) {
			*whitenout=atoi(value);
			if(*whitenout<0) *whitenout=0;
		} else if (strcmp(line,"#HHP_likelihood\n")==0) {
			*hhpl=atoi(value);
		} else if (strcmp(line,"#HHP_score\n")==0) {
			temp=atoi(value);
			if (temp==0) *hhps=false;
			else *hhps=true;
		} else if (strcmp(line,"#max_iter\n")==0) {
			temp=atoi(value);
			if (temp>=0) *maxniter=temp;
		} else if (strcmp(line,"#nin\n")==0) {
			temp=atoi(value);
			if (temp>0) *nin=temp;
		} else if (strcmp(line,"#nout\n")==0) {
			temp=atoi(value);
			if (temp>0) *nout=temp;
		} else if (strcmp(line,"#convergence_function\n")==0) {
			*stopf=atoi(value);
			if(*stopf<1 || *stopf>4) *stopf=4;
		} else if (strcmp(line,"#activation\n")==0) {
			strcpy(linearlayers,value);
			linearlayers[strlen(linearlayers)-1]='\0';
		} else if (strcmp(line,"#reset_alpha\n")==0) {
			temp=atoi(value);
			if (temp==0) *resetalpha=false;
			else *resetalpha=true;
		} else if (strcmp(line,"#reset_sigma\n")==0) {
			temp=atoi(value);
			if (temp==0) *resetsigma=false;
			else *resetsigma=true;
		} else if (strcmp(line,"#autoencoder\n")==0) {
			temp=atoi(value);
			if (temp==0) *autoencoder=false;
			else *autoencoder=true;
		} else if (strcmp(line,"#pretrain\n")==0) {
			temp=atoi(value);
			if (temp==0) *pretrain=false;
			else *pretrain=true;
		} else if (strcmp(line,"#nepoch\n")==0) {
			*nepoch=atoi(value);
		} else if (strcmp(line,"#data_independent\n")==0) {
			temp=atoi(value);
			if (temp==0) *indep=false;
			else *indep=true;
		} else if (strcmp(line,"#logL_range\n")==0) {
			*logLRange=(float)atof(value);
		} else if (strcmp(line,"#randomise_weights\n")==0) {
			*randweights=(float)fabs(atof(value));
		} else if (strcmp(line,"#verbose\n")==0) {
			*verbose=abs(atoi(value));
		} else if (strcmp(line,"#read_acc\n")==0) {
			*readacc=atoi(value)==0?false:true;
		} else if (strcmp(line,"#line_search\n")==0) {
			*lnsrch=atoi(value);
			if (*lnsrch<0 || *lnsrch>2) *lnsrch=0;
		} else if (strcmp(line,"#input_root\n")==0) {
			// do nothing
		} else if (strcmp(line,"#output_root\n")==0) {
			// do nothing
		} else if (strcmp(line,"#resume\n")==0) {
			// do nothing
		} else if (strcmp(line,"#discard_points\n")==0) {
			// do nothing
		} else if (strcmp(line,"#logL_range\n")==0) {
			// do nothing
		} else if (strlen(line)>0) {
			line[strlen(line)-1]='\0';
			fprintf(stderr,"Warning:  %s is not a recognized option\n",line);
		}
		line[0] = value[0] = '\0';
	}
	
	fclose(infile);
}

void ReadInputFile3(char filename[], bool *discardpts) {
	char line[100],value[100];
	int temp;
	
	FILE *infile=fopen(filename,"r");
	
	if (infile==NULL) {
		fprintf(stderr,"No such input file exists!\n");
		exit(-1);
	}
	
	while (!feof(infile)) {
		fgets(line,100,infile);
		fgets(value,100,infile);
		
		if (strcmp(line,"#discard_points\n")==0) {
			temp=atoi(value);
			if (temp==0) *discardpts=false;
			else *discardpts=true;
		}
	}
	
	fclose(infile);
}

void ReadInputFile4(char filename[], bool *noise, bool *wnoise, float *sigma, float *logLRange) {
	char line[100],value[100];
	int temp;
	
	FILE *infile=fopen(filename,"r");
	
	if (infile==NULL) {
		fprintf(stderr,"No such input file exists!\n");
		exit(-1);
	}
	
	while (!feof(infile)) {
		fgets(line,100,infile);
		fgets(value,100,infile);
		
		if (strcmp(line,"#noise_scaling\n")==0) {
			temp=atoi(value);
			if (temp==0) *noise=false;
			else *noise=true;
		} else if (strcmp(line,"#set_whitened_noise\n")==0) {
			temp=atoi(value);
			if (temp==0) *wnoise=false;
			else *wnoise=true;
		} else if (strcmp(line,"#sigma\n")==0) {
			*sigma=(float)atof(value);
		} else if (strcmp(line,"#logL_range\n")==0) {
			*logLRange=(float)atof(value);
		}
	}
	
	fclose(infile);
}

void AddNewTrainData(char *root, int ndata, int ndim, float **data, float trainfrac, bool *trainflag, int verbose) {
	char trainfile[100],testfile[100];
	FILE *train,*test;
	int i,j,c1=0,c2=0;
	
	strcpy(trainfile,root);
	strcpy(testfile,root);
	strcat(trainfile,"train.txt");
	strcat(testfile,"test.txt");
	
	train=fopen(trainfile,"w");
	test=fopen(testfile,"w");
	fprintf(train,"%d,\n%d,\n",ndim,1);
	fprintf(test,"%d,\n%d,\n",ndim,1);
	
	unsigned int iseed;
	iseed = (unsigned int)time(NULL);
	srand(iseed);
	long seed = rand();
	seed += 123321;
	seed *= -1;
	
	for (i=0;i<ndata;i++) {
		if (ran2(&seed)<trainfrac) {
			for (j=0;j<ndim;j++) fprintf(train,"%lf,",data[i][j]);
			fprintf(train,"\n%lf,\n",data[i][ndim]);
			trainflag[i]=true;
			c1++;
		} else {
			for (j=0;j<ndim;j++) fprintf(test,"%lf,",data[i][j]);
			fprintf(test,"\n%lf,\n",data[i][ndim]);
			trainflag[i]=false;
			c2++;
		}
	}
	
	if (verbose>0) printf("%d printed to train, %d printed to test.\n",c1,c2);
	
	fclose(train);
	fclose(test);
}

void PrintVersion() {
	char verstr[]="\n\
SkyNet v1.2\n\
Copyright F Feroz, P Graff, M P Hobson, A N Lasenby\n\
Release May 2014\n\n";
	printf("%s",verstr);
}

void PrintHelp() {
	char helpstr[]="\
--------------------------------------------------------------------------\n\
SkyNet v1.2\n\
Copyright F Feroz, P Graff, M P Hobson, A N Lasenby\n\
Release May 2014\n\
--------------------------------------------------------------------------\n\
\n\
An input file should be formatted with the following pairs of lines repeated:\n\
#[option_name]\n\
[option_value]\n\
Order does not matter except for specifying the hidden layers. Yes/no options have 0=no, 1=yes.\n\n\
--------------------------------------------------------------------------\n\
    Data-Handling options\n\
--------------------------------------------------------------------------\n\
input_root                  root of the data files\n\
classification_network      0=regression, 1=classification\n\
mini-batch_fraction         what fraction of training data to be used?\n\
validation_data             is there validation data to test against?\n\
whitenin                    input whitening transform to use\n\
whitenout                   output whitening transform to use\n\
\n\
--------------------------------------------------------------------------\n\
    Network and Training options\n\
--------------------------------------------------------------------------\n\
nhid                        no. of nodes in the hidden layer. For multiple hidden layers,\n\
                            define nhid multiple times with the no. of nodes required in\n\
                            each hidden layer in order.\n\
activation                  manually set activation function of layer connections\n\
                            options are: 0=linear, 1=sigmoid, 2=tanh,\n\
                                         3=rectified linear, 4=softsign\n\
                            default is 1 for all hidden and 0 for output\n\
			    e.g. for a network with 3 layers (input, hidden & output), 10 would\n\
			    set sigmoid & linear activation for hidden & output layers respectively\n\
prior                       use prior/regularization\n\
noise_scaling               if noise level (standard deviation of outputs) is to be estimated\n\
set_whitened_noise          whether the noise is to be set on whitened data\n\
sigma                       initial noise level, set on (un-)whitened data\n\
confidence_rate             step size factor, higher values are more aggressive. default=0.1\n\
confidence_rate_minimum     minimum confidence rate allowed\n\
max_iter                    max no. of iterations allowed\n\
startstdev                  the standard deviation of the initial random weights\n\
convergence_function        function to use for convergence testing, default is 4=error squared\n\
                            1=log-posterior, 2=log-likelihood, 3=correlation\n\
historic_maxent             experimental implementation of MemSys's historic maxent option\n\
resume                      resume from a previous job\n\
reset_alpha                 reset hyperparameter upon resume\n\
reset_sigma                 reset hyperparameters upon resume\n\
randomise_weights           random factor to add to saved weights upon resume\n\
line_search					perform line search for optimal distance\n\
                            0 = none (default), 1 = golden section, 2 = linbcg lnsrch\n\
\n\
--------------------------------------------------------------------------\n\
    Output options\n\
--------------------------------------------------------------------------\n\
output_root                 root where the resultant network will be written to\n\
verbose                     verbosity level of feedback sent to stdout (0=min, 3=max)\n\
iteration_print_frequency   stdout feedback frequency\n\
calculate_evidence          whether to calculate the evidence at the convergence\n\
\n\
--------------------------------------------------------------------------\n\
    Autoencoder options\n\
--------------------------------------------------------------------------\n\
pretrain                    perform pre-training?\n\
nepoch                      number of epochs to use in pre-training (default=10)\n\
autoencoder                 make autoencoder network\n\
\n\
--------------------------------------------------------------------------\n\
    RNN options\n\
--------------------------------------------------------------------------\n\
recurrent                   use a RNN\n\
norbias                     use a bias for the recurrent hidden layer connections\n\
\n\
--------------------------------------------------------------------------\n\
    Debug options\n\
--------------------------------------------------------------------------\n\
fix_seed                    use a fixed seed?\n\
fixed_seed                  seed to use\n\
\n";
	printf("%s",helpstr);
}
