#ifndef __NNOPT_H__
#define __NNOPT_H__ 1

#ifdef PARALLEL
#include <mpi.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <time.h>
#include "NeuralNetwork.h"
#include "fnmatrix.h"
#include "myrand.h"

#define CALL(x)    {if( (CALLvalue = (x)) < 0 ) goto Exit;}
#define TINY 1.e-20
#define HFMAX 5
#define FREEALLVARS free(x);free(alphas);free(gamma);free(beta);free(best);free(Save.X);free(Save.Y);free(Save.V);

typedef enum {
	silent,
	minimal,
	regular,
	everything
} VerboseLevel;

static float golden = (1.0+sqrt(5.0))/2.0;
static float resgold = 2.0-golden;

/*static bool fixseed=false;
static int fixedseed=1234;*/
static float utolg=0.1;

std::string concatenate(std::string in1, std::string in2);

std::string addnum(std::string in, int num);

typedef axisAlignedTransform trans_t;

typedef struct {
	int np;
	TrainingData *td;
	PredictedData *pd;
	NeuralNetwork *nn;
	float *alphas;
	float omicron;
	bool prior;
	bool noise;
	int nalphas;
	bool Bcode;
	int lnsrch;
} NN_args;

typedef struct          // USER-IMPLEMENTATION of transform
{
    int       Npar;     // dimension of parameter space
    float*   Mock;     // prepared by SCALAR for use in VECTOR
    float    alpha;    // regularisation coefficient
    int	      useSD;	// use structural damping
    float    mu;	// structual damping coefficient
    NN_args*  args;
} sMATRIX;

typedef struct {
	float *X,*Y,*V;
	int Nsize,Ntable;
} AlpOmTable;

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  HessianFreeOpt
// 
// Purpose:   Uses the Hessian-free optimization method, implementing the
//            conjugate gradient from John Skilling (MemSys).
//-----------------------------------------------------------------------------
void HessianFreeOpt(float *x, 		// network weights (position in parameter space)
		    NN_args *args, 	// struct with network and data pointers
		    float *lnew, 	// passes in starting value, passes out optimal value
		    float r0,		// rate
		    int useSD,		// use structural damping
	            float mu);		// structural damping coefficient

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  FindDirn
// 
// Purpose:   Perform the conjugate gradient optimization for the direction to
//            move in next.
//-----------------------------------------------------------------------------
int FindDirn(float *b, 	// input vector, gradient for dirn, gaussian for trace or determinant
	     float *p, 	// output direction
	     NN_args *args, 	// struct with network and data pointers
	     float alpha, 	// regularisation constant
	     float *scalar, 	// output scalar for trace/determinant
	     int what,		// flag to determine what to do, 0=dirn, 1=trace, 2=determinant
	     int useSD,		// use structural damping
	     float mu);		// structural damping coefficient

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  AdjustAlpha
// 
// Purpose:   Adjusts the regularization constant(s) based on Equation 2.22
//            of David Mackay's thesis. Then adjusts the betas (noise scaling)
//            based on Equation 2.24 of David Mackay's thesis.
//-----------------------------------------------------------------------------
bool AdjustAlpha(float* x, 		// network weights (position in parameter space)
		 float *alphas, 	// prior regularisation constant(s)
		 float *gamma, 	// array of gammas (# fitted parameters)
		 NN_args *args, 	// struct with network and data pointers
		 bool classnet, 	// is classification network or not
		 float *ratio, 	// for use when determining trace with random vectors
		 int *dotrials, 	// how to calculate trace
		 int niter,		// iteration number
		 float rz,		// rate
		 float *Omega,
		 float *Ovar,
		 AlpOmTable *Save,
		 whiteTransform *otrans,// output whiten transform
		 int myid,		// MPI ID
		 bool doprint,
		 float aim,
		 bool histmaxent,
		 int whitenin,
		 int whitenout,
		 int useSD,
		 float mu,
	         long *seed,		// RNG seed value
		 int verbose);		// verbose level

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  GetlogZ
// 
// Purpose:   Calculates the log-evidence based on Equation 2.20 of David
//            Mackay's thesis. Many evaluations are done for different values
//            of log(|A|) returned by using different random vectors. Returns
//            the mean and prints the mean and standard deviation.
//-----------------------------------------------------------------------------
float GetlogZ(float *x, 		// network weights (position in parameter space)
	       NN_args *args, 		// struct with network and data pointers
	       float *alphas, 		// prior regularisation constant(s)
	       float *gamma, 		// array of gammas (# fitted parameters)
	       bool classnet,		// is classification network or not
	       int myid,		// MPI ID
	       int useSD,		// use structural damping
	       float mu,		// structural damping coefficient
	       long *seed,		// RNG seed value
	       int verbose);		// verbose level

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  InitialiseAlpha
// 
// Purpose:   Initialises the prior alpha according to the MemSys formula.
//-----------------------------------------------------------------------------
void InitialiseAlpha(float *x,		// network weights (position in parameter space)
		     NN_args *args,	// struct with network and data pointers
		     AlpOmTable *Save,	// Omega(alpha) table
		     float rz,		// rate
		     float *Omega,
		     float *Ovar,
		     int myid,		// MPI ID
		     int verbose);	// verbose level

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  FindOmicron
// 
// Purpose:   Finds the regularisation constant according to the MemSys formula.
//-----------------------------------------------------------------------------
float FindOmicron(float *x,		// network weights (position in parameter space)
		   NN_args *args,	// struct with network and data pointers
		   float r0,		// rate
		   float *g);		// gradient of likelihood to pass out

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  FindGamma
// 
// Purpose:   Finds the number of well-fit parameters from the prior
//	      regularisation constant and the trace of the inverse Hessian.
//-----------------------------------------------------------------------------
void FindGamma(NN_args *args,		// struct with network and data pointers
	       float *alphas,		// prior regularisation
	       float *gamma,		// number of well-fit parameters
	       int *dotrials,		// flag for how to calculate trace
	       float *ratio,		// Tr(A^-1) stddev/mean
	       int myid,		// MPI ID
	       bool doprint,
	       int useSD,		// use structural damping
	       float mu,		// structural damping coefficient
	       long *seed,		// RNG seed value
	       int verbose);		// verbose level

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  OmegaVar
// 
// Purpose:   Calculates the std dev of a new Omega esitmate
//-----------------------------------------------------------------------------
float OmegaVar(float *x, NN_args *args, float *gradL, bool *Tcode, bool init);

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  WriteToTable
// 
// Purpose:   Writes to the table of Omega(alpha) values.
//-----------------------------------------------------------------------------
void WriteToTable(AlpOmTable *Save, float alpha, float Omega, float var);

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  ReadFromTable
// 
// Purpose:   Reads from the table of Omega(alpha) values.
//-----------------------------------------------------------------------------
void ReadFromTable(AlpOmTable *Save, float alpha, float *yval, float *sigma);

float logLike(float* x, void *arg);
void gradLike(float* grad, void *arg);
float logPost(float *x, void *arg);
float logPostMod(float *x, void *arg);

void ErrorExit(std::string msg);

extern "C" void PrintIntro(bool classnet, bool resuming, bool prior, NeuralNetwork *nn, int tdata, int vdata, float frac, int tndstore,
		bool noise, bool wnoise, bool histmaxent, bool recurrent, int whitenin, int whitenout, int stopf);

void PrintHelp();
void PrintVersion();

void GetPredictions(NN_args *args, NN_args *refargs, int i, float omicron, bool classnet, std::vector <float> &predout, std::vector <float> &sigma, bool geterror);
void PrintPredictions(std::string filename, int whitenin, int whitenout, whiteTransform *itrans, whiteTransform *otrans, NN_args *args, NN_args *refargs, float omicron,
		      bool classnet, bool printerror, bool autoencoder, int verbose);

void ReadInputFile1(char filename[], char inroot[], char outroot[], bool *resume);
void ReadInputFile2(char filename[], std::vector <size_t> &nhid, bool *classnet, float *frac, bool *prior,
		   bool *noise, bool *wnoise, float *sigma, float *rate, int *printfreq, bool *fixseed, int *fixedseed, bool *evidence,
		   bool *histmaxent, bool *recurrent, int *whitenin, int *whitenout, int *stopf, bool *hhps, int *hhpl, int *maxniter, int *nin, int *nout, 
		   bool *norbias, int *useSD, bool *text, bool *vdata, char linearlayers[], bool *resetalpha, bool *resetsigma, bool *autoencoder, bool *pretrain,
		   int *nepoch, bool *indep, float *ratemin, float *logLRange, float *randweights, int *verbose, bool
		   *readacc, float *stdev, int *lnsrch);
void ReadInputFile3(char filename[], bool *discardpts);
void ReadInputFile4(char filename[], bool *noise, bool *wnoise, float *sigma, float *logLRange);

double TrainNetwork(char *inputfile, char *inroot, char *outroot, size_t *nlayerspass, size_t *nnodespass, bool resume, bool printerror);

void AddNewTrainData(char *root, int ndata, int ndim, float **data, float trainfrac, bool *trainflag, int verbose);

#endif
