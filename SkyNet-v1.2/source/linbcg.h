#ifndef __LINBCG_H__
#define __LINBCG_H__ 1

void linbcg_solve(float *b, float *x, int &iter, float &err, NN_args *NN, int itmax);

float linbcg_snrm(float *sx, const int n);

void linbcg_atimes(float *x, float *r, const int itrnsp, NN_args *NN, const int n);

void linbcg_asolve(float *b, float *x, const int itrnsp, const int n);

void linbcg_lnsrch(
	float *xold,		// last point
	const float fold,	// fuction value at xold
	float *g,			// gradient at xold
	float *p,			// direction
	float *x,			// new point x = xold + \lambda * p
	float &alam,		// \lambda
	float &f,			// function value at new point x
	const float stpmax,	// max step-length (\lambda), usually 1
	bool &check,		// output, T if x is too close to xold
	NN_args *nn         // the NN
	);

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  goldenSectionSearch
// 
// Purpose:   Recursive golden search function.
//-----------------------------------------------------------------------------
float goldenSectionSearch(NN_args *args, float *x0, float *drn, float fb, float a, float b, float c, float tau);

#endif
