//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Filename:  fnmatrix.h
// 
// Purpose:   Header for fnmatrix.c
// 
// History:   John Skilling, Kenmare, Ireland, 1993-2010
//            email: skilling@eircom.net
//            15 Dec 1993  for Maximum Entropy Data Consultants
//            04 Apr 2000  minor polish
//            20 May 2006  modernised and re-packaged
//            01 Nov 2008  extensive polishing
//            18 Nov 2010  fnMatRepeat added; Vector independent of Scalar
//-----------------------------------------------------------------------------
#ifndef FNMATRIX
#define FNMATRIX

/**************/
/* Definition */
/**************/

typedef struct           // SYSTEM
{
    int        Ndim;     // dimension of real-space vectors to pass through
    int        Nmax;     // max # transforms                     n = (Nmax+2)/2
    int        ngam;     // # computed search directions   = (ntrans+2)/2
    int        ndel;     // # quantified search directions = (ntrans+1)/2
    float*    gam;      // subspace gradients                              [n]
    float*    del;      // subspace conjugates                             [n]
    float*    work;     // workspace                                       [n]
    float*    LoCoeff;  // Lo-subspace coefficients                        [n]
    float*    HiCoeff;  // Hi-subspace coefficients                        [n]
    float*    LoEvals;  // Lo-subspace eigenvalues                         [n]
    float*    HiEvals;  // Hi-subspace eigenvalues                         [n]
    float*    LoSpins;  // #pairs; (right,left) list             [1+3*n*(n+1)]
    float*    HiSpins;  // #pairs; (right,left) list             [1+3*n*(n+1)]
    float*    h;        // conjugate vector                             [Ndim]
    float*    u;        // working vector                               [Ndim]
    float**   g;        // gradient vectors                          [n][Ndim]
                         // (can be overlaid if Vector calls not to be made)
    float     random;   // random number for fnMatCheck only
} sfnMatrix;

/**************/
/* Prototypes */
/**************/

// Supplied in fnmatrix.c =====================================================
int    fnMatNew(      //   O  0=OK, -ve = memory allocation error
sfnMatrix* psfnMat,   //   O  fnMat structure, ready for use
int        Ndim,      // I    vector dimension
int        Nmax);     // I    max # transforms >= 1

int    fnMatPerform(  //   O  # transforms performed, -ve = error
sfnMatrix* psfnMat,   //   O  conjugate-gradient results
void*      A,         // I    transform 
float*    b,         // I    source vector                              [Ndim]
float     alpha,     // I    regularisation coefficient
float     utol);     // I    fractional tolerance

int    fnMatRepeat(   //   O  # transforms performed, -ve = error
sfnMatrix* psfnMat,   //   O  conjugate-gradient results
void*      A,         // I    transform 
float*    b);        // I    source vector                              [Ndim]

float fnMatLoScalar( //   O  lower bound to scalar  b'.f(A).b
sfnMatrix* psfnMat,   // I O  search direction coefficients
void*      A,         // I    transform 
int        flag);     // I    function identifier

void   fnMatLoVector(
sfnMatrix* psfnMat,   // I    search direction coefficients
void*      A,         // I    transform 
float*    v,         //   O  result f(A).b                              [Ndim]
int        flag);     // I    function identifier

float fnMatHiScalar( //   O  upper bound to scalar  b'.f(A).b
sfnMatrix* psfnMat,   // I O  search direction coefficients
void*      A,         // I    transform 
int        flag);     // I    function identifier

void   fnMatHiVector(
sfnMatrix* psfnMat,   // I    search direction coefficients
void*      A,         // I    transform 
float*    w,         //   O  result  F(A).b                             [Ndim]
int        flag);     // I    function identifier

float fnMatCheck(    //   O  dimensionless opus/tropus inconsistency
sfnMatrix* psfnMat,   //  (O) work-space
void*      A,         // I    transform 
int        Ndata,     // I    opus(=SCALAR) operates as a Ndata-by-Ndim matrix
float*    F);        // I    destination in A of SCALAR(...,h)         [Ndata]

void   fnMatOld(      //      free memory
sfnMatrix* psfnMat);  //  (O) fnMat structure

#if 0                 /* Alternative codes */
float invMatLoScalar(//   O  lower bound to scalar  b'.B^(-1).b
sfnMatrix* psfnMat,   // I    search direction coefficients
float     alpha);    // I    regularisation coefficient in B = alpha*I + A

void   invMatLoVector(
sfnMatrix* psfnMat,   // I    search direction coefficients
float     alpha,     // I    regularisation coefficient in B = alpha*I + A
float*    v);        //   O  vector B^(-1).b

float invMatHiScalar(//   O  upper bound to scalar  b'.B^(-1).b
sfnMatrix* psfnMat,   // I    search direction coefficients
float     alpha);    // I    regularisation coefficient in B = alpha*I + A

void   invMatHiVector(
sfnMatrix* psfnMat,   // I    search direction coefficients
float     alpha,     // I    regularisation coefficient in B = alpha*I + A
float*    w);        //   O  vector B^(-1).b / alpha
#endif

// To be supplied elsewhere by user ===========================================
int    SCALAR(        //      transform scalar
float*    delta,     //   O  v'.A.v
void*      A,         // I(O) & transform (prepared for VECTOR)
float*    v);        // I    input vector                               [Ndim]

int    VECTOR(        //      transform vector
float*    u,         //   O  A.v                                        [Ndim]
void*      A);        // I    & transform, prepared by SCALAR

float fnMat(         //   O  function(scalar) defining f(A)
void*      A,         // I    & transform    
float     x,         // I    scalar argument
int        flag);     // I    function identifier                

/**************************************/
/* Return codes (+ve=info, -ve=error) */
/**************************************/

#define E_MALLOC       -130  // Unable to allocate memory                   

#endif
