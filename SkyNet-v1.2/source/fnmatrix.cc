//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Filename:  fnmatrix.c
// 
// Purpose:   Function of matrix
// 
// History:   John Skilling, Kenmare, Ireland, 1993-2010
//            email: skilling@eircom.net
//            15 Dec 1993  for Maximum Entropy Data Consultants
//            04 Apr 2000  minor polish
//            20 May 2006  modernised and re-packaged
//            06 Nov 2008  re-worked and polished
//            06 Nov 2010  eigencalculation brought forward
//            18 Nov 2010  fnMatRepeat added; Vector independent of Scalar
//-----------------------------------------------------------------------------
/*
    Copyright (c) 1993-2008, Maximum Entropy Data Consultants Ltd,
                        115c Milton Road, Cambridge CB4 1XE, England

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation Inc.,
    59 Temple Place, Suite 330, Boston, MA  02111-1307  USA; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.
*/
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// 
//                    APPLY FUNCTION OF MATRIX TO VECTOR
//
//     Enter with positive symmetric matrix A and vector operand b
//     Also regularisation coefficient alpha, and scalar function f
//     (preferably of simulatable form --- see below).
//     The stopping criterion assumes that A is to be used in regularised form
//                         B = alpha*I + A
//
//     fnMatNew allocates memory for the fnMat structure.
//
//     fnMatPerform generates the conjugate-gradient scalars that define
//                         "Lo" and "Hi" subspaces, following fnMatNew.
//
//     fnMatLoScalar then generates lower bound to scalar  b'.f(A).b,
//                         following fnMatLoEigen.
//     fnMatLoVector then sets v approximating f(A).b
//                         following fnMatLoScalar.
//
//     fnMatHiScalar then generates upper bound to scalar  b'.f(A).b,
//                         following fnMatHiEigen.
//     fnMatHiVector then sets w approximating  ((f(0)-f(A))/A).b
//                         following fnMatHiScalar.
//
//     fnMatOld frees memory from the fnMat structure.
//
//            fnMatNew . . . . . . fnMatPerform . . . . . . fnMatOld
//                                      |
//                               ---------------
//                              |               |
//                         fnMatLoScalar   fnMatHiScalar
//                              |               |
//                         fnMatLoVector   fnMatHiVector
//
//     fnMatCheck checks that the SCALAR and VECTOR operators are transposes,
//                         as may be required.
//
// Theory:    Conjugate gradient approaches the inverse of A through
//            "Problem A", which is to maximise
//                     qa = 2 b'.v - v'.A.v ,  with solution
//                     qa(max) = b'.A^(-1).b  at  v = A^(-1).b
//            in progressively wider subspaces {b, A.b, A^2.b, ...}.
//            It obtains progessively higher estimates of qa at the cost of
//            successive applications of A.
//            The same vectors can be used to simulate results for
//                     B = alpha*I + A
//            without doing any additional transforms.
//            "Problem B" is to maximise
//                     qb = 2 b'.v - v'.B.v , with solution
//                     qb(max) = b'.B^(-1).b  at  v = B^(-1).b
//            "Problem AB" is to maximise
//                     qab = 2 b'.A.v - v'.A.B.v , with solution
//                     qab(max) = b'.A.B^(-1).b  at  v = B^(-1).b
//
//            Now, at maximum,
//                     alpha*qb(max) + qab(max) = b'.b ,
//            which is known at the outset.  This lets us terminate the
//            procedure when the progressively higher estimates of qb and qab
//            sum to an adequately high fraction of the input power b'.b .
//            This must happen eventually, because the solutions become exact
//            when the search directions span the vector space. 
//
//            In fact, we get lower and upper bounds to  b'.(t*I + A)^(-1).b
//            for any positive t, with improved precision if t > alpha, but
//            damaged precision if t < alpha.  We also get lower and upper
//            bounds for any positive linear combination.
//
// Definition: A "simulatable" function can be expressed in the form
//
//                            infty    W(t)
//                    f(x) = INTEGRAL ----- dt + constant,    W >= 0.
//                           t=alpha  t + x
//
//            This means that f = Laplace(Laplace(W)) must be a gently
//            decreasing function whose derivatives alternate in sign.  
//            Placing the lower limit at alpha (instead of 0) guards against
//            premature termination.
//
//            "Problem B" now approximates v = f(A).b in Lo-space
//                        with b'.v = qb <= b'.f(A).b
//            "Problem AB" now approximates w = F(A).b in Hi-space
//                        with b'.A.w = qab <= b'.A.F(A).b
//            When F(x) = (f(0) - f(x)) / x  (as hardwired in fnMatHiScalar)
//                      b'.f(A).b + b'.A.F(A).b = f(0)b'.b,
//            which is known, so that qab yields an upper limit on b'.f(A).b,
//            to accompany the lowewr limit qb.
//
//            Vectors v and w are integrals (over t) of vectors that have
//            maximising properties, though v and w themselves seem not to
//            maximise anything useful. However, as the subspaces expand,
//                   v --> f(A).b
//                   w --> F(A).b
//
// Examples:  The central simulatable function is the regularised inverse
//                    f(x) = 1 / (alpha + x)
//            from W(t) = delta(t - alpha), used to control termination.
//
//            Another simulatable function is
//                    f(x) = (beta + x)^(-k)
//            with fractional power 0 < k <= 1 and beta >= alpha,
//            derived from W(t) = t^(-k) integrated upwards from beta.
//
//            Yet another simulatable function is
//                    f(x) = -log(beta + x)   with beta >= alpha,
//            derivable as the suitably-offset limit of small k.
//
//            If f(x) is simulatable from W(t), then
//                    F(x) = (f(0)-f(x))/x   is simulatable from W(t)/t.
//
//            If a function is not simulatable, the procedure may still
//            give useful estimates, but termination may be premature with
//            too-wide bounds (if W extends into t < alpha), or broken bounds
//            (if f does not admit an underlying W).
//
// Outline:   Conjugate-gradient simulation provides a subspace spanned by
//            orthonormal gradient vectors, within which A is modelled by a
//            symmetric tridiagonal matrix expressed as bi-diagonal factors.
//            A general function f is applied by doing singular value
//            decomposition of these subspace factors, after which f can be
//            applied to the eigenvalues to simulate the effect of f(A).
//
//   --------------       --------   --------   -------------- 
//  |              |     | :    : | | ^    ^ | |......^ ......|
//  |              |     | :    : | | g'.A.g | |      g'      |
//  |              |     | : ^  : | |        | |..............|
//  |   model  A   |  =  | : g  : |  --------   -------------- 
//  |              |     | :    : |
//  |              |     | :    : |           ^
//  |              |     | :    : |      (the g are orthonormal base vectors)
//   --------------       --------   
//                                   --------     -------  ------- 
//                                  | ^    ^ |   |       ||       |
//  Subspace is generated as U'.U:  | g'.A.g | = |   U'  ||   U   |
//                                  |        |   |       ||       |
//                                   --------     -------  -------
//  where
//   -------     -------  -------  -------     --------  -------  --------
//  |       |   |\      ||1 -1   ||\      |   | :    : ||\      ||........|
//  |   U   | = |  del  ||   1 -1||  gam  | = | :Lvec: ||   d   ||  Rvec' |
//  |       |   |      \||      1||      \|   | :    : ||      \||........|
//   -------     -------  -------  -------     --------  -------  --------
//                  U can be inverted           U can also be diagonalised
//  Conjugate gradient is founded on U being inverted.
//  This matrix-function procedure is based on U being diagonalised so that
//                        -------  -------     --------  -------  --------
//                       |       ||       |   | :    : ||\      ||........|
//                       |   U'  ||   U   | = | :Rvec: ||  d^2  ||  Rvec' |
//                       |       ||       |   | :    : ||      \||........|
//                        -------  -------     --------  -------  --------
//  and
//   --------------     --------  --------  --------  --------  -------------- 
//  |              |   | :    : || :    : ||\       ||........||......^ ......|
//  |              |   | :    : || :Rvec: || f(d^2) ||  Rvec' ||      g'      |
//  |              |   | : ^  : || :    : ||       \||........||..............|
//  |  model f(A)  | = | : g  : | --------  --------  --------  -------------- 
//  |              |   | :    : |     
//  |              |   | :    : |
//  |              |   | :    : |     
//   --------------     -------- 
//
// History:   John Skilling    17 Dec 1993
//                             18 Mar 1994 External release
//                             20 May 2006 Modernisation and re-packaging
//                             24 Sep 2008 g vectors stored for vector results
//                             12 Oct 2008 Extensive polishing
//=============================================================================
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include "fnmatrix.h"

#ifdef PARALLEL
#include <mpi.h>
#endif

/************************/
/* Constants and macros */
/************************/
#undef  CALLOC    // allocates vector p[0:n-1] of type t
#undef  FREE      // frees CALLOC or NULL vector p[0:*], sets NULL
#undef  CALL      // call mechanism enabling clean abort if return code < 0

#define CALLOC(p,n,t) {p=NULL;if((n)>0){\
p=(t*)calloc((size_t)(n),sizeof(t));if(!(p)){CALLvalue=E_MALLOC;goto Exit;}\
 /*printf("%p %d\n",p,(size_t)(n)*sizeof(t));*/ }}
#define FREE(p) {if(p){ /*printf("%p -1\n",p);*/ (void)free((void*)p);}p=NULL;}
#define CALL(x)    {if( (CALLvalue = (x)) < 0 ) goto Exit;}

/**************/
/* Prototypes */
/**************/

static void   fnMatLoEigen(sfnMatrix*);
static void   fnMatHiEigen(sfnMatrix*);
static void   EigenStruct(float*, float*, int, float*);
static void   EigenSplit (float*, float*, int, int, float*);
static int    EigenMaxrot(float*, float*, int, int, float*);
static int    EigenRotate(float, float, float*, float*, int, int, float*);
static void   vzero      (float*, int);
static float vdot       (float*, float*, int);
static void   vsmula     (float*, float*, float, float*, int);
static void   vcopy      (float*, float*, int);
static float vrand      (float*, int, float*);

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatNew
// 
// Purpose:   Allocate and initialise fnMat structure 
//
//                      ======= SUBSPACE "SCALARS" ========   ==VECTORS==
//                      gam del Coeff Evals Spins Func work    g   h   u
//                              Lo Hi Lo Hi Lo Hi Lo Hi       all
//       fnMatNew        +   +   + +   + +   + +   + +   +     +   +   +
//       fnMatPerform    W   W   W W   W W   W W               W  (W) (W)
//
//       fnMatLoScalar           R     R           W
//       fnMatHiScalar   R         R     R           W
//       fnMatLoVector   R       R           R     R    (W)    R
//       fnMatHiVector       R     R           R     R  (W)    R
//
//       fnMatCheck                                               (W) (W)
//       fnMatOld        -   -   - -   - -   - -   - -   -     -   -   -
//
// History:   John Skilling    12 Oct 2008
//-----------------------------------------------------------------------------
int fnMatNew(         //   O  0=OK, -ve = memory allocation error
sfnMatrix* psfnMat,   //   O  fnMat structure, ready for use
int        Ndim,      // I    vector dimension
int        Nmax)      // I    max # transforms >= 1
{
    int  n = (Nmax + 2) / 2;
    int  i;
    int  CALLvalue = 0;
    memset(psfnMat, 0, sizeof(psfnMat));
    psfnMat->Ndim = Ndim;
    psfnMat->Nmax = Nmax;
    CALLOC(psfnMat->gam,     n, float)
    CALLOC(psfnMat->del,     n, float)
    CALLOC(psfnMat->work,    n, float)
    CALLOC(psfnMat->LoCoeff, n, float)
    CALLOC(psfnMat->HiCoeff, n, float)
    CALLOC(psfnMat->LoEvals, n, float)
    CALLOC(psfnMat->HiEvals, n, float)
    CALLOC(psfnMat->LoSpins, 1 + 3*n*(n+1), float)
    CALLOC(psfnMat->HiSpins, 1 + 3*n*(n+1), float)
    CALLOC(psfnMat->h,       Ndim, float)
    CALLOC(psfnMat->u,       Ndim, float)
    CALLOC(psfnMat->g,       n, float*)
    CALLOC(psfnMat->g[0],    n * Ndim, float)
    for( i = 1; i < n; i++ )
        psfnMat->g[i] = psfnMat->g[i-1] + Ndim;
    psfnMat->random = 1234567890.;  // fnMatCheck seed in [1,2147483646], not 0
Exit:
    return CALLvalue;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatPerform
// 
// Purpose:   Perform conjugate gradient.  Generate gradient vectors g and set
//            upper bi-diagonal model matrix U as
//            EITHER (e.g.)                      OR (e.g.)
//             --------  --------  --------       --------  --------  -------- 
//            |del0    ||1 -1    ||gam0    |     |del0    ||1 -1    ||gam0    |
//            |  del1  ||   1 -1 ||  gam1  |     |  del1  ||   1 -1 ||  gam1  |
//            |    del2||      1 ||    gam2|     |      0 ||      1 ||    gam2|
//             --------  --------  --------       --------  --------  --------
//              ndel=3              ngam=3         ndel=2              ngam=3
//            Lo-subspace calculation of qb      Hi-subspace calculation of qab
//            terminated with ngam=ndel          terminated with ngam=ndel+1
//
// History:   John Skilling    12 Oct 2008
//                             06 Nov 2010  eigencalculation forced here
//-----------------------------------------------------------------------------
int fnMatPerform(    //   O  # transforms performed, -ve = user error
sfnMatrix* psfnMat,  //   O  conjugate-gradient results
void*      A,        // I    transform 
float*    b,        // I    source vector                               [Ndim]
float     alpha,    // I    regularisation coefficient
float     utol)     // I    fractional tolerance
{
    float   SMALL  = FLT_EPSILON * sqrt(FLT_EPSILON);
// vectors
    int      Ndim   = psfnMat->Ndim;     // I    vector dimension
    float** g      = psfnMat->g;        //   O  gradients         [ngam][Ndim]
    float*  h      = psfnMat->h;        //  (O) conjugate vector        [Ndim]
    float*  u      = psfnMat->u;        //  (O) working vector          [Ndim]
// subspace
    int      Nmax   = psfnMat->Nmax;     // I    max # transforms >= 1
    float*  gam    = psfnMat->gam;      //   O  subspace gradients      [ngam]
    float*  del    = psfnMat->del;      //   O  subspace conjugates     [ndel]
// internal
    int      ntrans = 0;        // # transforms performed
    int      ngam   = 1;        // Hi-subspace dimension
    int      ndel   = 0;        // Lo-subspace dimension
    float   gamma;             // gradient scalar                     /* A  */
    float   gamma0;            // gradient scalar on entry            /* A  */
    float   delta  = alpha;    // conjugate scalar                    /* A  */
    float   qb     = 0.0;      // simulated result                    /*  B */
    float   delb   = alpha;    // simulated delta                     /*  B */
    float   phib   = 0.0;      // delb recurrence variable            /*  B */
    float   epsb   = 1.0;      // delb recurrence variable            /*  B */
    float   qab    = 0.0;      // simulated result                    /* AB */
    float   delab  = 0.0;      // simulated delta                     /* AB */
    float   phiab  = 0.0;      // delab recurrence variable           /* AB */
    float   epsab;             // delab recurrence variable           /* AB */
    float   temp;
    int      CALLvalue = 0;

    vcopy(g[0], b, Ndim);
    vzero(h, Ndim);
    vzero(u, Ndim);

// Initialise conjugate gradient
    epsab = gamma = gamma0 = vdot(g[0], g[0], Ndim);
    gam[0] = sqrt(gamma);

// Conjugate gradient loop
    if( gamma > 0.0 )
    for( ; ; )
    {
        temp = 1.0 / epsb - epsb * delb / delta;                       /*  B */
        epsb = delta / (epsb * delb);                                  /*  B */
        phib += alpha / (epsb * gamma * epsb) + temp * delta * temp;   /*  B */
        vsmula(h, h, 1.0 / gamma, g[ngam-1], Ndim);                    /* A  */
        CALL( SCALAR(&delta, A, h) )                                   /* A  */
        if( delta < 0.0 )                          /* USER ERROR */
            delta = 0.0;
        delb = phib + (delta / epsb) / epsb;                           /*  B */
        qb += 1.0 / delb;                                              /*  B */
        del[ndel++] = sqrt(delta);
        temp = alpha * qb * (1.0 + utol) + qab;
        if( ++ntrans >= Nmax || temp >= gamma0 || delta == 0.0 )
            break;

        temp = 1.0 / epsab - delab * epsab / gamma;                    /* AB */
        phiab += alpha / (epsab * delta * epsab) + temp * gamma * temp;/* AB */
        CALL( VECTOR(u, A) )                                           /* A  */
        vsmula(g[ngam], g[ngam-1], -1.0 / delta, u, Ndim);             /* A  */
        gamma = vdot(g[ngam], g[ngam], Ndim);                          /* A  */
        delab = phiab + (gamma / epsab) / epsab;                       /* AB */
        epsab = gamma / (epsab * delab);                               /* AB */
        qab += 1.0 / delab;                                            /* AB */

        del[ndel]= 0.0;
        gam[ngam++] = sqrt(gamma);
        temp = alpha * qb * (1.0 + utol) + qab;
        if( ++ntrans >= Nmax || temp >= gamma0 || gamma <= gamma0 * SMALL )
            break;    // final termination should have gamma = O(FLT_EPSILON^2)
    }

    psfnMat->ngam = ngam;
    psfnMat->ndel = ndel;

// Prepare eigenstructures;
    fnMatLoEigen(psfnMat);
    fnMatHiEigen(psfnMat);
    CALLvalue = ntrans;

Exit:
    return  CALLvalue;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatRepeat
// 
// Purpose:   Perform conjugate gradient on new source vector b,
//            passively using scalars from previous seed.
//
// History:   John Skilling    18 Nov 2010
//-----------------------------------------------------------------------------
int fnMatRepeat(     //   O  # transforms performed, -ve = user error
sfnMatrix* psfnMat,  //   O  conjugate-gradient results
void*      A,        // I    transform 
float*    b)        // I    source vector                               [Ndim]
{
// vectors
    int      Ndim   = psfnMat->Ndim;     // I    vector dimension
    float** g      = psfnMat->g;        //   O  gradients         [ngam][Ndim]
    float*  h      = psfnMat->h;        //  (O) conjugate vector        [Ndim]
    float*  u      = psfnMat->u;        //  (O) working vector          [Ndim]
// subspace
    float*  gam    = psfnMat->gam;      //   O  subspace gradients      [ngam]
    float*  del    = psfnMat->del;      //   O  subspace conjugates     [ndel]
// internal
    int      ngam   = 1;        // Hi-subspace dimension
    int      ndel   = 0;        // Lo-subspace dimension
    float   dummy;
    int      CALLvalue = 0;

    vcopy(g[0], b, Ndim);
    vzero(h, Ndim);
    vzero(u, Ndim);

// Conjugate gradient loop
    if( psfnMat->ngam + psfnMat->ndel > 1 )
    for( ; ; )
    {
        vsmula(h, h, 1.0 / (gam[ngam-1] * gam[ngam-1]), g[ngam-1], Ndim);
        CALL( SCALAR(&dummy, A, h) )
        ndel++;
        if( ngam + ndel >= psfnMat->ngam + psfnMat->ndel )
            break;

        CALL( VECTOR(u, A) )
        vsmula(g[ngam], g[ngam-1], -1.0 / (del[ndel-1] * del[ndel-1]), u, Ndim);
        ngam++;
        if( ngam + ndel >= psfnMat->ngam + psfnMat->ndel )
            break;
    }
    CALLvalue = ngam + ndel - 1;
Exit:
    return  CALLvalue;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatLoEigen
// 
// Purpose:   Eigenstructure of Lo-subspace with dimension ndel.
//
//   -------     --------  -------  --------     --------  -------  --------
//  |       |   |del0    ||1 -1   ||gam0    |   | :    : ||\      ||........|
//  |   U   | = |  del1  ||   1 -1||  gam1  | = | :Lvec: ||   d   ||  Rvec' |
//  |       |   |    del2||      1||    gam2|   | :    : ||      \||........|
//   -------     --------  -------  --------     --------  -------  --------
//
//            Also output
//                ---     --------   --- 
//               |   |   |........| | G |    (G=gam0)
//               | x | = |  Rvec' | | 0 |
//               |   |   |........| | 0 |
//                ---     --------   --- 
//            for later calculation of LoScalar and LoVector.
//
// History:   John Skilling    16 Oct 2008
//-----------------------------------------------------------------------------
static void fnMatLoEigen(
sfnMatrix* psfnMat)  // I O  conjugate-gradient and eigen results
{
// Results
    int     ndel  = psfnMat->ndel;     // I    Lo-subspace dimension
    float* gam   = psfnMat->gam;      // I    subspace gradients        [ndel]
    float* del   = psfnMat->del;      // I    subspace conjugates       [ndel]
    float* Spins = psfnMat->LoSpins;  //   O  (Rvec,Lvec)spins     [1+3*nSpin]
    float* Evals = psfnMat->LoEvals;  //   O  eigenvalues d             [ndel]
    float* x     = psfnMat->LoCoeff;  //   O  subspace coeffs           [ndel]
// Internal
    float  c, s, temp;
    int     i, j;
 
// Upper bidiagonal matrix U
    for( i = 0; i < ndel; ++i )
        Evals[i] = gam[i] * del[i];
    for( i = 1; i < ndel; ++i )
        x[i] = gam[i] * del[i-1];

// Eigenvalues and right eigenvectors needed for lower bound
    EigenStruct(Evals, x, ndel, Spins);
    for( i = 0; i < ndel; i++ )
        Evals[i] = Evals[i] * Evals[i];

// Obtain lower limit on scalar  b.f(A).b  by simulating 
//                 qb = b'. f(A) . b
    for( i = 0; i < ndel; ++i )
        x[i] = 0.0;
    if( ndel > 0 )
        x[0] = gam[0];

// Apply right eigenvectors stored as even spins;   x := Rvec[transpose].x
    for( i = 0; i < 3 * (int)Spins[0]; i += 3 )
    if( Spins[i+1] > 0.0 )  // select right
    {
        j = (int)Spins[i+1];    c = Spins[i+2];    s = Spins[i+3];
        temp   = s * x[j-1] + c * x[j];
        x[j-1] = c * x[j-1] - s * x[j];
        x[j]   = temp;
    }
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatLoScalar
// 
// Purpose:   Set         -------  --------  --------  --------   --- 
//                  qb = | G 0 0 || :    : ||\       ||........| | G |
//                        ------- | :Rvec: || f(d^2) ||  Rvec' | | 0 |
//                                | :    : ||       \||........| | 0 |
//                                 --------  --------  --------   --- 
//                     = lower bound to scalar  b'.f(A).b
//
// History:   John Skilling    06 Nov 2008
//                             06 Nov 2010  eigencalculation now done before
//-----------------------------------------------------------------------------
float fnMatLoScalar(//   O  lower bound to scalar  b'.f(A).b
sfnMatrix* psfnMat,  // I O  search direction coefficients
void*      A,        // I    transform 
int        flag)     // I    function identifier
{
    int     ndel  = psfnMat->ndel;     // I    Lo-subspace dimension
    float* Evals = psfnMat->LoEvals;  // I    eigenvalues               [ndel]
    float* x     = psfnMat->LoCoeff;  // I    subspace coeffs           [ndel]
    float  func;          // function values f(.)      [ndel]
    float  qb = 0.0;      // scalar result for Problem B with function fnMat
    int     i;

// Apply function f(x) of matrix
    for( i = 0; i < ndel; ++i )
    {
        func = fnMat(A, Evals[i], flag);    // pre-squared eigenvalues
        qb += x[i] * func * x[i];
    }
    return qb;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatLoVector
// 
// Purpose:   Set vector  v = f(A).b  obeying  b'.v = qb
//                ---     --------  --------  --------  --- 
//               |   |   | :    : || :    : ||\       ||   |
//               |   |   | :    : || :Rvec: || f(d^2) || x |
//               |   |   | : ^  : || :    : ||       \||   |
//               | v | = | : g  : | --------  --------  --- 
//               |   |   | :    : |
//               |   |   | :    : |     
//               |   |   | :    : |     
//                ---     --------   
//
// History:   John Skilling    16 Oct 2008, 18 Nov 2010
//-----------------------------------------------------------------------------
void fnMatLoVector(
sfnMatrix* psfnMat,  // I    search direction coefficients
void*      A,        // I    transform 
float*    v,        //   O  result f(A).b                               [Ndim]
int        flag)     // I    function identifier
{
    int      ndel  = psfnMat->ndel;     // I    Lo-subspace dimension
    float*  gam   = psfnMat->gam;      // I    subspace gradients       [ndel]
    float*  Evals = psfnMat->LoEvals;  // I    eigenvalues              [ndel]
    float*  x     = psfnMat->LoCoeff;  // I    subspace coeffs          [ndel]
    float*  Spins = psfnMat->LoSpins;  // I    (Rvec,Lvec)spins    [1+3*nSpin]
    float** g     = psfnMat->g;        // I    gradients          [ndel][Ndim]
    int      Ndim  = psfnMat->Ndim;     // I    vector dimension
    float*  w     = psfnMat->work;     //  (O) workspace                [ndel]
    int      i, j;
    float   c, s, temp;

    for( i = 0; i < ndel; i++ )
        w[i] = x[i] * fnMat(A, Evals[i], flag);
// Apply right eigenvectors stored as even spins;   w = Rvec.x (not transpose)
    for( i = 3 * (int)Spins[0] - 3; i >= 0; i -= 3 )
    if( Spins[i+1] > 0.0 )  // select right
    {
        j = (int)Spins[i+1];    c = Spins[i+2];    s = Spins[i+3];
        temp   = c * w[j] - s * w[j-1];
        w[j-1] = s * w[j] + c * w[j-1];
        w[j]   = temp;
    }
// Accumulate and de-normalise to compensate for the g's being un-normalised
    vzero(v, Ndim); 
    for( i = 0; i < ndel; i++ )
        vsmula(v, v, w[i]/gam[i], g[i], Ndim);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatHiEigen
// 
// Purpose:   Eigenstructure of Hi-subspace with dimension ngam.
//
//   -------     --------  -------  --------     --------  -------  --------
//  |       |   |del0    ||1 -1   ||gam0    |   | :    : ||\      ||........|
//  |   U   | = |  del1  ||   1 -1||  gam1  | = | :Lvec: ||   d   ||  Rvec' |
//  |       |   |      0 ||      1||    gam2|   | :    : ||      \||........|
//   -------     --------  -------  --------     --------  -------  --------
//
//            Also output
//                ---     --------   --- 
//               |   |   |........| |GGD|      (G=gam0)
//               | y | = |  Lvec' | | 0 |      (D=del0)
//               |   |   |........| | 0 |
//                ---     --------   --- 
//            for later calculation of HiScalar and HiVector.
//
// History:   John Skilling    16 Oct 2008
//-----------------------------------------------------------------------------
static void fnMatHiEigen(
sfnMatrix* psfnMat)  // I O  conjugate-gradient and eigen results
{
// Results
    int     nhi    = psfnMat->ngam - 1; // I    Hi-subspace dimension
    float* gam    = psfnMat->gam;      // I    subspace gradients       [ngam]
    float* del    = psfnMat->del;      // I    subspace conjugates      [ndel]
    float* Spins  = psfnMat->HiSpins;  //   O  (Rvec,Lvec)spins    [1+3*nSpin]
    float* Evals  = psfnMat->HiEvals;  //   O  eigenvalues d            [ngam]
    float* y      = psfnMat->HiCoeff;  //   O  subspace coeffs           [nhi]
// Internal
    float  c, s, temp;
    int     i, j;

// Upper bidiagonal matrix U
    for( i = 0; i < nhi; ++i )
        Evals[i] = gam[i] * del[i];
    Evals[nhi] = 0.0;
    for( i = 1; i <= nhi; ++i )
        y[i] = gam[i] * del[i-1];

// Eigenvalues and left eigenvectors needed for upper bound
    EigenStruct(Evals, y, nhi + 1, Spins);   // Evals[nhi]=0 will be ignored
    for( i = 0; i < nhi; i++ )
        Evals[i] = Evals[i] * Evals[i];

// Obtain upper limit on scalar  b.f(A).b  by simulating 
//             qab = b'. A . (f(0)-f(A))/A . b
    y[0] = gam[0] * del[0] * gam[0];
    for( i = 1; i < nhi; ++i )
        y[i] = 0.0;

// Apply left eigenvectors stored as odd spins;   e := Lvec[transpose].e
    for( i = 0; i < 3 * (int)Spins[0]; i += 3 )
    if( Spins[i+1] < 0.0 )  // select left
    {
        j = -(int)Spins[i+1];    c = Spins[i+2];    s = Spins[i+3];
        temp   = s * y[j-1] + c * y[j];
        y[j-1] = c * y[j-1] - s * y[j];
        y[j]   = temp;
    }
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatHiScalar
// 
// Purpose:   With F(x) = ((f(0) - f(x)) / x, set 
//                        --------  --------  --------  --------   --- 
//                 qab = |GGD 0 0 || :    : ||\       ||........| |GGD|
//                        -------- | :Lvec: || F(d^2) ||  Lvec' | | 0 |
//                                 | :    : ||       \||........| | 0 |
//                                  --------  --------  --------   --- 
//            and return upper bound  f(0)b'.b - qab  to scalar  b'.f(A).b
//
// History:   John Skilling    06 Nov 2008
//                             06 Nov 2010  eigencalculation now done before
//-----------------------------------------------------------------------------
float fnMatHiScalar(//   O  upper bound to scalar  b'.f(A).b
sfnMatrix* psfnMat,  // I O  search direction coefficients
void*      A,        // I    transform 
int        flag)     // I    function identifier
{
    int      nhi   = psfnMat->ngam - 1; // I    Hi-subspace dimension
    float*  gam   = psfnMat->gam;      // I    subspace gradients          [1]
    float*  Evals = psfnMat->HiEvals;  // I    eigenvalues               [nhi]
    float*  y     = psfnMat->HiCoeff;  // I    subspace coeffs           [nhi]
    float   qab   = 0.0;   // scalar result for Problem AB with function fnMat
    float   func, func0;
    int      i;

    func0 = fnMat(A, 0.0, flag);
// Apply function (f(0)-f(x))/x of matrix
    for( i = 0; i < nhi; ++i )
    {
        func = (func0 - fnMat(A, Evals[i], flag)) / Evals[i]; // pre-squared
        qab += y[i] * func * y[i];
    }
    return  gam[0] * func0 * gam[0] - qab;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatHiVector
// 
// Purpose:   Set vector  w = F(A).b  according to
//                ---     --------  -------  --------  --------  --- 
//               |   |   | :    : ||    -1 || :    : ||\       ||   |
//               |   |   | :    : ||   U   || :Lvec: || F(d^2) || y |
//               |   |   | : ^  : ||       || :    : ||       \||   |
//               | w | = | : g  : | -------  --------  --------  --- 
//               |   |   | :    : |     
//               |   |   | :    : |
//               |   |   | :    : |     
//                ---     -------- 
//
// History:   John Skilling    16 Oct 2008, 18 Nov 2010
//-----------------------------------------------------------------------------
void fnMatHiVector(
sfnMatrix* psfnMat,  // I    search direction coefficients
void*      A,        // I    transform 
float*    w,        //   O  result  F(A).b                              [Ndim]
int        flag)     // I    function identifier
{
    int      nhi   = psfnMat->ngam - 1; // I    Hi-subspace dimension
    float*  gam   = psfnMat->gam;      // I    subspace gradients        [nhi]
    float*  del   = psfnMat->del;      // I    subspace conjugates       [nhi]
    float*  y     = psfnMat->HiCoeff;  // I    subspace coeffs           [nhi]
    float*  Spins = psfnMat->HiSpins;  // I    (Rvec,Lvec)spins    [1+3*nSpin]
    float*  Evals = psfnMat->HiEvals;  // I    eigenvalues               [nhi]
    int      Ndim  = psfnMat->Ndim;     // I    vector dimension
    float** g     = psfnMat->g;        // I    gradients           [nhi][Ndim]
    float*  t     = psfnMat->work;     //  (O) workspace                [ngam]
    int      i, j;
    float   c, s, temp;
    float   func0;

    func0 = fnMat(A, 0.0, flag);
    for( i = 0; i < nhi; i++ )
        t[i] = y[i] * (func0 - fnMat(A, Evals[i], flag)) / Evals[i];
// Apply left eigenvectors stored as odd spins;   w = Lvec.y (not transpose)
    for( i = 3 * (int)Spins[0] - 3; i >= 0; i -= 3 )
    if( Spins[i+1] < 0.0 )  // select left
    {
        j = -(int)Spins[i+1];    c = Spins[i+2];    s = Spins[i+3];
        temp   = c * t[j] - s * t[j-1];
        t[j-1] = s * t[j] + c * t[j-1];
        t[j]   = temp;
    }
// Apply U^-1, with extra factor of gam because g not normalised
    for( i = 0; i < nhi; i++ )
        t[i] /= del[i];
    for( i = nhi - 1; i > 0; i-- )
        t[i-1] += t[i];
    for( i = 0; i < nhi; i++ )
        t[i] /= gam[i] * gam[i];
// Accumulate
    vzero(w, Ndim);
    for( i = 0; i < nhi; i++ )
       vsmula(w, w, t[i], g[i], Ndim);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatCheck
// 
// Purpose:   Check for inconsistent transpose in traditional opus/tropus form
//            of SCALAR and VECTOR procedures.
//            If opus and tropus are correctly each other's transposes, then
//            dimensionless result err should be of order of rounding error,
//
// Method:    Set  h = random in real space,  F = random in data space.
//  
//                           fabs( h . tropus(F)  -  F . opus(h) ) 
//            Return   err = -------------------------------------
//                           sqrt( |h| |tropus(F)| |F| |opus(h)| ) 
//
// Note:      Calls can be scattered through an application to make sure
//            opus and tropus remain transpose.  The only side effects are on
//            the vectors h amd u (which are only needed within fnMatPerform).
//
// History:   John Skilling    12 Oct 2008   modified from MemTrq in MemSys5
//-----------------------------------------------------------------------------
float fnMatCheck(   //   O  dimensionless opus/tropus inconsistency
sfnMatrix* psfnMat,  //  (O) work-space
void*      A,        // I    transform 
int        Ndata,    // I    opus(=SCALAR) operates as a Ndata-by-Ndim matrix
float*    F)        // I    destination in A of SCALAR(...,h)          [Ndata]
{
    int      Ndim  = psfnMat->Ndim;  // I    real dimension
    float*  h     = psfnMat->h;     //  (O) real-space (0 on exit)      [Ndim]
    float*  u     = psfnMat->u;     //  (O) real-space (0 on exit)      [Ndim]
    float   hh;     // |h|
    float   hf;     // h.tropus(F)
    float   ff;     // |tropus(F)|
    float   FF;     // |F|
    float   FH;     // F.opus(h)
    float   HH;     // |opus(h)|
    int      CALLvalue = 0;
    
    int myid = 0;
#ifdef PARALLEL
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#endif
    
    if( myid == 0 )
    {
    	vzero(h, Ndim);
    	vrand(h, Ndim, &psfnMat->random);
    }
#ifdef PARALLEL
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(h, Ndim, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
    hh = sqrt(vdot(h, h, Ndim));
    CALL( SCALAR(&HH, A, h) )
    HH = sqrt(HH);
    if( myid == 0 ) FH = vrand(F, Ndata, &psfnMat->random);
#ifdef PARALLEL
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(F, Ndata, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif
    FF = sqrt(vdot(F, F, Ndata));
    CALL( VECTOR(u, A) )
    hf = vdot(h, u, Ndim);
    ff = sqrt(vdot(u, u, Ndim));
    return  fabs(hf - FH) / sqrt(hh * ff * FF * HH + FLT_MIN);
Exit:
    return CALLvalue;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  fnMatOld
// 
// Purpose:   Free fnMat structure 
//
// History:   John Skilling    12 Oct 2008
//-----------------------------------------------------------------------------
void fnMatOld(sfnMatrix* psfnMat)
{
    FREE(psfnMat->g[0])
    FREE(psfnMat->g)
    FREE(psfnMat->u)
    FREE(psfnMat->h)
    FREE(psfnMat->HiSpins)
    FREE(psfnMat->LoSpins)
    FREE(psfnMat->HiEvals)
    FREE(psfnMat->LoEvals)
    FREE(psfnMat->HiCoeff)
    FREE(psfnMat->LoCoeff)
    FREE(psfnMat->work)
    FREE(psfnMat->del)
    FREE(psfnMat->gam)
}
    
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//                             Eigenstructure
//-----------------------------------------------------------------------------

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  EigenStruct
//
// Purpose:   "Eigenstructure" (SVD) of upper bidiagonal matrix.
//
// Method:    Rotate U into diagonal form
//                    ---------     ---------   ---------   ---------
//                   |d e      |   |         | |D 0      | | ------- |
//                   |  d e    |   | |     | | |  D 0    | |      t  |
//               U = |    d e  | = | |Lvec | | |    D 0  | |  Rvec   |
//                   |      d e|   | |     | | |      D 0| |         |
//                   |        d|   |         | |        D| | ------- |
//                    ---------     ---------   ---------   ---------
//            for which the eigenvalues are D (overwriting d), whilst
//            the superdiagonal e is overwritten by small residuals.
//            Eigenvectors form the right and left rotation matrices
//            Rvec and Lvec, which are output directly into the "Spins"
//            list of N right-spin and N left-spin 2x2 matrices {j,c,s}.
//                                -----------     -------------
//                               |           |   |1            |
//                               |   Spins   |   |   1         |
//            On the right of U, |  [right]  | = |      c -s   | j-1
//                               |           |   |      s  c   | j
//                               |           |   |            1|
//                                -----------     -------------
//                                                     j-1 j
//            in terms of which
//                     ---------     ----------        ----------   ----------
//                    | ------- |   |1         |      |c -s      | |1         |
//                    |      t  |   |  c -s    |      |s  c      | |  1       |
//                    |  Rvec   | = |  s  c    | .... |     1    | |    1     |
//                    |         |   |       1  |      |       1  | |      c -s|
//                    | ------- |   |         1|      |         1| |      s  c|
//                     ---------     ----------        ----------   ----------
//                                   Spins[2N-4] ....   Spins[2]     Spins[0]
//  
//                                -----------     -------------
//                               |           |   |1            |
//                               |   Spins   |   |   1         |
//            On the left of U,  |   [left]  | = |      c  s   | j-1
//                               |           |   |     -s  c   | j
//                               |           |   |            1|
//                                -----------     -------------
//                                                     j-1 j
//            in terms of which
//                     ---------     ----------        ----------   ----------
//                    | ------- |   |1         |      |c -s      | |1         |
//                    |      t  |   |  c -s    |      |s  c      | |  1       |
//                    |  Lvec   | = |  s  c    | .... |     1    | |    1     |
//                    |         |   |       1  |      |       1  | |      c -s|
//                    | ------- |   |         1|      |         1| |      s  c|
//                     ---------     ----------        ----------   ----------
//                                   Spins[2N-3] ....   Spins[3]     Spins[1]
// 
// Notes: (1) Input diagonal elements in d[0..n-1] must be strictly positive,
//            except that d[n-1] may be 0.  Input superdiagonal elements
//            e[1..n-1] must be strictly negative, except that e[n-1] may be 0.
//            (The vector e stores their positive negations.)
//            fnMatPerform automatically conforms to this signage.
//        (2) If d[n-1]=0, that sole null eigenvalue is guaranteed to remain
//            in place at the last position n-1, accessed by a zero-angle spin.
//        (3) The positive eigenvalues usually return in decreasing order,
//            unless awkward conditioning makes the matrix block-diagonal.
//        (4) Spins are output instead of the direct eigenvectors because
//            spinning all n base vectors to get the n eigenvectors involves
//            heavily redundant computation that can damage the accuracy of
//            results for any specific operand.
//        (5) The full number of left and right spins is <= n(n-1).
//        (6) CPU = O(n^2 P), P = # bits of arithmetic precision.
//
// History:   JS         12 Dec 1993     Modified from MemDet in MemSys5
//                       20 May 2006     Eigenvectors replaced by spin matrices
//                       23 Sep 2008     General polish
//-----------------------------------------------------------------------------
static void EigenStruct(
float*  d,            // I    +(  Diagonal    elements 0..n-1 of U)
                       //   O  eigenvalues D                                [n]
float*  e,            // I    -(Superdiagonal elements 1..n-1 of U)
                       //   O  small negative residuals                     [n]
int      n,            // I    dimension, nSpin <= n(n-1)/2
float*  Spins)        //   O  #spins, then (right,left) list       [1+3*nSpin]
{
    Spins[0] = 0;      // length of spin storage
    EigenSplit(d, e, 0, n-1, Spins);
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  EigenSplit
// 
// Purpose:   Rotate upper bidiagonal matrix U, using maximum angle which
//            keeps all off-diagonal components -ve (stored in e as +ve).
//
// Method:    At the maximum angle, an e[.] will be near zero.  Split the
//            matrix at this point into block diagonal factors, and use 
//            recursive calls to rotate the two factors.
//
// Notes:     Usually, the limiting component is the last, e[n-1], in which
//            case the second factor is the trivial 1x1 matrix d[n-1].
//            But accumulated magnification of rounding errors can prevent
//            e[n-1] being driven close to 0, in which case some intervening
//            e[.] will get smaller.
// 
// History:   JS         12 Dec 1993     Modified from MemDet in MemSys5
//                       20 May 2006     Modernised with Spins
//-----------------------------------------------------------------------------
static void EigenSplit(
float*  d,            // I O  +(  Diagonal    elements 0..n-1 of U)
float*  e,            // I O  -(Superdiagonal elements 1..n-1 of U)
int      i,            // I    start index  0 <= [i,..
int      k,            // I    end index              ..,k] < n
float*  Spins)        // I O  spin list
{
    int  j;
    if( k > i )
    {
        j = EigenMaxrot(d, e, i,   k, Spins);
        if( j > i+1 )
            EigenSplit (d, e, i, j-1, Spins);
        if( j < k )
            EigenSplit (d, e, j,   k, Spins);
    }
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  EigenMaxrot
// 
// Purpose:   Rotate upper bidiagonal matrix U, using maximum angle which
//            keeps all off-diagonal components -ve (stored in e as +ve).
//
// History:   JS         12 Dec 1993     Modified from MemDet in MemSys5
//                       20 May 2006     Modernised with spins
//                       23 Sep 2008     Slight polish
//-----------------------------------------------------------------------------
static int EigenMaxrot(//   O  &(smallest super-diagonal element)
float*  d,            // I O  +(  Diagonal    elements 0..n-1 of U)
float*  e,            // I O  -(Superdiagonal elements 1..n-1 of U)
int      i,            // I    start index  0 <= [i,..
int      k,            // I    end index              ..,k] < n
float*  Spins)        // I O  spin list
{
const float LOOSE = 1.0;   /* Use 2^R to allow loss of R bits of precision */
    float  C, S;           // cos(theta),sin(theta)
    float  COK, SOK;       // cos and sin of smallest acceptable theta
    float  a, b, c;        // binary chop a < b < c
    float  hold, hnew;     // interval h
    float  x;              // smallest superdiagonal e
    float  t;              // temp
    int     n;              // index along off-diagonals

// Exit if input already OK
    if( e[k] <= 0.0 )
        return k;
// Chop (c,s) = (cos(theta),sin(theta)) between theta=0 (guaranteed not enough)
// and theta=pi/2 (guaranteed to change sign of very first 2x2 superdiagonal)
    a = LOOSE;    c = LOOSE + 1.0;    /* divisible down to allowed precision */
    hnew = c - a;
    COK = 1.0;    SOK = 0.0;
    do                 // chop b (= controller of theta) within [LOOSE,LOOSE+1]
    {
        hold = hnew;    b = (a + c) / 2.0;
        S = b - LOOSE;    C = 1.0 - S;     // preliminary sin,cos
        t = sqrt(C * C + S * S);
        S /= t;    C /= t;                 // normalised sin,cos
// Try rotation
        if( EigenRotate(C, S, d, e, i, k, NULL) )
        {                                  // OK, remember and increase
            a = b;    COK = C;    SOK = S;
        }
        else                               // not OK, reduce
            c = b;
        hnew = c - a;
    } while( hold > hnew );                // until allowed precision
// Use last (best) OK rotation
    EigenRotate(COK, SOK, d, e, i, k, Spins);
// Find smallest superdiagonal
    x = e[i+1] / (d[i] + d[i+1]);    n = i+1;
    for( i++; i <= k; i++ )
    {
        t = e[i] / (d[i-1] + d[i]);
        if( x > t )
        {
            x = t;    n = i;
        }
    }
    return n;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  EigenRotate
// 
// Purpose:   Rotate upper bidiagonal matrix U, preserving its signed form.
//            If Spins=NULL, simulate rotation without doing it and check signs
//
//            Otherwise, do the rotation (starting with angle theta), and
//            record the rotation matrix.  At the critical theta, one of the
//            superdiagonal elements of U (usually the last) will be close to 0
//            and the left and right rotation matrices will be moving towards
//            the left and right eigenvectors.
//
// Method:    Apply right-Spin (c=cos,s=sin) to upper bidiagonal matrix 
//                          -----------   ----------- 
//                         |d0 e1      | | c  s      |
//                         |   d1 e2   | |-s  c      |   (e.g. dimension n=4)
//               U.Spin =  |      d2 e3| |       1   |   (e actually negative)
//                         |         d3| |          1|
//                          -----------   -----------
//            to mix first two columns, then left-Spin to mix first two rows,
//            then further right and left spins down the chain, with angles
//            chosen to keep the upper bidiagonal form.  Successively:-
//            ------- 
//           |d e    |
//           |  d e  |
//           |    d e|
//           |      d|
//            ------- 
//    Spins[0]| |on right
//            | |
//            V V
//            -------  Spins[1]  ------- 
//           |d e    |  ------> |d e X  |
//           |X d e  |  ------> |  d e  |
//           |    d e| on left  |    d e|
//           |      d|          |      d|
//            -------            ------- 
//                         Spins[2]| |on right
//                                 | |
//                                 V V
//                               -------            ------- 
//                              |d e    | Spins[3] |d e    |
//                              |  d e  |  ------> |  d e X|
//                              |  X d e|  ------> |    d e|
//                              |      d| on left  |      d|
//                               -------            ------- 
//                                              Spins[4]| |on right
//                                                      | |
//                                                      V V
//                                                  -------            ------- 
//                                                 |d e    |          |d e    |
//                                                 |  d e  | Spins[5] |  d e  |
//                                                 |    d e|  ------> |    d e|
//                                                 |    X d|  ------> |      d|
//                                                  -------  on left   -------
//            Exit immediately with "rotation too large" if any off-diagonal
//            term would swap sign if matrix were terminated at the currently-
//            processed dimension.
//
// Notes: (1) Input diagonal elements in d must be strictly positive,
//            except that d[n-1] may be 0.  Input superdiagonal elements
//            must be strictly negative, except that e[n-1] may be 0.
//            (The vector e stores their positive negations.)
//        (2) This sign convention is designed to be preserved on output.
//            The internal signs in this routine are specifically chosen
//            to ensure that the spinning preserves these signs, and
//            that all internal variables should stay positive.
//        (3) If d[n-1]=0, the final left spin of this rotation is guaranteed
//            to be null, leaving d[n-1] unchanged as the null eigenvalue.
// 
// History:   JS         12 Dec 1993     Modified from MemDet in MemSys5
//                       24 Mar 1994     Case d[n-1]=0 debugged
//                       20 May 2006     Eigenvectors replaced by Spin matrices
//                       23 Sep 2008     Slight polish
//                       16 Oct 2008     null spins not output
//-----------------------------------------------------------------------------
static int EigenRotate(//   O  1=OK, but 0 if any sign of e tries to change
float   c,            // I    cos theta              1 >= c >= 0
float   s,            // I    sin theta              0 <= s <= 1
float*  d,            // I O  +(  Diagonal    elements  i...k of U)
float*  e,            // I O  -(Superdiagonal elements i+1..k of U)
int      i,            // I    start index  0 <= [i,..
int      k,            // I    end index              ..,k] < n
float*  Spins)        // I O  #spins then list of spins, or NULL
{
    float  d0, d1, e1, r, x;
    int     j = i+1;
    int     n = 0;
    if( Spins )
         n = 3 * (int)Spins[0];

    d1 = d[i];    e1 = e[i+1];    r = e[i];
    for( ; ; )
    {
// Mix columns with spin on right; x becomes overflow element on lower-left.
        if( Spins )
        {
            e[j-1] = r;
            if( s > 0.0 )         // null spin ignored
            {
                Spins[n+1] = j;   // right spins stored +ve
                Spins[n+2] = c;
                Spins[n+3] = s;
                n += 3;
            }
        }
        d0 = c * d1 + s * e1;
        e1 = c * e1 - s * d1;
        d1 = d[j];    x  = s * d1;    d1 = c * d1;
// Mix rows with spin on left; x becomes overflow element on upper-right.
        r = sqrt(d0 * d0 + x * x);    c = d0 / r;    s = x / r;
        if( Spins )
        {
            d[j-1] = r;
            if( s > 0.0 )         // null spin ignored
            {
                Spins[n+1] = -j;  // left spins stored -ve
                Spins[n+2] = c;
                Spins[n+3] = s;
                n += 3;
            }
        }
        d0 = s * d1 + c * e1;
        d1 = c * d1 - s * e1;
// Require all e[last] in 1x1, 2x2, 3x3... to stay +ve
        if( d0 <= 0.0 )
            return 0;
// Finished?
        if( j == k )
        {
            if( Spins )
            {
                e[j] = d0;    d[j] = d1;
                Spins[0] = n / 3;
            }
            return 1;   // OK: no changes of sign
        }
        e1 = e[++j];    x  = s * e1;    e1 = c * e1;
        r = sqrt(d0 * d0 + x * x);    c = d0 / r;    s = x / r;
    }
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//                              Vector library
//-----------------------------------------------------------------------------

static void vzero(       //  u = 0 
float* u,
int     n)
{
    int    i;
    for( i = 0; i < n; ++i )
        u[i] = 0.0;
}

static float vdot(       //  a = u.v 
float* u,
float* v,
int     n)
{
    int    i;
    float a = 0.0;
    for( i = 0; i < n; ++i )
        a += u[i] * v[i];
    return a;
}

static void vsmula(       //  u += v + a * w 
float* u,
float* v,
float  a,
float* w,
int     n)
{
    int  i;
    for( i = 0; i < n; ++i )
        u[i] = v[i] + a * w[i];
}

static void vcopy(        //  u = v 
float* u,
float* v,
int     n)
{
    int  i;
    for( i = 0; i < n; ++i )
        u[i] = v[i];
}

static float vrand(  //   O  u[input].u[output]
float* u,            // I O  output u = random, uniform(-1,1)
int     n,            // I    dimension
float* random)       // I O  adequate "minimal standard" standalone generator
{
    float r, a = 0.0;
    int    i;
    for( i = 0; i < n; ++i )
    {
        *random = fmod(16807.0 * *random, 2147483647.0); // [1,2147483646]
        r = *random * 2.0 / 2147483647.0 - 1.0;          // (-1,1)
        a += u[i] * r;
        u[i] = r;
    }
    return a;
}

#if 0                 /* Alternative codes */
//=============================================================================
//     invMatLoScalar is alternate to fnMatLoScalar for inverse b'.B^(-1).b
//                         with no need for eigen-analysis.
//     invMatLoVector is alternate to fnMatLoVector for inverse B^(-1).b
//                         but twice the vector operations.
//
//     invMatHiScalar is alternate to fnMatHiScalar for inverse,
//                         with no need for eigen-analysis.
//     invMatHiVector is alternate to fnMatHiVector for inverse,
//                         but thrice the vector operations.
//
//            fnMatNew . . . . . . fnMatPerform . . . . . . fnMatOld
//                                      |
//                               ---------------
//                              |               |
//                        invMatLoScalar  invMatHiScalar
//                              |               |
//                        invMatLoVector  invMatHiVector
//
//                       ======= SUBSPACE "SCALARS" ========    ===VECTORS===
//                       gam del Coeff Evals Spins Func work    g   h   u   F
//                               Lo Hi Lo Hi Lo Hi Lo Hi       all
//       invMatLoScalar   R   R
//       invMatHiScalar   R   R
//       invMatLoVector   R   R                                 R      (W)
//       invMatHiVector   R   R                                 R  (W) (W)
//
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  invMatLoScalar, alternative to fnMatLoScalar for inverse.
//                                          -1
// Purpose:   Set   qb = lower bound to b'.B .b   in subspace
//
// History:   John Skilling    09 Oct 2008
//-----------------------------------------------------------------------------
float invMatLoScalar(//   O  lower bound to scalar  b'.B^(-1).b
sfnMatrix* psfnMat,   // I    search direction coefficients
float     alpha)     // I    regularisation coefficient in B = alpha*I + A
{
    int      ntrans = psfnMat->ngam + psfnMat->ndel - 1;  // I    # transforms
    float*  gam    = psfnMat->gam;    // I    subspace gradients        [ngam]
    float*  del    = psfnMat->del;    // I    subspace conjugates       [ndel]
    int      ngam   = 1;               // Hi-subspace dimension
    int      ndel   = 0;               // Lo-subspace dimension
    float   gamma  = gam[0] * gam[0]; // gradient scalar              /* A  */
    float   delta  = alpha;           // conjugate scalar             /* A  */
    float   qb     = 0.0;             // simulated result             /*  B */
    float   delb   = alpha;           // simulated delta              /*  B */
    float   phib   = 0.0;             // delb recurrence variable     /*  B */
    float   epsb   = 1.0;             // delb recurrence variable     /*  B */
    float   temp;
    for( ; ; )
    {
        temp = 1.0 / epsb - epsb * delb / delta;                       /*  B */
        epsb = delta / (epsb * delb);                                  /*  B */
        phib += alpha / (epsb * gamma * epsb) + temp * delta * temp;   /*  B */
        delta = del[ndel] * del[ndel];    ndel++;
        delb = phib + (delta / epsb) / epsb;                           /*  B */
        qb += 1.0 / delb;                                              /*  B */
        if( ngam + ndel > ntrans )
            break;
        gamma = gam[ngam] * gam[ngam];    ngam++;
        if( ngam + ndel > ntrans )
            break;
    }
    return qb;
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  invMatLoVector, alternative to fnMatLoVector for inverse.
//
// Purpose:   Set vector v maximising qb in subspace and obeying b'.v = qb
// 
// History:   John Skilling    09 Oct 2008
//-----------------------------------------------------------------------------
void invMatLoVector(
sfnMatrix* psfnMat,   // I    search direction coefficients
float     alpha,     // I    regularisation coefficient in B = alpha*I + A
float*    v)         //   O  vector B^(-1).b
{
    int      ntrans = psfnMat->ngam + psfnMat->ndel - 1;  // I    # transforms
    int      Ndim   = psfnMat->Ndim;   // I    vector dimension
    float*  gam    = psfnMat->gam;    // I    subspace gradients        [ngam]
    float*  del    = psfnMat->del;    // I    subspace conjugates       [ndel]
    float** g      = psfnMat->g;      // I    gradients           [ngam][Ndim]
    float*  u      = psfnMat->u;      //  (O) working vector            [Ndim]
    int      ngam   = 1;               // Hi-subspace dimension
    int      ndel   = 0;               // Lo-subspace dimension
    float   gamma  = gam[0] * gam[0]; // gradient scalar              /* A  */
    float   delta  = alpha;           // conjugate scalar             /* A  */
    float   delb   = alpha;           // simulated delta              /*  B */
    float   phib   = 0.0;             // delb recurrence variable     /*  B */
    float   epsb   = 1.0;             // delb recurrence variable     /*  B */
    float   temp;

    vzero(v, Ndim);
    vzero(u, Ndim);
    for( ; ; )
    {
        temp = 1.0 / epsb - epsb * delb / delta;                       /*  B */
        epsb = delta / (epsb * delb);                                  /*  B */
        phib += alpha / (epsb * gamma * epsb) + temp * delta * temp;   /*  B */
        delta = del[ndel] * del[ndel];    ndel++;
        delb = phib + (delta / epsb) / epsb;                           /*  B */
        vsmula(u, u, 1.0 / (epsb * gamma), g[ngam-1], Ndim);
        vsmula(v, v, 1.0 / delb, u, Ndim);
        if( ngam + ndel > ntrans )
            break;
        gamma = gam[ngam] * gam[ngam];    ngam++;
        if( ngam + ndel > ntrans )
            break;
    }
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  invMatHiScalar, alternative to fnMatHiScalar for inverse.
//                                                  -1
// Purpose:   Set qab and return upper bound to b'.B .b
// 
// History:   John Skilling    09 Oct 2008
//-----------------------------------------------------------------------------
float invMatHiScalar(//   O  upper bound to scalar  b'.B^(-1).b
sfnMatrix* psfnMat,   // I    search direction coefficients
float     alpha)     // I    regularisation coefficient in B = alpha*I + A
{
    int      ntrans = psfnMat->ngam + psfnMat->ndel - 1;  // I    # transforms
    float*  gam    = psfnMat->gam;    // I    subspace gradients        [ngam]
    float*  del    = psfnMat->del;    // I    subspace conjugates       [ndel]
    int      ngam   = 1;               // Hi-subspace dimension
    int      ndel   = 0;               // Lo-subspace dimension
    float   gamma  = gam[0] * gam[0]; // gradient scalar              /* A  */
    float   delta  = alpha;           // conjugate scalar             /* A  */
    float   qab    = 0.0;             // simulated result             /* AB */
    float   delab  = 0.0;             // simulated delta              /* AB */
    float   phiab  = 0.0;             // delab recurrence variable    /* AB */
    float   epsab  = gamma;           // delab recurrence variable    /* AB */
    float   temp;
    for( ; ; )
    {
        delta = del[ndel] * del[ndel];    ndel++;
        if( ngam + ndel > ntrans )
            break;
        temp = 1.0 / epsab - delab * epsab / gamma;                    /* AB */
        phiab += alpha / (epsab * delta * epsab) + temp * gamma * temp;/* AB */
        gamma = gam[ngam] * gam[ngam];    ngam++;
        delab = phiab + (gamma / epsab) / epsab;                       /* AB */
        epsab = gamma / (epsab * delab);                               /* AB */
        qab += 1.0 / delab;                                            /* AB */
        if( ngam + ndel > ntrans )
            break;
    }
    return  (gam[0] * gam[0] - qab) / alpha;                     // upper bound
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  invMatHiVector, alternative to fnMatHiVector for inverse.
//
// Purpose:   Set vector w maximising qab in subspace and obeying b'.A.w = qab
//
// History:   John Skilling    09 Oct 2008
//-----------------------------------------------------------------------------
void invMatHiVector(
sfnMatrix* psfnMat,   // I    search direction coefficients
float     alpha,     // I    regularisation coefficient in B = alpha*I + A
float*    w)         //   O  vector B^(-1).b / alpha
{
    int      ntrans = psfnMat->ngam + psfnMat->ndel - 1;  // I    # transforms
    int      Ndim   = psfnMat->Ndim;   // I    vector dimension
    float*  gam    = psfnMat->gam;    // I    subspace gradients        [ngam]
    float*  del    = psfnMat->del;    // I    subspace conjugates       [ndel]
    float** g      = psfnMat->g;      // I    gradients           [ngam][Ndim]
    float*  h      = psfnMat->h;      //  (O) working vector            [Ndim]
    float*  u      = psfnMat->u;      //  (O) working vector            [Ndim]
    int      ngam   = 1;               // Hi-subspace dimension
    int      ndel   = 0;               // Lo-subspace dimension
    float   gamma  = gam[0] * gam[0]; // gradient scalar              /* A  */
    float   delta  = alpha;           // conjugate scalar             /* A  */
    float   delab  = 0.0;             // simulated delta              /* AB */
    float   phiab  = 0.0;             // delab recurrence variable    /* AB */
    float   epsab  = gamma;           // delab recurrence variable    /* AB */
    float   temp;
    vzero(w, Ndim);
    vzero(h, Ndim);
    vzero(u, Ndim);
    for( ; ; )
    {
        vsmula(h, h, 1.0 / gamma, g[ngam-1], Ndim);                    /* A  */
        delta = del[ndel] * del[ndel];    ndel++;
        if( ngam + ndel > ntrans )
            break;
        vsmula(u, u, 1.0 / (epsab * delta), h, Ndim);
        temp = 1.0 / epsab - delab * epsab / gamma;                    /* AB */
        phiab += alpha / (epsab * delta * epsab) + temp * gamma * temp;/* AB */
        gamma = gam[ngam] * gam[ngam];    ngam++;
        delab = phiab + (gamma / epsab) / epsab;                       /* AB */
        epsab = gamma / (epsab * delab);                               /* AB */
        vsmula(w, w, 1.0 / (alpha * delab), u, Ndim);
        if( ngam + ndel > ntrans )
            break;
    }
}
#endif
