#ifndef _MYRAND_H
#define _MYRAND_H

#include <math.h>

// Values needed for ran2 and gasdev
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1 + IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0 - EPS)

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  ran2
// 
// Purpose:   Generates a uniform random variable between 0 and 1.
//-----------------------------------------------------------------------------
float ran2(long *idum);        // seed variable

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  gasdev
// 
// Purpose:   Generates a standard normal random variable.
//-----------------------------------------------------------------------------
float gasdev(long *idum);      // seed variable

#endif
