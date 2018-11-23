#include "myrand.h"

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  ran2
// 
// Purpose:   Generates a uniform random variable between 0 and 1.
//-----------------------------------------------------------------------------
float ran2(long *idum)
{
  int j;
  long k;
  static long idum2 = 123456789;
  static long iy = 0;
  static long iv[NTAB];
  float temp;

  if (*idum <= 0) {
    if (-(*idum) < 1) *idum = 1;
    else *idum = -(*idum);
    idum2 = (*idum);
    for (j = NTAB + 7; j >= 0; j--) {
      k = (*idum)/IQ1;
      *idum = IA1*(*idum - k*IQ1) - k*IR1;
      if (*idum < 0) *idum += IM1;
      if (j < NTAB) iv[j] = *idum;
    }
    iy = iv[0];
  }
  k = (*idum)/IQ1;
  *idum = IA1*(*idum - k*IQ1) - k*IR1;
  if (*idum < 0) *idum += IM1;
  k = idum2/IQ2;
  idum2 = IA2*(idum2 - k*IQ2) - k*IR2;
  if (idum2 < 0) idum2 += IM2;
  j = iy/NDIV;
  iy = iv[j] - idum2;
  iv[j] = *idum;
  if (iy < 1) iy += IMM1;
  if ((temp = AM*iy) > RNMX) return RNMX;
  else return temp;
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  gasdev
// 
// Purpose:   Generates a standard normal random variable.
//-----------------------------------------------------------------------------
float gasdev(long *idum)
{
        static int iset=0;
        static float gset;
        float fac,rsq,v1,v2;

        if  (iset == 0) {
                do {
                        v1=2.0*ran2(idum)-1.0;
                        v2=2.0*ran2(idum)-1.0;
                        rsq=v1*v1+v2*v2;
                } while (rsq >= 1.0 || rsq == 0.0);
                fac=sqrt(-2.0*log(rsq)/rsq);
                gset=v1*fac;
                iset=1;
                return v2*fac;
        } else {
                iset=0;
                return gset;
        }
}
