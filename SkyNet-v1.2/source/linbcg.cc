#include "NNopt.h"
#include "linbcg.h"

int linbcg_itol=5;
float linbcg_tol=1E-3;
bool symmetric=true;

void linbcg_solve(float *b, float *x, int &iter, float &err, NN_args *NN, int itmax)
{
	float ak,akden,bk,bkden=1.0,bknum,bnrm,dxnrm,xnrm,zm1nrm,znrm;
	const float eps=1.0e-14;
	int j,n=NN->np;
	float p[n],pp[n],r[n],rr[n],z[n],zz[n];
	std::vector <float> phiHist;
	iter=0;
	linbcg_atimes(x,r,0,NN,n);
	for(j=0;j<n;j++) {
		r[j]=b[j]-r[j];
		rr[j]=r[j];
	}
	//linbcg_atimes(r,rr,0,NN,n);
	if (linbcg_itol == 1 || linbcg_itol == 5) {
		bnrm=linbcg_snrm(b,n);
		linbcg_asolve(r,z,0,n);
	}
	else if (linbcg_itol == 2) {
		linbcg_asolve(b,z,0,n);
		bnrm=linbcg_snrm(z,n);
		linbcg_asolve(r,z,0,n);
	}
	else if (linbcg_itol == 3 || linbcg_itol == 4) {
		linbcg_asolve(b,z,0,n);
		bnrm=linbcg_snrm(z,n);
		linbcg_asolve(r,z,0,n);
		znrm=linbcg_snrm(z,n);
	}
	else throw("illegal itol in linbcg");
	
	while (iter < itmax) {
		++iter;
		
		if(!symmetric) linbcg_asolve(rr,zz,1,n);
		for(bknum=0.0,j=0;j<n;j++) bknum += z[j]*rr[j];
		if (iter == 1) {
			for(j=0;j<n;j++) {
				p[j]=z[j];
				pp[j]=symmetric?p[j]:zz[j];
			}
		} else {
			bk=bknum/bkden;
			for(j=0;j<n;j++) {
				p[j]=bk*p[j]+z[j];
				pp[j]=symmetric?p[j]:bk*pp[j]+zz[j];
			}
		}
		bkden=bknum;
		linbcg_atimes(p,z,0,NN,n);
		for(akden=0.0,j=0;j<n;j++) akden += z[j]*pp[j];
		ak=bknum/akden;
		if(!symmetric) linbcg_atimes(pp,zz,1,NN,n);
		for(j=0;j<n;j++) {
			x[j] += ak*p[j];
			r[j] -= ak*z[j];
			if(symmetric)
				rr[j] = r[j];
			else
				rr[j] -= ak*zz[j];
		}
		linbcg_asolve(r,z,0,n);
		if (linbcg_itol == 1 || linbcg_itol == 5) {
			err=linbcg_snrm(r,n)/bnrm;
			
			if(err > linbcg_tol && linbcg_itol == 5) {
				int itolk = fmax(10, round(0.1*iter));
				//float rx = -inner_product(x.begin(), x.end(), r.begin(), 0E0);
				//float bx = inner_product(b.begin(), b.end(), x.begin(), 0E0);
				float rx=0E0, bx=0E0;
				for (j=0;j<n;j++) {
					rx -= x[j] * r[j];
					bx += b[j] * x[j];
				}
				float phi = (rx - bx)/2E0;
				if( phiHist.size() >= itolk ) {
					int nerase = phiHist.size() - itolk;
					if(nerase > 0) phiHist.erase(phiHist.begin(), phiHist.begin()+nerase);
					err = (phi - phiHist[0])/(float(itolk)*phi);
				}
				phiHist.push_back(phi);
			}
		}
		else if (linbcg_itol == 2)
			err=linbcg_snrm(z,n)/bnrm;
		else if (linbcg_itol == 3 || linbcg_itol == 4) {
			zm1nrm=znrm;
			znrm=linbcg_snrm(z,n);
			if (fabs(zm1nrm-znrm) > eps*znrm) {
				dxnrm=fabs(ak)*linbcg_snrm(p,n);
				err=znrm/fabs(zm1nrm-znrm)*dxnrm;
			} else {
				err=znrm/bnrm;
				continue;
			}
			xnrm=linbcg_snrm(x,n);
			if (err <= 0.5*xnrm) err /= xnrm;
			else {
				err=znrm/bnrm;
				continue;
			}
		}
		if (err <= linbcg_tol) break;
	}
}

float linbcg_snrm(float *sx, const int n)
{
	int i,isamax;
	float ans;
	if (linbcg_itol <= 3) {
		ans = 0.0;
		for(i=0;i<n;i++) ans += sx[i]*sx[i];
		return sqrt(ans);
	} else {
		isamax=0;
		for(i=0;i<n;i++) {
			if (fabs(sx[i]) > fabs(sx[isamax])) isamax=i;
		}
		return fabs(sx[isamax]);
	}
}

void linbcg_atimes(float *x, float *r, const int itrnsp, NN_args *NN, const int n)
{
	float pHv[n];
	/*pclass->PriorHv(x, pHv);
	lclass->LikeHv(x, r);
	transform(r.begin(), r.end(), pHv.begin(), r.begin(), plus<float>());
	transform(r.begin(), r.end(), r.begin(), bind1st(multiplies<float>(), -1E0));*/
	NN->nn->alphaIv(x,NN->alphas,pHv);
	NN->nn->Av(x,*(NN->td),*(NN->pd),r);
	for(int i=0;i<n;i++)
	{
		r[i] -= pHv[i];
		r[i] *= -1.;
	}
}

void linbcg_asolve(float *b, float *x, const int itrnsp, const int n)
{
	for(int i=0;i<n;i++) x[i] = b[i];
}


void linbcg_lnsrch(float *xold, const float fold, float *g, float *p, float *x, float &alam, float &f,
			const float stpmax, bool &check, NN_args *NN)
{
	const float ALF = 1.0e-4, TOLX = FLT_EPSILON;
    float a,alam2=0.0,alamin,b,disc,f2=0.0;
    float rhs1,rhs2,slope=0.0,sum=0.0,temp,test,tmplam;
    int i,n=NN->np;
    check=false;
    for(i=0;i<n;i++) sum += p[i]*p[i];
    sum=sqrt(sum);
    if (sum > stpmax)
        for(i=0;i<n;i++)
            p[i] *= stpmax/sum;
    for(i=0;i<n;i++)
        slope += g[i]*p[i];
    test=0.0;
    for(i=0;i<n;i++) {
        temp=fabs(p[i])/fmax(fabs(xold[i]),1.0);
        if (temp > test) test=temp;
    }
    alamin=TOLX/test;
    alam=1.0;
    for(;;) {
    	for(i=0;i<n;i++) x[i]=xold[i]+alam*p[i];
        f = -logLike(x, NN);
        if (alam < alamin) {
            for(i=0;i<n;i++) x[i]=xold[i];
			alam = 0E0;
            check=true;
            return;
        } else if (f <= fold+ALF*alam*slope) return;
        else {
            if (alam == 1.0)
                tmplam = -slope/(2.0*(f-fold-slope));
            else {
                rhs1=f-fold-alam*slope;
                rhs2=f2-fold-alam2*slope;
                a=(rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2);
                b=(-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2);
                if (a == 0.0) tmplam = -slope/(2.0*b);
                else {
                    disc=b*b-3.0*a*slope;
                    if (disc < 0.0) tmplam=0.5*alam;
                    else if (b <= 0.0) tmplam=(-b+sqrt(disc))/(3.0*a);
                    else tmplam=-slope/(b+sqrt(disc));
                }
                if (tmplam>0.5*alam)
                    tmplam=0.5*alam;
        	}
        }
        alam2=alam;
        f2 = f;
        alam=fmax(tmplam,0.1*alam);
    }
}

//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Function:  goldenSectionSearch
// 
// Purpose:   Recursive golden search function.
//-----------------------------------------------------------------------------
float goldenSectionSearch(NN_args *args, float *x0, float *drn, float fb, float a, float b, float c, float tau)
{
	float x,fx,xx[args->np];
	//float fb,xb[args->np];
	int i;

	//printf("Bounds are:\t%f\t%f\t%f\t==>\t",a,b,c);

	if (c - b > b - a)
		x = b + resgold * (c - b);
    else
    	x = b - resgold * (b - a);

    //printf("New x = %f\n",x);

    if (fabs(c-a) < tau*(fabs(b)+fabs(x)))
    	return (c+a)/2.;

    for (i=0;i<args->np;i++)
    {
    	//xb[i] = x0[i] + b * drn[i];
    	xx[i] = x0[i] + x * drn[i];
    }
    //fb=-logPost(xb,args);
    fx=-logLike(xx,args);
    //printf("New f(x) = %f\n",fx);
    
    if (fx==fb) return (x+b)/2.;

    if (fx<fb)
    {
    	if (c-b > b-a) return goldenSectionSearch(args,x0,drn,fx,b,x,c,tau);
    	else return goldenSectionSearch(args,x0,drn,fx,a,x,b,tau);
    }
    else
    {
    	if (c-b > b-a) return goldenSectionSearch(args,x0,drn,fb,a,b,x,tau);
    	else return goldenSectionSearch(args,x0,drn,fb,x,b,c,tau);
    }
}

