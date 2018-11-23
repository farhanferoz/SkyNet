#include "NeuralNetwork.h"

void PrintIntro() {
	printf("*******************************************\n");
	printf(" SkyNet Predictions v0.6\n");
	printf(" Copyright F Feroz, P Graff, M P Hobson\n");
	printf(" Release June 2011\n");
	printf("*******************************************\n\n");
}

void PrintPredictions(std::string filename, NeuralNetwork *nn, TrainingData &td, PredictedData &pd, bool hhps) {
	int i,j,h;
	size_t k;
	FILE *of=fopen(filename.c_str(),"w");
	//fprintf(of,"%d\t%d\n",(int)td.nin,(int)td.nout);
	std::vector <float> predin(td.nin);
	std::vector <float> predout(td.nout);
	float score=0.0;
	int n=0;
	for( i = 0; i < td.ndata; i++ ) {
		nn->forward(td, pd, i, false);
		for (h=td.ntimeignore[i];h<td.ntime[i];h++) {
			k = td.cuminsets[i]*pd.totnnodes+(h+1)*pd.totnnodes-pd.nout;
			//for( j = 0; j < td.nin; j++ ) predin[j] = td.inputs[td.cuminsets[i]*td.nin+h*td.nin+j];
			for( j = 0; j < td.nout; j++ ) predout[j] = pd.out[k+j];
			//for( j = 0; j < td.nin; j++ ) fprintf(of,"%lf\t",predin[j]);
			for( j = 0; j < td.nout; j++ ) {
				float outactual=hhps?exp(td.outputs[(td.cumoutsets[i]+h-td.ntimeignore[i])*td.nout+j])-1.0:td.outputs[(td.cumoutsets[i]+h-td.ntimeignore[i])*td.nout+j];
				float outpred=hhps?exp(predout[j])-1.0:predout[j];
				if(hhps && outpred<0.0) outpred=0.0;
				if(hhps && outpred>15.0) outpred=15.0;
				outpred=round(outpred*1E6)/1E6;
				fprintf(of,"%.6lf\t",outactual);
				fprintf(of,"%lf\t",outpred);

				if(hhps) {
					score += pow(log(outpred+1.0)-log(outactual+1.0),2.0);
					n++;
				}

			}
			fprintf(of,"\n");
		}
	}
	fclose(of);

	score=sqrt(score/float(n));
	printf("score=%g\n",score);
}

int main(int argc, char **argv) {
	if (argc<2) {
		printf("You need to specify:\n");
		printf("\t1) input network save file\n");
		abort();
	}
	
	PrintIntro();
	
	int i,j,n,nalphas=1;
	float logL,logP,logC,*alpha,*beta,temp;
	size_t nhid   = argc-7;
	
	std::string innet  = argv[4];
	std::string datin  = argv[5];
	std::string datout = argv[6];
	int hhpl=atoi(argv[7]);
	if(hhpl>3 || hhpl<1) hhpl=1;
	bool hhps=hhpl==1?false:true;
	
	bool classnet;				// whether it's a classification net or not
	bool recurrent;
	bool known;
	
	if (atoi(argv[1])==0) known=false;
	else known=true;
	
	if (atoi(argv[2])==0) classnet=false;
	else classnet=true;
	if (classnet) hhps=false;
	if (!hhps) hhpl=1;
	
	if (atoi(argv[3])==0) recurrent=false;
	else recurrent=true;
	
	TrainingData td(datin,classnet,recurrent);

	if(hhpl==2 || hhpl==3) {
		for( size_t i = 0; i < td.cumoutsets[td.ndata]*td.nout; i++ ) td.outputs[i] = log(td.outputs[i]+1.0);
		if(hhpl==3) hhpl=1;
	}

	printf("Read in %d data points.\n",(int)td.ndata);
	
	NeuralNetwork *nn;
	if (classnet)
		nn = new FeedForwardClassNetwork();
	else
		nn = new FeedForwardNeuralNetwork();

	alpha=(float *)malloc(nalphas*sizeof(float));
	beta=(float *)malloc(td.nout*sizeof(float));
	
	nn->read(innet,alpha,beta);
	nn->loglikef=hhpl;

	if (!classnet) {
		for (n=0;n<td.ndata;n++)
			for (i=0;i<td.ntime[n]-td.ntimeignore[n];i++)
				for (j=0;j<td.nout;j++)
					td.acc[td.cumoutsets[n]*td.nout+i*td.nout+j]=beta[j];
	}
	
	printf("Read in network.\n");
	
	// initialize the PredictedData object
	PredictedData pd(nn->totnnodes, nn->totrnodes, td.ndata, td.nin, td.nout, td.cuminsets[td.ndata], td.cumoutsets[td.ndata]);
	
	PrintPredictions(datout,nn,td,pd,hhps);
	printf("Predictions printed.\n");
	
	if (known) {
		printf("\n");
		
		int Npar = nn->nparams();
		float *x=(float *)malloc(Npar*sizeof(float));
		nn->getweights(x);
		
		nn->setweights(x);
		nn->logLike(td,pd,logL);
		nn->logPrior(alpha,logP);
		if (classnet) logC=0.;
		else nn->logC(td,logC);
		
		printf("logP=%g\tlogL=%g\tChisq=%g\n",logL+logP+logC,logL+logC,-2.*logL);
		if(hhps) printf("score=%g\n",-nn->HHPscore(td,pd));
		
		// Print best-fit correlation (and error rate)
		if (classnet) {
			nn->correlations(td,pd);
			nn->ErrorSquared(td,pd);
			nn->CorrectClass(td,pd);
			printf("The correlation is %g, error squared is %g, %g%% in correct class.\n\n",pd.corr[0],pd.errsqr[0],pd.predrate*100.);
		} else {
			nn->correlations(td,pd);
			for (temp=0.,i=0;i<td.nout;i++) temp+=pd.corr[i];
			temp/=(float)td.nout;
			printf("The combined correlation for the data is %g\n\n",temp);
		}
		
		free(x);
	}
	
	free(alpha);
	free(beta);
	
	return 0;
}

