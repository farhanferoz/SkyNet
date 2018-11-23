#include "NNopt.h"
#include "linbcg.h"

#include "NNopt.cc"
#include "fnmatrix.cc"
#include "myrand.cc"
#include "linbcg.cc"

void PrintIntro() {
	printf("*******************************************\n");
	printf(" SkyNet Predictions v1.2\n");
	printf(" Copyright F Feroz, P Graff, M P Hobson\n");
	printf(" Release May 2014\n");
	printf("*******************************************\n\n");
}

void PrintPredictions(std::string filename, NeuralNetwork *nn, TrainingData &td, PredictedData &pd, bool hhps, bool text, std::map <int, char> &revcharmap) {
	int i,j,h;
	size_t k;
	FILE *of=fopen(filename.c_str(),"w");
	//fprintf(of,"%d\t%d\n",(int)td.nin,(int)td.nout);
	std::vector <float> predin(td.nin);
	std::vector <float> predout(td.nout);
	std::vector <float> trueout(td.nout);
	float score=0.0;
	int n=0;
	int cc[] = {0, 0};
	for( i = 0; i < td.ndata; i++ ) {
		nn->forward(td, pd, i, false);
		
		std::string context, actualstr, predstr;
		if (text) {
			for (h=0;h<td.ntimeignore[i]+1;h++) {
				for (j=0;j<td.nin;j++) {
					if (td.inputs[td.cuminsets[i]*td.nin+h*td.nin+j]==1.0) {
						context += revcharmap.find(j+1)->second;
					}
				}
			}
			context += "---";
			actualstr=predstr=context;
		}
		
		for (h=td.ntimeignore[i];h<td.ntime[i];h++) {
			int predchar, actualchar;
			float predprob=0.0;
		
			k = td.cuminsets[i]*pd.totnnodes+(h+1)*pd.totnnodes-pd.nout;
			for( j = 0; j < td.nin; j++ ) predin[j] = td.inputs[td.cuminsets[i]*td.nin+h*td.nin+j];
			for( j = 0; j < td.nout; j++ ) predout[j] = pd.out[k+j];
			for( j = 0; j < td.nin; j++ ) fprintf(of,"%lf\t",predin[j]);
			
			// new, updated loop - should work
			if (text) {
				for( j = 0; j < td.nout; j++ ) {
					float outactual=td.outputs[(td.cumoutsets[i]+h-td.ntimeignore[i])*td.nout+j];
					float outpred=predout[j];
					if (outactual==1.0) {
						actualstr += revcharmap.find(j+1)->second;
						actualchar=j;
					}
					if (outpred>predprob) {
						predprob=outpred;
						predchar=j;
					}
				}
				
				predstr += revcharmap.find(predchar+1)->second;
				if (predchar==actualchar) cc[0]++;
				cc[1]++;
			} else {
				for( j = 0; j < td.nout; j++ ) {
					float outactual=hhps?exp(td.outputs[(td.cumoutsets[i]+h-td.ntimeignore[i])*td.nout+j])-1.0:td.outputs[(td.cumoutsets[i]+h-td.ntimeignore[i])*td.nout+j];
					float outpred=hhps?exp(predout[j])-1.0:predout[j];
					if(hhps && outpred<0.0) outpred=0.0;
					if(hhps && outpred>15.0) outpred=15.0;
					trueout[j]=outactual;
					predout[j]=outpred;
					if (hhps) {
						score += pow(log(outpred+1.0)-log(outactual+1.0),2.0);
						n++;
					}
					fprintf(of,"%.6lf\t",outactual);
				}
				for( j = 0; j < td.nout; j++ ) fprintf(of,"%lf\t",predout[j]);
			}
			
			
			/*for( j = 0; j < td.nout; j++ ) {
				float outactual=hhps?exp(td.outputs[(td.cumoutsets[i]+h-td.ntimeignore[i])*td.nout+j])-1.0:td.outputs[(td.cumoutsets[i]+h-td.ntimeignore[i])*td.nout+j];
				float outpred=hhps?exp(predout[j])-1.0:predout[j];
				if(hhps && outpred<0.0) outpred=0.0;
				if(hhps && outpred>15.0) outpred=15.0;
				//outpred=round(outpred*1E6)/1E6;
				
				if (text) {
					if (outactual==1.0) {
						actualstr += revcharmap.find(j+1)->second;
						actualchar=j;
					}
					if (outpred>predprob) {
						predprob=outpred;
						predchar=j;
					}
				}
				else
				{
					fprintf(of,"%.6lf\t",outactual);
					fprintf(of,"%lf\t",outpred);
				}

				if(hhps) {
					score += pow(log(outpred+1.0)-log(outactual+1.0),2.0);
					n++;
				}

			}*/
			
			if (text) {
				predstr += revcharmap.find(predchar+1)->second;
				if (predchar==actualchar) cc[0]++;
				cc[1]++;
			}
			else {
				fprintf(of,"\n");
			}
		}
		
		if (text) fprintf(of,"%s\n%s\n",actualstr.c_str(),predstr.c_str());
	}
	fclose(of);
	
	if (hhps) {
		score=sqrt(score/float(n));
		printf("score=%g\n",score);
	}
	
	if (text) printf("%g%% of the characters correctly predicted.\n",100.0*float(cc[0])/float(cc[1]));
}

int main(int argc, char **argv) {
	if (argc<11) {
		printf("You need to specify:\n");
		printf("\t1) data type: 0=blind, 1=known outputs\n");
		printf("\t2) network type: 0=reg, 1=class, 2=text, 3=autoencoder\n");
		printf("\t3) network type: 0=reg, 1=recurrent\n");
		printf("\t4) input network save file\n");
		printf("\t5) input training data file\n");
		printf("\t6) data set file\n");
		printf("\t7) output file\n");
		printf("\t8) loglike function: 0=none, 1=standard, 2=HHPScore, 3=standard on log(out+1)\n");
		printf("\t9) print error? 0=no, 1=yes\n");
		printf("\t10) read in accuracies? 0=no, 1=yes\n");
		printf("\t11) output transformation file (optional)\n");
		abort();
	}
	
	PrintIntro();
	
	int i,j,n,nalphas=1;
	float logL,logP,logC,*alpha,*beta,temp,temp2;
	size_t nhid   = argc-7;
	
	std::string innet = argv[4];
	std::string traindata = argv[5];
	std::string datin = argv[6];
	std::string datout = argv[7];
	int hhpl=atoi(argv[8]);
	if(hhpl>3 || hhpl<0) hhpl=1;
	bool hhps=(hhpl==1||hhpl==0)?false:true;
	
	bool classnet=false;				// whether it's a classification net or not
	bool text=false;
	bool autoencoder=false;
	bool recurrent;
	bool known;
	bool printerr;
	bool readacc;
		
	if (atoi(argv[1])==0) known=false;
	else known=true;
	
	if (atoi(argv[2])==1) classnet=true;
	else if (atoi(argv[2])==2) text=classnet=true;
	else if (atoi(argv[2])==3) autoencoder=true;
	if (classnet) hhps=false;
	//if (!hhps) hhpl=1;
	
	if (atoi(argv[3])==0) recurrent=false;
	else recurrent=true;
	if (text) recurrent=true;
	
	if (atoi(argv[9])==0) printerr=false;
	else printerr=true;
	
	if (atoi(argv[10])==0) readacc=false;
	else readacc=true;
	
	TrainingData td1,td2;
	std::map <char, int> charmap;
	std::map <int, char> revcharmap;
	
	if (text) {
		std::string mapfile = innet.substr(0, innet.find("network.txt")) + "charmap.txt";
		td2.ReadTextMap(mapfile, charmap);
		td2.GenTrainingDataFromText(datin, charmap);
		
		std::map<char, int>::iterator it;
		for( it = charmap.begin(); it != charmap.end(); it++ )
			revcharmap[(*it).second] = (*it).first;
	}
	else {
		if(!classnet) td1=TrainingData(traindata,classnet,autoencoder,readacc);
		td2=TrainingData(datin,classnet,autoencoder,readacc);
	}

	if(hhpl==2 || hhpl==3) {
		for( size_t i = 0; i < td2.cumoutsets[td2.ndata]*td2.nout; i++ ) td2.outputs[i] = log(td2.outputs[i]+1.0);
		if(hhpl==3) hhpl=1;
	}

	printf("Read in %d data points.\n",(int)td2.ndata);
	
	NeuralNetwork *nn;
	if (classnet)
		nn = new FeedForwardClassNetwork();
	else
		nn = new FeedForwardNeuralNetwork();

	alpha=(float *)malloc(nalphas*sizeof(float));
	beta=(float *)malloc(td2.nout*sizeof(float));
	
	float rate;
	nn->read(innet,&rate,alpha,beta);
	if (hhpl>0) nn->loglikef=hhpl;

	if (!classnet) {
		for (n=0;n<td1.ndata;n++)
			for (i=0;i<td1.ntime[n]-td1.ntimeignore[n];i++)
				for (j=0;j<td1.nout;j++)
					td1.acc[td1.cumoutsets[n]*td1.nout+i*td1.nout+j]=beta[j];
		for (n=0;n<td2.ndata;n++)
			for (i=0;i<td2.ntime[n]-td2.ntimeignore[n];i++)
				for (j=0;j<td2.nout;j++)
					td2.acc[td2.cumoutsets[n]*td2.nout+i*td2.nout+j]=beta[j];
	}
	
	printf("Read in network.\n");
	
	// initialize the PredictedData object
	PredictedData pd2(nn->totnnodes, nn->totrnodes, td2.ndata, td2.nin, td2.nout, td2.cuminsets[td2.ndata], td2.cumoutsets[td2.ndata]);
	
	if(text || classnet) {
		PrintPredictions(datout,nn,td2,pd2,hhps,text,revcharmap);
	} else {
		PredictedData pd1(nn->totnnodes, nn->totrnodes, td1.ndata, td1.nin, td1.nout, td1.cuminsets[td1.ndata], td1.cumoutsets[td1.ndata]);
		NN_args args,vargs;
		args.np=vargs.np=nn->nweights;
		args.td=&td1;vargs.td=&td2;
		args.pd=&pd1;vargs.pd=&pd2;
		args.nn=vargs.nn=nn;
		args.nalphas=vargs.nalphas=1;
		args.alphas=vargs.alphas=alpha;
		args.prior=vargs.prior=true;
		args.noise=vargs.noise=false;
		args.Bcode=vargs.Bcode=true;
		
		whiteTransform *itrans, *otrans;
		itrans = new axisAlignedTransform(1, true);
		otrans = new axisAlignedTransform(1, true);
		int whitenout=0;
		
		if (!nn->linear[nn->nlayers-1] && argc>=12) {
			whitenout=1;
			std::ifstream otf(argv[11]);
			otrans->read(otf);
			otf.close();
			printf("Applying the specified output transformation.\n");
		}
		
		float wt[nn->nweights],g[nn->nweights];
		nn->getweights(wt);
		float omicron=FindOmicron(wt,&args,rate,g);
		
		float logL;
		if (hhpl>0) nn->logLike(td1, pd1, logL, true);
		
		PrintPredictions(datout, whitenout, 0, itrans, otrans, &vargs, &args, omicron, classnet, printerr, autoencoder, 3);
	}
	printf("Predictions printed.\n");
	
	if (known) {
		printf("\n");
		
		int Npar = nn->nparams();
		float *x=(float *)malloc(Npar*sizeof(float));
		nn->getweights(x);
		
		nn->setweights(x);
		if (hhpl>0) {
			nn->logLike(td2,pd2,logL);
			nn->logPrior(alpha,logP);
			if (classnet) logC=0.;
			else nn->logC(td2,logC);
			printf("logP=%g\tlogL=%g\tChisq=%g\n",logL+logP+logC,logL+logC,-2.*logL);
		}
		
		if (hhps) printf("score=%g\n",-nn->HHPscore(td2,pd2));
		
		// Print best-fit correlation (and error rate)
		if (classnet) {
			nn->correlations(td2,pd2);
			nn->ErrorSquared(td2,pd2);
			nn->CorrectClass(td2,pd2);
			float temp=0.0;
			for (i=0;i<td2.ncat;i++) {
				temp+=pd2.predrate[i];
			}
			temp/=float(td2.ncat);
			printf("The correlation is %g, error squared is %g, %g%% in correct class.\n\n",pd2.corr[0],pd2.errsqr[0],temp*100.);
		} else {
			nn->correlations(td2,pd2);
			nn->ErrorSquared(td2,pd2);
			for (temp=0.,temp2=0.,i=0;i<td2.nout;i++) {
				temp+=pd2.corr[i];
				temp2+=pd2.errsqr[i];
			}
			temp/=(float)td2.nout;
			temp2/=(float)td2.nout;
			printf("The correlation is %g, error squared is %g\n\n",temp,temp2);
		}
		
		free(x);
	}
	
	free(alpha);
	free(beta);
	
	return 0;
}

