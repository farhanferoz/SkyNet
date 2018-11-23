#include "NNopt.h"
#include "linbcg.h"

#include "NNopt.cc"
#include "fnmatrix.cc"
#include "myrand.cc"
#include "linbcg.cc"

void PrintIntro() {
	printf("*******************************************\n");
	printf(" SkyNet Autoencoder Codec v1.2\n");
	printf(" Copyright F Feroz, P Graff, M P Hobson\n");
	printf(" Release May 2014\n");
	printf("*******************************************\n\n");
}

int main(int argc, char **argv) {
	if (argc<6) {
		printf("You need to specify:\n");
		printf("\t1) 0=encoder, 1=decoder\n");
		printf("\t2) Encoder/Decoder network save file\n");
		printf("\t3) Data set file to encode/decode\n");
		printf("\t   ==> Encode with training data format.\n");
		printf("\t   ==> Decode with just feature vector weights.\n");
		printf("\t4) Output file\n");
		printf("\t5) Extra flag...\n");
		printf("\t   Encoder: 0=regression, 1=classification for original data.\n");
		printf("\t   Decoder: 0=no outputs to ignore, 1=outputs to ignore (in data file).\n");
		printf("\t6) Encoder: 0=output in training data format (default), 1=output in predictions format\n");
		printf("\t   Decoder: output transformation file (optional)\n");
		exit(0);
	}
	
	PrintIntro();
	
	bool encode;
	if (atoi(argv[1])==0) encode=true;
	else if (atoi(argv[1])==1) encode=false;
	else {
		printf("Invalid input for encoder/decoder option, must be 0 for encoder or 1 for decoder.\n");
		abort();
	}
	
	bool classnet=false,ignore=true;
	if (encode) {
		if (atoi(argv[5])==0) classnet=false;
		else if (atoi(argv[5])==1) classnet=true;
		else {
			printf("Invalid input for regression/classificaton option, must be 0 for regression or 1 for classification.\n");
			abort();
		}
	} else {
		if (atoi(argv[5])==0) ignore=false;
		else if (atoi(argv[5])==1) ignore=true;
		else {
			printf("Invalid input for ignoring outputs option, must be 0 for none to ignore or 1 for ignore.\n");
			abort();
		}
	}
	
	NeuralNetwork *nn = new FeedForwardNeuralNetwork();
	nn->read(argv[2]);
	int i,j,a,b,c,nin,nout,ncat;
	int *nclasses;
	nin = nn->nnodes[0]-1;
	nout = nn->nnodes[nn->nlayers-1];
	printf("Read in encoder/decoder with %d inputs and %d outputs.\n",nin,nout);
	
	TrainingData td = TrainingData(argv[3],classnet,false);
	
	FILE *infile,*outfile;
	infile=fopen(argv[3],"r");
	outfile=fopen(argv[4],"w");
	
	fscanf(infile,"%d,\n",&a);
	/*if (classnet) {
		ncat=-1;
		do {
			ncat++;
			fscanf(infile,"%d,",&b);
		} while (b!=0);
		fclose(infile);
		nclasses=(int *)malloc(ncat*sizeof(int));
		infile=fopen(argv[3],"r");
		fscanf(infile,"%d,\n",&a);
		for (i=0;i<ncat;i++) fscanf(infile,"%d,",&nclasses[i]);
		fscanf(infile,"\n");
	} else {
		fscanf(infile,"%d,\n",&b);
	}
	if (encode) {
		if (a!=nin) {
			printf("Number of inputs in data file doesn't match the network.\n");
			abort();
		}
	} else {
		if (a!=nin) {
			printf("Number of inputs in data file doesn't match the network.\n");
			abort();	
		}
	}*/
	if (!encode) {
		fscanf(infile,"%d,\n",&b);
		if (a!=nin) {
			printf("Number of inputs in data file doesn't match the network.\n");
			abort();	
		}
	}
	
	float inputs[nin],outputs[nout],save[td.nout];
	char line[10000];
	
	if (encode) {
		int outform = 0;
		printf("Encoding from %d inputs to %d feature vectors.\n\n",nin,nout);
		if (argc>=7)
			if (atoi(argv[6])==1) outform = 1;
		
		if (outform==0) {
			fprintf(outfile,"%d,\n",nout);
			if (classnet) {
			  for (i=0;i<td.ncat;i++) fprintf(outfile,"%d,",(int)td.nclasses[i]);
			  fprintf(outfile,"\n");
			} else {
			  fprintf(outfile,"%d,\n",(int)td.nout);
			}
		}
		/*while (!feof(infile)) {
			// Read in true inputs and outputs
			for (i=0;i<nin-1;i++) fscanf(infile,"%f,",&inputs[i]);
			fscanf(infile,"%f,\n",&inputs[nin-1]);
			if (classnet) {
				for (i=0;i<ncat;i++) fscanf(infile,"%d,",&nclasses[i]);
				fscanf(infile,"\n");
			} else {
				for (i=0;i<b-1;i++) fscanf(infile,"%f,",&save[i]);
				fscanf(infile,"%f,\n",&save[b-1]);
			}
			
			// Encode inputs to feature vectors
			nn->forwardOne(1,inputs,outputs);
			
			// Print feature vectors weights and true outputs
			for (i=0;i<nout;i++) fprintf(outfile,"%f,",outputs[i]);
			fprintf(outfile,"\n");
			if (classnet) {
				for (i=0;i<ncat;i++) fprintf(outfile,"%d,",nclasses[i]);
				fprintf(outfile,"\n");
			} else {
				for (i=0;i<b;i++) fprintf(outfile,"%f,",save[i]);
				fprintf(outfile,"\n");
			}
		}*/
		
		for (i=0;i<td.ndata;i++) {
			// Read in true inputs and outputs
			for (j=0;j<td.nin;j++) inputs[j] = td.inputs[td.cuminsets[i]*td.nin+j];
			for (j=0;j<td.nout;j++) save[j] = td.outputs[td.cumoutsets[i]*td.nout+j];
			
			// Encode inputs to feature vectors
			nn->forwardOne(1,inputs,outputs);
			
			// Print feature vectors weights and true outputs
			if (outform==0) {
				for (j=0;j<nout;j++) fprintf(outfile,"%f,",outputs[j]);
				fprintf(outfile,"\n");
				for (j=0;j<td.nout;j++)
				  if (classnet) {
					if (save[j]==1.0)
						fprintf(outfile,"%d,",j);
				  } else {
				    fprintf(outfile,"%f,",save[j]);
				  }
				fprintf(outfile,"\n");
			} else {
				for (j=0;j<nout;j++) fprintf(outfile,"%f\t",outputs[j]);
				for (j=0;j<td.nout-1;j++) fprintf(outfile,"%f\t",save[j]);
				fprintf(outfile,"%f\n",save[td.nout-1]);
			}
		}
		
		printf("Encoding completed.\n");
	} else {
		printf("Decoding from %d feature vectors to %d outputs.\n\n",nin,nout);
		
		int whitenout=0;
		whiteTransform *otrans;
		otrans = new axisAlignedTransform(1, true);
		if (!nn->linear[nn->nlayers-1] && argc>=7) {
			whitenout=1;
			std::ifstream otf(argv[6]);
			otrans->read(otf);
			otf.close();
			printf("Applying the specified output transformation.\n");
		}
		
		while (!feof(infile)) {
			// Read in feature vector weights
			for (i=0;i<nin-1;i++) fscanf(infile,"%f,",&inputs[i]);
			fscanf(infile,"%f,\n",&inputs[nin-1]);
			if (ignore) fgets(line,10000,infile);
			
			// Decode feature vectors to outputs
			nn->forwardOne(1,inputs,outputs);
			if (whitenout==1) otrans->inverse(outputs,outputs,nout);
			
			// Print feature vector weights and outputs
			for (i=0;i<nin;i++) fprintf(outfile,"%f\t",inputs[i]);
			for (i=0;i<nout-1;i++) fprintf(outfile,"%f\t",outputs[i]);
			fprintf(outfile,"%f\n",outputs[nout-1]);
		}
		
		printf("Decoding completed.\n");
	}
	
	if (!encode) fclose(infile);
	fclose(outfile);
	
	return 0;
}
