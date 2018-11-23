#include "NNsimple.h"

void readNNsimple(const char *filename, NetworkVariables *net)
{
	char temp[20];
	int i;
	
	// open the file
	FILE *ifp = fopen(filename,"r");
	if (ifp==NULL)
	{
		fprintf(stderr,"Invalid networksimple file \"%s\".\n",filename);
		exit(-1);
	}
	
	// nlayers
	fgets(temp,20,ifp);
	fscanf(ifp,"%d\n",&net->nlayers);
	
	net->nnodes = (int *)malloc(net->nlayers*sizeof(int));
	net->rnodes = (int *)malloc(net->nlayers*sizeof(int));
	net->linear = (int *)malloc(net->nlayers*sizeof(int));
	
	// nin
	fgets(temp,20,ifp);
	fscanf(ifp,"%d\t%d\n",&net->nnodes[0],&net->linear[0]);
	net->nin = net->nnodes[0];
	
	// nhidden
	fgets(temp,20,ifp);
	for (i=1;i<net->nlayers-1;i++) fscanf(ifp,"%d\t%d\n",&net->nnodes[i],&net->linear[i]);
	
	// add bias
	for (i=0;i<net->nlayers-1;i++) (net->nnodes[i])++;
	
	// nout
	fgets(temp,20,ifp);
	fscanf(ifp,"%d\t%d\n",&net->nnodes[net->nlayers-1],&net->linear[net->nlayers-1]);
	net->nout = net->nnodes[net->nlayers-1];
	
	// recurrent flag
	fgets(temp,20,ifp);
	fscanf(ifp,"%d\n",&net->recurrent);
	
	// norbias flag
	fgets(temp,20,ifp);
	fscanf(ifp,"%d\n",&net->norbias);
	
	// classnet flag
	fgets(temp,20,ifp);
	fscanf(ifp,"%d\n",&net->classnet);
	
	// fill in recurrent nodes and count nodes and weights
	net->totnnodes = 0;
	net->totrnodes = 0;
	net->nweights = 0;
	net->nnweights = 0;
	int nrweights = 0;
	for (i=0; i<net->nlayers; i++)
	{
		if (net->recurrent && !net->norbias && i>0 && i<net->nlayers-1)
			net->rnodes[i] = net->nnodes[i] - 1;
		else
			net->rnodes[i] = 0;
		
		net->totnnodes += net->nnodes[i];
		net->totrnodes += net->rnodes[i];
		
		if (i == net->nlayers-1)
		{
			net->nnweights += net->nnodes[i] * net->nnodes[i-1];
		}
		else if (i > 0)
		{
			net->nnweights += (net->nnodes[i]-1) * net->nnodes[i-1];
			if (net->recurrent) nrweights += (int)pow((float)(net->nnodes[i]-1), 2.) + net->rnodes[i];
		}
	}
	
	net->nweights = net->nnweights + nrweights;
	net->weights = (float *)malloc(net->nweights*sizeof(float));
	
	// weights
	fgets(temp,20,ifp);
	for (i=0; i<net->nweights; i++) fscanf(ifp,"%f\n",&net->weights[i]);
	
	// close the file
	fclose(ifp);
	
	// arrange the weights
	arrangeweights(net);
}

void readOTrans(const char *filename, NetworkVariables *net)
{
	if (net->linear[net->nlayers-1] || filename==NULL)
	{
		net->scale = NULL;
		net->offset = NULL;
		net->otrans=0;
		return;
	}
	
	int i;
	
	// open the file
	FILE *ifp = fopen(filename,"r");
	if (ifp==NULL)
	{
		fprintf(stderr,"Invalid otrans file \"%s\".\n",filename);
		exit(-1);
	}
	
	// allocate arrays
	net->scale = (float *)malloc(net->nout*sizeof(float));
	net->offset = (float *)malloc(net->nout*sizeof(float));
	
	// read in size of arrays (should equal nout)
	int temp;
	fscanf(ifp,"%d\n",&temp);
	if (temp != net->nout)
	{
		fprintf(stderr,"Offset/scale size (%d) does not equal nout (%d).\n",temp,net->nout);
		exit(-1);
	}
	
	// read in scale and offset
	for (i=0;i<net->nout;i++) fscanf(ifp,"%f %f\n",&net->scale[i],&net->offset[i]);
	
	// close the file
	fclose(ifp);
	
	// set flag
	net->otrans=1;
}

void arrangeweights(NetworkVariables *net)
{
	int i,j,k,jend;
	
	// allocate memory
	net->w = (float ***)malloc((net->nlayers-1)*sizeof(float **));
	for ( i=1; i<net->nlayers; i++)
	{
		jend = i == net->nlayers-1 ? net->nnodes[i] : net->nnodes[i]-1;
		net->w[i-1] = (float **)malloc(jend*sizeof(float *));
		for (j=0; j<jend; j++) net->w[i-1][j] = (float *)malloc(net->nnodes[i-1]*sizeof(float));
	}
	
	// arrange the weights
	int p = -1;
	for( i = 1; i < net->nlayers; i++ )
	{
		jend = i == net->nlayers-1 ? net->nnodes[i] : net->nnodes[i]-1;
		for( j = 0; j < jend; j++ )
		{
			for( k = 0; k < net->nnodes[i-1]; k++ )
			{
				p++;
				net->w[i-1][j][k] = net->weights[p];
			}
		}
	}
}

void forwardOne(int ntime, float *in, float *out, NetworkVariables *net)
{
	//create temporary variables
	float *pd_in,*pd_out,*temp;
	pd_in = (float *)malloc((ntime*net->totnnodes+net->totrnodes)*sizeof(float));
	pd_out = (float *)malloc((ntime*net->totnnodes+net->totrnodes)*sizeof(float));
	temp = (float *)malloc(net->nnodes[net->nlayers-1]*sizeof(float));
	int i,j,k,m,q,p,t,r,rp,nrw;
	
	// do forward calculation
	q  = -1;
	for( t = 0; t < ntime; t++ )
	{
		p = t*net->totnnodes; // starting index of non-recurrent nodes
		rp = ntime*net->totnnodes; // starting index of recurrent nodes
		nrw = net->nnweights; // starting index of recurrent weights
		if( net->recurrent && t > 0 ) nrw += net->totrnodes;
		
		for( i = 0; i < net->nlayers; i++ )
		{
			// set up the RNN initial state
			if( net->recurrent && !net->norbias && t == 0 && i > 0 && i < net->nlayers-1 )
			{
				for( j = 0; j < net->rnodes[i]; j++ )
				{
					pd_in[rp] = 0E0;
					pd_out[rp] = 1E0;
					rp++;
				}
			}
			
			for( j = 0; j < net->nnodes[i]; j++ )
			{
				q++;
				pd_in[q] = 0E0;
				
				if( i < net->nlayers-1 && j == net->nnodes[i]-1 )
				{
					// node with the bias
					pd_out[q] = 1E0;
				}
				else
				{
					if( i > 0 )
					{
						pd_in[q] = 0E0;
						
						// contribution from non-recurrent connections
						for( k = 0; k < net->nnodes[i-1]; k++ )
							pd_in[q] += net->w[i-1][j][k] * pd_out[p+k];
							
						// contribution from recurrent connections
						if( net->recurrent && i < net->nlayers-1 && j < net->nnodes[i]-1 )
						{
							if( t == 0 )
							{
								if( !net->norbias )
								{
									pd_in[q] += net->weights[nrw];
									nrw++;
								}
							}
							else
							{
								for( k = 0; k < net->nnodes[i]-1; k++ )
								{
									pd_in[q] += net->weights[nrw] * pd_out[q-j-net->totnnodes+k];
									nrw++;
								}
							}
						}
						
						pd_out[q] = net->linear[i] ? pd_in[q] : sigmoid(pd_in[q], 1);
						
						if( i == net->nlayers-1 )
						{
							if (net->classnet)
							{
								if( j == net->nnodes[i]-1 )
								{
									r = (t+1)*net->totnnodes-net->nnodes[i];
									for( k = 0; k < net->nnodes[i]; k++ )
									{
										temp[k] = 0E0;
										for( m = 0; m < net->nnodes[i]; m++ )
											temp[k] += exp( pd_out[r+m] - pd_out[r+k] );
									}
									for( k = 0; k < net->nnodes[i]; k++ )
									{
										pd_out[r+k] = 1E0 / temp[k];
										out[k] = pd_out[r+k];
									}
								}
							}
							else
							{
								out[j] = pd_out[q];
							}
						}
					}
					else
					{
						pd_out[q] = in[j];
					}
				}
			}
			
			if( i > 0 ) p += net->nnodes[i-1];
		}
	}
	
	// apply otrans if necessary
	if (net->otrans)
		for (i=0; i<net->nout; i++)
			out[i] = out[i] * net->scale[i] + net->offset[i];
	
	// free temporary variables
	free(pd_in);
	free(pd_out);
	free(temp);
}

void clearNNsimple(NetworkVariables *net)
{
	int i,j,jend;
	
	for (i=1; i<net->nlayers; i++)
	{
		jend = i == net->nlayers-1 ? net->nnodes[i] : net->nnodes[i]-1;
		for (j=0; j<jend; j++) free(net->w[i-1][j]);
		free(net->w[i-1]);
	}
	free(net->w);
	
	free(net->linear);
	free(net->scale);
	free(net->offset);
	free(net->weights);
	free(net->nnodes);
	free(net->rnodes);
}

float sigmoid(float x, int flag)
{
	if( flag == 1 )
	{
		if( x >= 0E0 )
			return 1E0 / (1E0 + exp(-x));
		else
			return exp(x) / (1E0 + exp(x));
	}
	else if( flag == 2 )
	{
		if( x >= 0E0 )
			return exp(-x) / pow((1E0 + exp(-x)), 2.0);
		else
			return exp(x) / pow((1E0 + exp(x)), 2.0);
	}
	else
	{
		fprintf(stderr,"wrong flag passed to sigmoid function.");
		exit(-1);
	}
}
