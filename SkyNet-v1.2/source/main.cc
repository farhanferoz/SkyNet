#include "NNopt.h"

int main(int argc, char **argv) {
	//time_t c1=time(NULL),c2;
	
	int myid = 0;
#ifdef PARALLEL
 	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
#endif
	
	if (argc!=2) {
		if (myid==0)
			fprintf(stderr,"Please provide an input file of run settings. Run with --help/-h to see options or --version/-v to see version information.\n");
		exit(-1);
	} else if (argc==2 && (!strcmp(argv[1],"--help") || !strcmp(argv[1],"-h"))) {
		if (myid==0) PrintHelp();
#ifdef PARALLEL
		MPI_Finalize();
#endif
		return 0;
	} else if (argc==2 && (!strcmp(argv[1],"--version") || !strcmp(argv[1],"-v"))) {
		if (myid==0) PrintVersion();
#ifdef PARALLEL
		MPI_Finalize();
#endif
		return 0;
	}
	
	size_t nlayers,nnodes[50];
	bool resume=true;
	char inroot[200],outroot[200];
	strcpy(inroot,"data");
	strcpy(outroot,"data");
	ReadInputFile1(argv[1],inroot,outroot,&resume);
	
	TrainNetwork(argv[1],inroot,outroot,&nlayers,&nnodes[0],resume,false);
	
	//c2=time(NULL);
	//if(myid==0) printf("\nTotal time taken is %g seconds.\n",difftime(c2,c1));
	
#ifdef PARALLEL
	MPI_Finalize();
#endif
	
	return 0;
}
