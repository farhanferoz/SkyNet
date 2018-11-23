#GSL directory
GSLDIR = /opt/local
#GSLDIR = /usr/local/other/SLES11.1/gsl/1.15/intel-12.1.0.233

#CBlas directory
CBLASDIR = /usr/local/

#SERIAL
ICC = $(CC)
ICXX = $(CXX)
ICFLAGS +=  -I. -I$(GSLDIR)/include/ -O3
ICXXFLAGS +=  -I. -I$(GSLDIR)/include/ -O3

#MAC OSX
CFLAGS += -I. -O3
CXXFLAGS += -I. -O3
LAPACKLIB = -llapack -latlas

#PARALLEL on Darwin
#CC = mpicc
#CXX = mpicxx
#CFLAGS += -I. -O3 -ipo -xHost -DPARALLEL
#CXXFLAGS += -I. -O3 -ipo -xHost -DPARALLEL
#LAPACKLIB = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread

#PARALLEL on Universe
#CC = icc -lmpi -vec-report1
#CXX = icpc -lmpi -vec-report1
#CFLAGS += -O3 -xHost -DPARALLEL -I$(CBLASDIR)
#CXXFLAGS = $(CFLAGS)
#LAPACKLIB = -mkl=sequential

#PARALLEL on Discover
#CC = mpicc
#CXX = mpiicpc
#CFLAGS += -I. -I$(GSLDIR)/include/ -O3 -DPARALLEL
#CXXFLAGS += -I. -I$(GSLDIR)/include/ -O3 -DPARALLEL
#LAPACKLIB = -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm -lmkl_core -lmkl_lapack95_lp64

AR = ar r
LINKLIB = ld -shared
LIBS = -lstdc++ -lcblas
#-mkl=parallel

LIBOBJFILES = linbcg.o NNopt.o RBMpretrain.o fnmatrix.o myrand.o skynet.o
SNOBJFILES = main.o
TESTRBMOBJFILES = TestRBM.o myrand.o

default: libskynet.a skynet

all: libskynet.a skynet pred codec

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $*.cc

%.o: %.c
	$(CC) $(CFLAGS) -c $*.c

libskynet.so: $(LIBOBJFILES) 
	$(LINKLIB) -o $(LIBS) $@ $^ 

libskynet.a: $(LIBOBJFILES) 
	$(AR) $@ $^ 

skynet: $(SNOBJFILES)
	$(CXX) $(CXXFLAGS) -o ../SkyNet $(SNOBJFILES) -L./ -lskynet $(LIBS)

testrbm: $(TESTRBMOBJFILES)
	$(CXX) $(CXXFLAGS) -o TestRBM $(TESTRBMOBJFILES) $(LIBS)

pred:
	$(ICXX) $(ICXXFLAGS) -o ../CalPred CalculatePredictions.cc

codec:
	$(ICXX) $(ICXXFLAGS) -o ../Codec Codec.cc

tidy:
	rm -f *.o *.mod *~ *.btr

clean:
	rm -f libskynet.* *.o *.mod ../SkyNet ../CalPred ../Codec
