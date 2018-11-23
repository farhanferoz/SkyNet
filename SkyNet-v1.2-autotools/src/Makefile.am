library_includedir=$(includedir)

SUBDIRS = mpi

AM_CXXFLAGS = -O3

if !HAVE_MPI
lib_LTLIBRARIES = libskynet-1.2.la

libskynet_1_2_la_SOURCES = NNopt.cc \
				 RBMpretrain.cc \
				 fnmatrix.cc \
				 myrand.cc \
				 skynet.cc \
				 linbcg.cc

library_include_HEADERS = NNopt.h \
						  NeuralNetwork.h \
						  TrainingData.h \
						  PredictedData.h \
						  RBMpretrain.h \
						  whiten.h \
						  myrand.h \
						  fnmatrix.h \
						  linbcg.h

SkyNet_SOURCES = main.cc
SkyNet_LDADD = -lskynet-1.2

notmpiprog = SkyNet
endif

CalPred_SOURCES = CalculatePredictions.cc

Codec_SOURCES = Codec.cc

bin_PROGRAMS = ${notmpiprog} CalPred Codec
