##### Insert here location of headers and libraries
###################################################

# headers
ARMADILLO_INC	= /media/mircea/Stuff/Mircea/Programming/C++/libraries/armadillo-6.100.1/include

# libraries
OPENBLAS_LIB 	= /usr/lib/openblas/lib
LAPACK_LIB		= /usr/lib/lapack

###################################################
###################################################

CC				= g++
MPICC			= mpic++
DEBUG			= -DARMA_NO_DEBUG -DNDEBUG
PROG            = main LSM dual_approx dual_stopping primal_dual
INCLUDES        = -I $(ARMADILLO_INC)
LIBS            = -L $(OPENBLAS_LIB) -lopenblas -L $(LAPACK_LIB) -llapack
CFLAGS        	= -O3 -std=c++11 -fopenmp -march=native $(DEBUG)

## Generic rules:
#.SUFFIXES:          # get rid of that annoying Modula rule
.SUFFIXES: .cpp .h .o


#
# Rules
#
#.cpp.o:
#	$(CC) -c $*.cpp $(INCLUDES) $(LIBS) $(CFLAGS)

