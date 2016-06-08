include ../helpers.mk

## Generic rules:
.SUFFIXES:          # get rid of that annoying Modula rule
.SUFFIXES: .cpp .h .o

OBJECTS			= beta.o LSM.o dualAVF.o dualSR.o
#
# Rules
#

all: $(OBJECTS)

.cpp.o:
	$(CC) -c $*.cpp $(INCLUDES) $(LIBS) $(CFLAGS)
#
# Dependencies
#
#
beta.o: beta.h ../util.h
#
LSM.o: LSM.h ../util.h
#
dualAVF.o: dualAVF.h ../util.h
#
dualSR.o: dualSR.h ../util.h
#

clean:
	\rm *.o *~
