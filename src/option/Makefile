include ../helpers.mk

## Generic rules:
.SUFFIXES:          # get rid of that annoying Modula rule
.SUFFIXES: .cpp .h .o

OBJECTS			= option.o payoff.o
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
payoff.o: payoff.h
#
option.o: option.h payoff.h
#


clean:
	\rm *.o *~
