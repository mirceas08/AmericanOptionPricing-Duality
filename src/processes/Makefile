include ../helpers.mk


OBJECTS	= processes.o euler.o
#
# Rules
#
.cpp.o:
	$(CC) -c $*.cpp $(INCLUDES) $(LIBS) $(CFLAGS)
#
# Dependencies
#
#
all: $(OBJECTS)
#
gbmDiscr.o: processes.h ../option/option.h
#
gbmEuler.o: euler.h processes.h ../option/option.h
#


clean:
	\rm *.o *~
