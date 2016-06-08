include ../helpers.mk


OBJECTS	= mlmc.o LSM_mlmc.o dualAVF_mlmc.o dualSR_mlmc.o dualAVF_martingale.o dualSR_martingale.o
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
mlmc.o: ../util.h
#
LSM_mlmc.o: ../util.h ../option/option.h ../bounds/LSM.h 
#
dualAVF_mlmc.o: ../util.h ../option/option.h ../bounds/dualAVF.h
#
dualSR_mlmc.o: ../util.h ../option/option.h ../bounds/dualSR.h
#
dualAVF_martingale.o: ../util.h ../option/option.h ../bounds/dualAVF.h
#
dualSR_martingale.o: ../util.h ../option/option.h ../bounds/dualSR.h
#

clean:
	\rm *.o *~
