include helpers.mk

.PHONY: test main LSM LSM_asian dualAVF dualAVF_asian dualSR_asian LSM_MPI dualAVF_MPI dualAVF_MARTINGALE dualSR_MPI dualSR_MARTINGALE hedging

PROG = test main LSM dualAVF dualAVF_asian dualSR_asian LSM_MPI dualAVF_MPI dualAVF_martingale dualSR_martingale dualSR_MPI LSM_asian hedging

#
#Dependencies
#
#
OBJECTS = processes/processes.o processes/euler.o option/option.o option/payoff.o

MAIN_OBJECTS = 
main: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	$(CC) -o main main.cpp $(OBJECTS) $(MAIN_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)
	
TEST_OBJECTS = 
test: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	$(CC) -o test test.cpp $(OBJECTS) $(TEST_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)

LSM_OBJECTS = bounds/beta.o bounds/LSM.o
LSM: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	$(CC) -o LSM LSM.cpp $(OBJECTS) $(LSM_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)
	
LSM_ASIAN_OBJECTS = bounds/beta.o bounds/LSM.o
LSM_asian: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	$(CC) -o LSM_asian LSM_asian.cpp $(OBJECTS) $(LSM_ASIAN_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)
	
LSM_MPI_OBJECTS = bounds/beta.o bounds/LSM.o mlmc/mlmc.o mlmc/LSM_mlmc.o
LSM_MPI: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	+$(MAKE) -C mlmc
	$(MPICC) -o LSM_MPI LSM_MPI.cpp $(OBJECTS) $(LSM_MPI_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)

dualAVF_OBJECTS = bounds/beta.o bounds/dualAVF.o
dualAVF: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	$(CC) -o dualAVF dualAVF.cpp $(OBJECTS) $(dualAVF_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)
	
HEDGING_OBJECTS = bounds/beta.o bounds/dualAVF.o
hedging: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	$(CC) -o hedging hedging.cpp $(OBJECTS) $(HEDGING_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)
	
dualAVF_ASIAN_OBJECTS = bounds/beta.o bounds/dualAVF.o
dualAVF_asian: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	$(CC) -o dualAVF_asian dualAVF_asian.cpp $(OBJECTS) $(dualAVF_ASIAN_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)	
		
dualAVF_MPI_OBJECTS = bounds/beta.o bounds/dualAVF.o mlmc/mlmc.o mlmc/dualAVF_mlmc.o
dualAVF_MPI: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	+$(MAKE) -C mlmc
	$(MPICC) -o dualAVF_MPI dualAVF_MPI.cpp $(OBJECTS) $(dualAVF_MPI_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)
	
dualAVF_MARTINGALE_OBJECTS = bounds/beta.o bounds/dualAVF.o mlmc/mlmc.o mlmc/dualAVF_martingale.o
dualAVF_MARTINGALE: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	+$(MAKE) -C mlmc
	$(MPICC) -o dualAVF_martingale dualAVF_martingale.cpp $(OBJECTS) $(dualAVF_MARTINGALE_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)


dualSR_OBJECTS = bounds/beta.o bounds/dualSR.o
dualSR: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	$(CC) -o dualSR dualSR.cpp $(OBJECTS) $(dualSR_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)
	

dualSR_ASIAN_OBJECTS = bounds/beta.o bounds/dualSR.o
dualSR_asian: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	$(CC) -o dualSR_asian dualSR_asian.cpp $(OBJECTS) $(dualSR_ASIAN_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)


dualSR_MPI_OBJECTS = bounds/beta.o bounds/dualSR.o mlmc/mlmc.o mlmc/dualSR_mlmc.o
dualSR_MPI: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	+$(MAKE) -C mlmc
	$(MPICC) -o dualSR_MPI dualSR_MPI.cpp $(OBJECTS) $(dualSR_MPI_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)
	
dualSR_MARTINGALE_OBJECTS = bounds/beta.o bounds/dualSR.o mlmc/mlmc.o mlmc/dualSR_martingale.o
dualSR_MARTINGALE: 
	+$(MAKE) -C processes
	+$(MAKE) -C option
	+$(MAKE) -C bounds
	+$(MAKE) -C mlmc
	$(MPICC) -o dualSR_martingale dualSR_martingale.cpp $(OBJECTS) $(dualSR_MARTINGALE_OBJECTS) $(INCLUDES) $(LIBS) $(CFLAGS)

clean:
	\rm *.o *~ $(PROG)

	
