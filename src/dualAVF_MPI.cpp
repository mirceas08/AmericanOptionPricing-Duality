#include <iostream>
#include <algorithm>
#include <armadillo>
#include <cmath>
#include <iomanip>

#include "option/option.h"
#include "option/payoff.h"
#include "processes/processes.h"
#include "processes/euler.h"
#include "util.h"

#include "bounds/beta.h"
#include "bounds/dualAVF.h"

#include "mlmc/mlmc.h"
#include "mlmc/dualAVF_mlmc.h"

#include <mpi.h>


int main(int argc, char **argv)
{
    /* ---- MPI ---- */
    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool mpiroot = (rank == 0);

    arma::arma_rng::set_seed_random();

    double numSimsPilot;                            // number of simulations
    double numIntervals;
    double numSims;
    double numSimsBeta;
    double S0;                              // initial spot price
    double K;                               // strike
    double r;                               // risk-free rate
    double sigma;                              // initial variance
    double T;                               // maturity
    double J;
    double kappa;
    double epsilon;
    double stride;
    double numSuccessors;

    std::string dataFile = argv[1];
    std::ifstream fIN(dataFile.c_str());
    std::string line;

    if (fIN.is_open()) {
        while (std::getline(fIN, line)) {
        std::stringstream stream(line);
        std::string variable;
        std::string value;

        stream >> variable >> value;

        if (variable == "numSimsPilot")
            numSimsPilot = atoi(value.c_str());
        else if (variable == "numSimsBeta")
            numSimsBeta = atof(value.c_str());
        else if (variable == "S0")
            S0 = atof(value.c_str());
        else if (variable == "strike")
            K = atof(value.c_str());
        else if (variable == "r")
            r = atof(value.c_str());
        else if (variable == "J")
            J = atof(value.c_str());
        else if (variable == "sigma")
            sigma = atof(value.c_str());
        else if (variable == "maturity")
            T = atof(value.c_str());
        else if (variable == "kappa")
            kappa = atof(value.c_str());
        else if (variable == "epsilon")
            epsilon = atof(value.c_str());
        else if (variable == "stride")
            stride = atof(value.c_str());
        else if (variable == "numSuccessors")
            numSuccessors = atof(value.c_str());
        }
    }
    else {
        std::cout << "Error opening file" << std::endl;
        return -1;
    }

    /* ------------------------ Set option, payoff and discretization scheme objects ------------------------ */
    PayOff* myPayoff = new PayOffPut(K);
    Option* myOption = new Option(K, r, sigma, T, myPayoff);
    Processes* scheme = new Euler(myOption);

    if (mpiroot) {
        std::cout << "=================== Option parameters ===================" << std::endl;
        std::cout << "Spot price: " << S0 << std::endl;
        std::cout << "Strike: " << K << std::endl;
        std::cout << "Risk free rate: " << r << std::endl;
        std::cout << "Volatility: " << sigma << std::endl;
        std::cout << "Maturity: " << T << std::endl;
        std::cout << "================= Dual-AVF parameters =================" << std::endl;
        std::cout << "Number of basis functions: " << J << std::endl;
        std::cout << "Number of successors in Dual AVF: " << numSuccessors << std::endl;
        std::cout << "================= MLMC parameters =================" << std::endl;
        std::cout << "Number of simulations for pilot run: " << numSimsPilot << std::endl;
        std::cout << "Kappa: " << kappa << std::endl;
        std::cout << "Number of levels: " << size << std::endl;
        std::cout << "Stride: " << stride << std::endl;
    }

    // coefficient estimation

    // Timer
    arma::wall_clock timer;
    timer.tic();

    double l = rank + stride;
    numIntervals = std::pow(kappa, l);
    double dt = myOption->T / numIntervals;
    double C = 100000 * std::pow(kappa, size + stride);

    /* ***************** Beta estimation ******************** */
    Beta* betaEstimation = new Beta(J);
    arma::mat stock(numSimsBeta, numIntervals+1);
    for (int i = 0; i < numSimsBeta; i++) {
        stock.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
    }
    arma::mat beta = betaEstimation->computeBeta(myOption, stock);

    /* ***************** MLMC pilot run ******************** */
    DualAVF* myDualAVF = new DualAVF(numSuccessors);

    stock.resize(numSimsPilot, numIntervals+1);
    for (int i = 0; i < numSimsPilot; i++) {
        stock.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
    }
    arma::vec AVFprice = myDualAVF->computeDualAVF(myOption, stock, beta);
    double variancePilot = arma::var(AVFprice) / numSimsPilot;

    arma::vec varianceFromPilot(size);
    arma::vec dtVec(size);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&variancePilot, 1, MPI_DOUBLE, varianceFromPilot.begin(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&dt, 1, MPI_DOUBLE, dtVec.begin(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    arma::vec Nl(size);
    if (mpiroot) {
        arma::vec denominatorVec = arma::sqrt(varianceFromPilot / dtVec);
        double denominator = arma::sum(denominatorVec);

        for (int i = 0; i < size; i++) {
            double squareRoot = std::sqrt(varianceFromPilot(i) * dtVec(i));
            Nl(i) = std::ceil((C*squareRoot) / (denominator));
        }

//        for (int i = 0; i < size; i++) {
//            double squareRoot = varianceFromPilot(i) * dtVec(i);
//            Nl(i) = std::ceil((2*std::pow(epsilon, -2) * squareRoot) * denominator);
//        }
    }
    MPI_Scatter(Nl.begin(), 1, MPI_DOUBLE, &numSims, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    /* ***************** MLMC estimator ******************** */
    double priceMLMC;
    MLMC* myMLMC = new MLMC(numSims, l, kappa);
    DualAVF_mlmc* myDualAVF_mlmc = new DualAVF_mlmc(l, kappa, numSuccessors);
    double MLMCestimator;
    double variance;
    double correlation = 0.0;

    if (mpiroot) {
        stock.resize(numSims, numIntervals+1);
        for (int i = 0; i < numSims; i++) {
            stock.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
        }
        AVFprice = myDualAVF->computeDualAVF(myOption, stock, beta);
        priceMLMC = arma::mean(AVFprice);
        variance = arma::var(AVFprice) / numSims;
    }
    else {
        arma::field<arma::mat> S = myMLMC->stock(S0, myOption);
        arma::field<arma::mat> betaML(2,1);
        betaML(0,0) = betaEstimation->computeBeta(myOption, S(0,0));
        betaML(1,0) = betaEstimation->computeBeta(myOption, S(1,0));
        S = myMLMC->stock(S0, myOption);

        arma::vec mlmcPrice = myDualAVF_mlmc->computeDualAVF_mlmc(myOption, S, betaML);
        priceMLMC = arma::mean(mlmcPrice);
        variance = arma::var(mlmcPrice) / numSims;
        correlation = myDualAVF_mlmc->correlation;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&priceMLMC, &MLMCestimator, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    arma::vec MLMCprices(size);
    arma::vec MLMCvariances(size);
    arma::vec MLMCcorrelations(size);
    MPI_Gather(&priceMLMC, 1, MPI_DOUBLE, MLMCprices.begin(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&variance, 1, MPI_DOUBLE, MLMCvariances.begin(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&correlation, 1, MPI_DOUBLE, MLMCcorrelations.begin(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpiroot) {
        std::cout << "=================== Multilevel Monte Carlo ===================" << std::endl;
        std::cout << "Level" << setw(12) << "# paths" << setw(14) << "Timesteps" << setw(12) << "Price" << setw(23) << "Variance"
                 << setw(23) << "Correlation" << std::endl;
        std::cout << std::string(12*3 + 7*8, '-') << std::endl;

        for (int i = 0; i < size; i++) {
            printElement(i + stride, 12);
            printElement(Nl(i), 12);
            printElement((myOption->T) / dtVec(i), 14);
            printElement(MLMCprices(i), 20);
            printElement(MLMCvariances(i), 20);
            printElement(MLMCcorrelations(i), 20);
            std::cout << std::endl;
        }
        std::cout << "*************************************" << std::endl;
        std::cout << "MLMC estimator: " << MLMCestimator << std::endl;
        std::cout << "MLMC total variance: " << arma::sum(MLMCvariances) << std::endl;
    }

    delete myPayoff;
    delete myOption;
    delete scheme;
    delete betaEstimation;
    delete myDualAVF;
    delete myMLMC;
    delete myDualAVF_mlmc;

    MPI_Finalize();
    return 0;
}
