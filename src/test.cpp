#include <iostream>
#include <algorithm>
#include <armadillo>
#include <cmath>

#include "option/option.h"
#include "option/payoff.h"
#include "processes/processes.h"
#include "processes/euler.h"

#include "bounds/beta.h"
#include "bounds/LSM.h"
#include "bounds/dualAVF.h"
#include "bounds/dualSR.h"

#include "mlmc/mlmc.h"
#include "mlmc/LSM_mlmc.h"
#include "mlmc/dualAVF_mlmc.h"


int main(int argc, char **argv)
{
    arma::arma_rng::set_seed_random();

    double numSims;                            // number of simulations
    double numSimsBeta;
    double numIntervals;                       // number of time steps
    double S0;                              // initial spot price
    double K;                               // strike
    double r;                               // risk-free rate
    double sigma;                              // initial variance
    double T;                               // maturity
    std::string discretizationScheme;       // discretization scheme
    double J;
    double numSuccessors;
    double subSims;

    std::string dataFile = argv[1];
    std::ifstream fIN(dataFile.c_str());
    std::string line;

    if (fIN.is_open()) {
        while (std::getline(fIN, line)) {
        std::stringstream stream(line);
        std::string variable;
        std::string value;

        stream >> variable >> value;

        if (variable == "numSims")
            numSims = atoi(value.c_str());
        else if (variable == "timeSteps")
            numIntervals = atoi(value.c_str());
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
        else if (variable == "numSuccessors")
            numSuccessors = atof(value.c_str());
        else if (variable == "subSims")
            subSims = atof(value.c_str());
        else if (variable == "discretizationScheme")
            discretizationScheme = value;
        }
    }
    else {
        std::cout << "Error opening file" << std::endl;
        return -1;
    }

    transform(discretizationScheme.begin(), discretizationScheme.end(), discretizationScheme.begin(), ::toupper);

    /* ------------------------ Set option, payoff and discretization scheme objects ------------------------ */
    PayOff* myPayoff = new AsianCall(K);
    Option* myOption = new Option(K, r, sigma, T, myPayoff);
    Processes* scheme = new Euler(myOption);
    double dt = myOption->T / numIntervals;


    std::cout << "=================== Option parameters ===================" << std::endl;
    std::cout << "Spot price: " << S0 << std::endl;
    std::cout << "Strike: " << K << std::endl;
    std::cout << "Risk free rate: " << r << std::endl;
    std::cout << "Volatility: " << sigma << std::endl;
    std::cout << "Maturity: " << T << std::endl;
    std::cout << "================= Simulation parameters =================" << std::endl;
    std::cout << "Number of simulations: " << numSims << std::endl;
    std::cout << "Number of successors for Duality AVF: " << numSuccessors << std::endl;
    std::cout << "Number of subsimulations for Duality SR: " << subSims << std::endl;
    std::cout << "Number of time steps: " << numIntervals << std::endl;
    std::cout << "Number of basis functions: " << J << std::endl;



    // coefficient estimation

    // Timer
    arma::wall_clock timer;
    timer.tic();
//
//    arma::mat stock(numSimsBeta, numIntervals+1);
//    for (int i = 0; i < numSimsBeta; i++) {
//        stock.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
//    }
//
//    Beta* myBeta = new Beta(J);
//    arma::mat beta = myBeta->computeBeta(myOption, stock);
//
//    stock.resize(numSims, numIntervals+1);
//    for (int i = 0; i < numSims; i++) {
//        stock.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
//    }
//
//    LSM* myLSM = new LSM();
//    arma::vec LSMprice = myLSM->computeLSM(myOption, stock, beta);
//    std::cout << std::endl;
//    std::cout << "LSM price: " << arma::mean(LSMprice) << std::endl;
//
//    DualAVF* myDualAVF = new DualAVF(numSuccessors);
//    arma::vec AVFprice = myDualAVF->computeDualAVF(myOption, stock, beta);
//    std::cout << "DualAVF price: " << arma::mean(AVFprice) << std::endl;
//
//    DualSR* myDualSR = new DualSR(subSims);
//    arma::vec SRprice = myDualSR->computeDualSR(myOption, stock, beta);
//    std::cout << "DualSR price: " << arma::mean(SRprice) << std::endl;
//
//    delete myPayoff;
//    delete myOption;
//    delete scheme;
//    delete myBeta;
//    delete myLSM;
//    delete myDualAVF;


//    double M = 2;
//    arma::vec Ml(3);
//    Ml(0) = 32;
//    Ml(1) = 64;
//    Ml(2) = 128;
//    double numSims1 = 10000;
//    double numSims2 = 5000;
//    double numSims3 = 1000;
//
//    arma::mat stock(numSims1, Ml(0)+1);
//    for (int i = 0; i < numSims1; i++) {
//        stock.row(i) = scheme->calculateStockPath(S0, Ml(0)+1);
//    }
//    Beta* myBeta = new Beta(J);
//    arma::mat beta = myBeta->computeBeta(myOption, stock);
//
//    for (int i = 0; i < numSims1; i++) {
//        stock.row(i) = scheme->calculateStockPath(S0, Ml(0)+1);
//    }
//
//    LSM* myLSM = new LSM();
//    arma::vec LSMprice = myLSM->computeLSM(myOption, stock, beta);
//    DualAVF* myDualAVF = new DualAVF(numSuccessors);
//    arma::vec AVFprice = myDualAVF->computeDualAVF(myOption, stock, beta);
//
//    MLMC* myMLMC = new MLMC(numSims2, 6, M);
//    arma::field<arma::mat> S = myMLMC->stock(S0, myOption);
//    arma::field<arma::mat> betaML(2,1);
//    betaML(0,0) = myBeta->computeBeta(myOption, S(0,0));
//    betaML(1,0) = myBeta->computeBeta(myOption, S(1,0));
//    S = myMLMC->stock(S0, myOption);
//
//    LSM_mlmc* myLSM_mlmc = new LSM_mlmc(6, M);
//    arma::vec mlmcPrice = myLSM_mlmc->computeLSM_mlmc(myOption, S, betaML);
//    DualAVF_mlmc* myAVF_mlmc = new DualAVF_mlmc(6, M, numSuccessors);
//    arma::vec mlmcPriceAVF = myAVF_mlmc->computeDualAVF_mlmc(myOption, S, betaML);
//
//    std::cout << "****************** LSM *****************" << std::endl;
//    std::cout << "P_32: " << arma::mean(LSMprice) << std::endl;
//    std::cout << "P_64 - P_32: " << arma::mean(mlmcPrice) << std::endl;
//    std::cout << "Option price: " << arma::mean(LSMprice) + arma::mean(mlmcPrice) << std::endl;
//
//    std::cout << "****************** Duality AVF *****************" << std::endl;
//    std::cout << "P_32: " << arma::mean(AVFprice) << std::endl;
//    std::cout << "P_64 - P_32: " << arma::mean(mlmcPriceAVF) << std::endl;
//    std::cout << "Option price: " << arma::mean(AVFprice) + arma::mean(mlmcPriceAVF) << std::endl;
//
//    delete myBeta;
//    delete myLSM;
//    delete myMLMC;
//    delete myLSM_mlmc;
//    delete myAVF_mlmc;

    arma::mat test(2,3);
    test(0,0) = 2;
    test(0,1) = 3;
    test(0,2) = 7;
    test(1,0) = 4;
    test(1,1) = 5;
    test(1,2) = 9;

    arma::mat succ(2,2);
    succ(0,0) = 4;
    succ(0,1) = 3;
    succ(1,0) = 6;
    succ(1,1) = 1;

    arma::mat Zs = arma::zeros<arma::mat>(2, 2);
    arma::mat tempStock = test.head_cols(1);
    for (int k = 0; k < 2; k++) {
        arma::mat temp = arma::join_horiz(tempStock, succ.col(k));
        temp = arma::join_horiz(arma::zeros(2), temp);
        arma::mat temporaryPayoff = myOption->payoff->operator()(temp);
        std::cout << temporaryPayoff << std::endl;
        Zs.col(k) = temporaryPayoff.tail_cols(1);
    }

    std::cout << Zs << std::endl;

    arma::mat average(2, 3);
    for (int i = 0; i < average.n_rows; i++) {
        for (int j = 0; j < average.n_cols; j++) {
            average(i,j) = arma::mean(test.row(i).head(j+2));
        }
    }

    arma::mat payoff(2, 2);
    for (int i = 0; i < 2; i++) {
        payoff.col(i) = arma::max(average.col(i+1) - K, arma::zeros<arma::vec>(numSims));
    }


    delete myPayoff;
    delete myOption;
    delete scheme;
    return 0;
}
