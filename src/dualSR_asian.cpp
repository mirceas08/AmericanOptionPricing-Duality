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
#include "bounds/dualSR.h"


int main(int argc, char **argv)
{
    arma::arma_rng::set_seed_random();

    double numIntervals;
    double numSims;
    double numSimsBeta;
    double subSims;
    double S0;                              // initial spot price
    double K;                               // strike
    double r;                               // risk-free rate
    double sigma;                              // initial variance
    double T;                               // maturity
    double J;
    double kappa;
    double epsilon;

    std::string dataFile = argv[1];
    std::ifstream fIN(dataFile.c_str());
    std::string line;

    if (fIN.is_open()) {
        while (std::getline(fIN, line)) {
        std::stringstream stream(line);
        std::string variable;
        std::string value;

        stream >> variable >> value;

        if (variable == "numSimsBeta")
            numSimsBeta = atof(value.c_str());
        else if (variable == "S0")
            S0 = atof(value.c_str());
        else if (variable == "timeSteps")
            numIntervals = atof(value.c_str());
        else if (variable == "numSims")
            numSims = atof(value.c_str());
        else if (variable == "subSims")
            subSims = atof(value.c_str());
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
        }
    }
    else {
        std::cout << "Error opening file" << std::endl;
        return -1;
    }

    /* ------------------------ Set option, payoff and discretization scheme objects ------------------------ */
    PayOff* myPayoff = new PayOffCall(K);
    Option* myOption = new Option(K, r, sigma, T, myPayoff);
    Processes* scheme = new Euler(myOption);

    std::cout << "=================== Option parameters ===================" << std::endl;
    std::cout << "Spot price: " << S0 << std::endl;
    std::cout << "Strike: " << K << std::endl;
    std::cout << "Risk free rate: " << r << std::endl;
    std::cout << "Volatility: " << sigma << std::endl;
    std::cout << "Maturity: " << T << std::endl;
    std::cout << "================= Dual SR parameters =================" << std::endl;
    std::cout << "Number of basis functions: " << J << std::endl;
    std::cout << "================= Monte Carlo parameters =================" << std::endl;
    std::cout << "Numer of time steps: " << numIntervals << std::endl;
    std::cout << "Number of simulations for beta estimation: " << numSimsBeta << std::endl;
    std::cout << "Number of simulations: " << numSims << std::endl;
    std::cout << "Number of subsimulations: " << subSims << std::endl;


    // coefficient estimation

    // Timer
    arma::wall_clock timer;
    timer.tic();
    double dt = myOption->T / numIntervals;

    /* ***************** Beta estimation ******************** */
    Beta* betaEstimation = new Beta(J);
    arma::mat stock(numSimsBeta, numIntervals+1);
    for (int i = 0; i < numSimsBeta; i++) {
        stock.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
    }
    arma::mat beta = betaEstimation->computeBeta(myOption, stock);

    /* ***************** MLMC pilot run ******************** */
    DualSR* myDualSR = new DualSR(subSims);

    stock.resize(numSims, numIntervals+1);
    for (int i = 0; i < numSims; i++) {
        stock.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
    }

    arma::mat average(numSims, numIntervals+1);
    for (int i = 0; i < average.n_rows; i++) {
        for (int j = 0; j < average.n_cols; j++) {
            average(i,j) = arma::accu(stock.row(i).head(j+1)) / (j+1);
        }
    }

    arma::vec SRprice = myDualSR->computeDualSR(myOption, average, beta);
    double priceMLMC = arma::mean(SRprice);
    double variance = arma::var(SRprice) / numSims;


    std::cout << "=================== Standard Monte Carlo ===================" << std::endl;
    std::cout << "Standard Monte Carlo estimator: " << priceMLMC << std::endl;
    std::cout << "Standard Monte Carlo variance: " << variance << std::endl;


    delete myPayoff;
    delete myOption;
    delete scheme;
    delete betaEstimation;
    delete myDualSR;

    return 0;
}
