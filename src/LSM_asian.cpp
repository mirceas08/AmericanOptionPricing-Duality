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
#include "bounds/LSM.h"


int main(int argc, char **argv)
{
    arma::arma_rng::set_seed_random();

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
    PayOff* myPayoff = new AsianPut(K);
    Option* myOption = new Option(K, r, sigma, T, myPayoff);
    Processes* scheme = new Euler(myOption);

    std::cout << "=================== Option parameters ===================" << std::endl;
    std::cout << "Spot price: " << S0 << std::endl;
    std::cout << "Strike: " << K << std::endl;
    std::cout << "Risk free rate: " << r << std::endl;
    std::cout << "Volatility: " << sigma << std::endl;
    std::cout << "Maturity: " << T << std::endl;
    std::cout << "================= LSM parameters =================" << std::endl;
    std::cout << "Number of basis functions: " << J << std::endl;
    std::cout << "================= Monte Carlo parameters =================" << std::endl;
    std::cout << "Numer of time steps: " << numIntervals << std::endl;
    std::cout << "Number of simulations for beta estimation: " << numSimsBeta << std::endl;
    std::cout << "Number of simulations: " << numSims << std::endl;


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
    LSM* myLSM = new LSM();

    stock.resize(numSims, numIntervals+1);
    for (int i = 0; i < numSims; i++) {
        stock.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
    }
    arma::vec LSMprice = myLSM->computeLSM(myOption, stock, beta);
    double priceMLMC = arma::mean(LSMprice);
    double variance = arma::var(LSMprice) / numSims;


    std::cout << "=================== Standard Monte Carlo ===================" << std::endl;
    std::cout << "Standard Monte Carlo estimator: " << priceMLMC << std::endl;
    std::cout << "Standard Monte Carlo variance: " << variance << std::endl;


    delete myPayoff;
    delete myOption;
    delete scheme;
    delete betaEstimation;
    delete myLSM;

    return 0;
}
