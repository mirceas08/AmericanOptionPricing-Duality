#ifndef MLMC_CPP
#define MLMC_CPP

#include "mlmc.h"
#include <cmath>

MLMC::MLMC(double _numSims, double _l, double _M):
    numSims(_numSims), l(_l), M(_M) {}

MLMC::~MLMC() {}

arma::field<arma::mat> MLMC::stock(const double S0, const Option* myOption)
{
    double numIntervals = std::pow(M, l);
    double previousIntervals = std::pow(M, l-1);
    double r = myOption->r;
    double sigma = myOption->sigma;
    double dt = myOption->T / numIntervals;

    arma::vec dW1 = arma::randn(numSims, numIntervals);
    arma::vec dW2(numSims, previousIntervals);
    for (int i = 0; i < previousIntervals; i++) {
        dW2.col(i) = dW1.col(M*i) + dW1.col(M*i+1);
    }

    arma::mat stockFine(numSims, numIntervals+1);
    stockFine.col(0) = S0*arma::ones<arma::vec>(numSims);
    for (int i = 1; i < numIntervals+1; i++) {
        stockFine.col(i) = stockFine.col(i-1) % arma::exp((r-0.5*sigma*sigma)*dt + sigma*std::sqrt(dt)*dW1.col(i-1));
    }

    arma::mat stockCoarse(numSims, previousIntervals+1);
    stockCoarse.col(0) = S0*arma::ones<arma::vec>(numSims);
    for (int i = 1; i < previousIntervals+1; i++) {
        stockCoarse.col(i) = stockCoarse.col(i-1) % arma::exp((r-0.5*sigma*sigma)*(2*dt) + sigma*std::sqrt(dt)*dW2.col(i-1));
    }

    arma::field<arma::mat> F(2,1);
    F(0,0) = stockFine;
    F(1,0) = stockCoarse;

    return F;
}

#endif // MLMC_CPP
