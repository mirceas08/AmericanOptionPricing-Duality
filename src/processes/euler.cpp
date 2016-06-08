#ifndef EULER_CPP
#define EULER_CPP

#include "euler.h"


/* ------------------- HestonEulerReflection ----------------------- */

Euler::Euler(Option* _myOption):
    Processes(_myOption) {}

Euler::~Euler() {}

arma::vec Euler::calculateStockPath(const double S0, const double numIntervals)
{
    arma::vec stockPath(numIntervals);
    double dt = myOption->T / (numIntervals-1);

    stockPath(0) = S0;
    for (int i = 1; i < stockPath.size(); i++) {
        double sigma = myOption->sigma;
        stockPath(i) = stockPath(i-1) * std::exp( (myOption->r - 0.5 * sigma *sigma) * dt + sigma*std::sqrt(dt)*arma::randn());
    }

    return stockPath;
}


#endif // EULER_CPP
