#ifndef MLMC_H
#define MLMC_H

#include "../option/option.h"

class MLMC
{
private:
    double numSims;
    double l;
    double M;
public:
    MLMC(double _numSims, double _l, double _M);
    virtual ~MLMC();

    arma::field<arma::mat> stock(const double S0, const Option* myOption);
};


#endif // MLMC_H
