#ifndef DUALAVF_MLMC_H
#define DUALAVF_MLMC_H

#include <armadillo>
#include "../option/option.h"

class DualAVF_mlmc
{
public:
    double correlation;
private:
    double l;
    double M;
    double numSuccessors;
public:
    DualAVF_mlmc(double _l, double _M, double _numSuccessors);
    virtual ~ DualAVF_mlmc();

    arma::vec computeDualAVF_mlmc(Option*  myOption, const arma::field<arma::mat> &stock, const arma::field<arma::mat> &beta);
};


#endif // DUALAVF_MLMC_H
