#ifndef DUALSR_MLMC_H
#define DUALSR_MLMC_H

#include <armadillo>
#include "../option/option.h"

class DualSR_mlmc
{
public:
    double correlation;
private:
    double l;
    double kappa;
    double subSims;
public:
    DualSR_mlmc(double _l, double _kappa, double _subSims);
    virtual ~ DualSR_mlmc();

    arma::vec computeDualSR_mlmc(Option*  myOption, const arma::field<arma::mat> &stock, const arma::field<arma::mat> &beta);
};


#endif // DUALSR_MLMC_H
