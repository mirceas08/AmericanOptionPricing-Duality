#ifndef DUALSR_MLMC_CPP
#define DUALSR_MLMC_CPP

#include "dualSR_mlmc.h"
#include "../bounds/dualSR.h"
#include "../util.h"

DualSR_mlmc::DualSR_mlmc(double _l, double _kappa, double _subSims):
    l(_l), kappa(_kappa), subSims(_subSims) {}

DualSR_mlmc::~DualSR_mlmc() {}

arma::vec DualSR_mlmc::computeDualSR_mlmc(Option*  myOption, const arma::field<arma::mat> &stock, const arma::field<arma::mat> &beta)
{
    DualSR* myDualSR = new DualSR(subSims);
    arma::vec finePrice = myDualSR->computeDualSR(myOption, stock(0,0), beta(0,0));
    arma::vec coarsePrice = myDualSR->computeDualSR(myOption, stock(1,0), beta(1,0));

    correlation = arma::as_scalar(arma::cor(finePrice, coarsePrice));

    delete myDualSR;
    return finePrice - coarsePrice;
}

#endif // DUALSR_MLMC_CPP
