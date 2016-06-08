#ifndef DUALAVF_MLMC_CPP
#define DUALAVF_MLMC_CPP

#include "dualAVF_mlmc.h"
#include "../bounds/dualAVF.h"
#include "../util.h"

DualAVF_mlmc::DualAVF_mlmc(double _l, double _M, double _numSuccessors):
    l(_l), M(_M), numSuccessors(_numSuccessors) {}

DualAVF_mlmc::~DualAVF_mlmc() {}

arma::vec DualAVF_mlmc::computeDualAVF_mlmc(Option*  myOption, const arma::field<arma::mat> &stock, const arma::field<arma::mat> &beta)
{
    DualAVF* myDualAVF = new DualAVF(numSuccessors);
    arma::vec finePrice = myDualAVF->computeDualAVF(myOption, stock(0,0), beta(0,0));
    arma::vec coarsePrice = myDualAVF->computeDualAVF(myOption, stock(1,0), beta(1,0));

    correlation = arma::as_scalar(arma::cor(finePrice, coarsePrice));

    delete myDualAVF;
    return finePrice - coarsePrice;
}

#endif // DUALAVF_MLMC_CPP
