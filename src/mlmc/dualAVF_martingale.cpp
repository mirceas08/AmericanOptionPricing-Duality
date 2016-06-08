#ifndef DUALAVF_MARTINGALE_CPP
#define DUALAVF_MARTINGALE_CPP

#include <cmath>
#include "dualAVF_martingale.h"
#include "../bounds/dualAVF.h"
#include "../util.h"

DualAVF_martingale::DualAVF_martingale(double _l, double _kappa, double _numSuccessors):
    l(_l), kappa(_kappa), numSuccessors(_numSuccessors) {}

DualAVF_martingale::~DualAVF_martingale() {}

arma::vec DualAVF_martingale::computeDualAVF_martingale(Option*  myOption, const arma::mat &stock, const arma::mat &beta)
{
    DualAVF* myDualAVF1 = new DualAVF(numSuccessors);
    DualAVF* myDualAVF2 = new DualAVF(std::ceil(numSuccessors/kappa));

    arma::vec finePrice = myDualAVF1->computeDualAVF(myOption, stock, beta);
    arma::vec coarsePrice = myDualAVF2->computeDualAVF(myOption, stock, beta);

    correlation = arma::as_scalar(arma::cor(finePrice, coarsePrice));

    delete myDualAVF1;
    delete myDualAVF2;
    return finePrice - coarsePrice;
}

#endif // DUALAVF_MARTINGALE_CPP
