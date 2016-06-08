#ifndef DUALSR_MARTINGALE_CPP
#define DUALSR_MARTINGALE_CPP

#include <cmath>
#include "dualSR_martingale.h"
#include "../bounds/dualSR.h"
#include "../util.h"

DualSR_martingale::DualSR_martingale(double _l, double _kappa, double _subSims):
    l(_l), kappa(_kappa), subSims(_subSims) {}

DualSR_martingale::~DualSR_martingale() {}

arma::vec DualSR_martingale::computeDualSR_martingale(Option*  myOption, const arma::mat &stock, const arma::mat &beta)
{
    DualSR* myDualSR1 = new DualSR(subSims);
    DualSR* myDualSR2 = new DualSR(std::ceil(subSims/kappa));

    arma::vec finePrice = myDualSR1->computeDualSR(myOption, stock, beta);
    arma::vec coarsePrice = myDualSR2->computeDualSR(myOption, stock, beta);

    correlation = arma::as_scalar(arma::cor(finePrice, coarsePrice));

    delete myDualSR1;
    delete myDualSR2;
    return finePrice - coarsePrice;
}

#endif // DUALSR_MARTINGALE_CPP
