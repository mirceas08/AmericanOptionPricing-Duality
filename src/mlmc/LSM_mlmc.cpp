#ifndef LSM_MLMC_CPP
#define LSM_MLMC_CPP

#include "LSM_mlmc.h"
#include "../bounds/LSM.h"
#include "../util.h"


LSM_mlmc::LSM_mlmc(double _l, double _M):
    l(_l), M(_M) {}

LSM_mlmc::~LSM_mlmc() {}

arma::vec LSM_mlmc::computeLSM_mlmc(Option*  myOption, const arma::field<arma::mat> &stock, const arma::field<arma::mat> &beta)
{
    LSM* myLSM = new LSM();
    arma::vec finePrice = myLSM->computeLSM(myOption, stock(0,0), beta(0,0));
    arma::vec coarsePrice = myLSM->computeLSM(myOption, stock(1,0), beta(1,0));

    correlation = arma::as_scalar(arma::cor(finePrice, coarsePrice));

    delete myLSM;
    return finePrice - coarsePrice;
}

#endif // LSM_MLMC_CPP
