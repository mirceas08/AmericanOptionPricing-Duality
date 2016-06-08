#ifndef LSM_MLMC_H
#define LSM_MLMC_H

#include <armadillo>
#include "../option/option.h"

class LSM_mlmc
{
public:
    double correlation;
private:
    double l;
    double M;
public:
    LSM_mlmc(double _l, double _M);
    virtual ~ LSM_mlmc();

    arma::vec computeLSM_mlmc(Option*  myOption, const arma::field<arma::mat> &stock, const arma::field<arma::mat> &beta);
};


#endif // LSM_MLMC_H
