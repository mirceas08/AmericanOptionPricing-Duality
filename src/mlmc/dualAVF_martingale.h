#ifndef DUALAVF_MARTINGALE_H
#define DUALAVF_MARTINGALE_H

#include <armadillo>
#include "../option/option.h"

class DualAVF_martingale
{
public:
    double correlation;
private:
    double l;
    double kappa;
    double numSuccessors;
public:
    DualAVF_martingale(double _l, double _kappa, double _numSuccessors);
    virtual ~ DualAVF_martingale();

    arma::vec computeDualAVF_martingale(Option*  myOption, const arma::mat &stock, const arma::mat &beta);
};


#endif // DUALAVF_MARTINGALE_H
