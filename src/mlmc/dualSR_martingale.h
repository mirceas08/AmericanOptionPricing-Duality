#ifndef DUALSR_MARTINGALE_H
#define DUALSR_MARTINGALE_H

#include <armadillo>
#include "../option/option.h"

class DualSR_martingale
{
public:
    double correlation;
private:
    double l;
    double kappa;
    double subSims;
public:
    DualSR_martingale(double _l, double _kappa, double _subSims);
    virtual ~ DualSR_martingale();

    arma::vec computeDualSR_martingale(Option*  myOption, const arma::mat &stock, const arma::mat &beta);
};


#endif // DUALSR_MARTINGALE_H
