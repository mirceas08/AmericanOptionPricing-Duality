#ifndef DUALSR_H
#define DUALSR_H

#include <armadillo>
#include "../option/option.h"

class DualSR
{
private:
    double subSims;
public:
    DualSR(double _subSims);
    virtual ~DualSR();

    arma::vec computeDualSR(Option*  myOption, const arma::mat &stock, const arma::mat &beta);
    arma::vec computeDualSR_matrixForm(Option*  myOption, const arma::mat &stock, const arma::mat &beta);
};


#endif // DUALSR_H
