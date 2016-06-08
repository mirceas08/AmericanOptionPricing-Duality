#ifndef DUALAVF_H
#define DUALAVF_H

#include <armadillo>
#include "../option/option.h"

class DualAVF
{
private:
    double numSuccessors;
public:
    DualAVF(double numSuccessors);
    virtual ~DualAVF();

    arma::vec computeDualAVF(Option*  myOption, const arma::mat &stock, const arma::mat &beta);
    arma::vec computeDualAVF_asian(Option*  myOption, const arma::mat &stock, const arma::mat &beta);
};


#endif // DUALAVF_H
