#ifndef BETA_H
#define BETA_H

#include <armadillo>
#include "../option/option.h"

class Beta
{
private:
    double J;
public:
    Beta(double J);
    virtual ~Beta();

    arma::mat computeBeta(Option*  myOption, const arma::mat &stock);
};


#endif // BETA_H
