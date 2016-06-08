#ifndef LSM_H
#define LSM_H

#include <armadillo>
#include "../option/option.h"

class LSM
{
public:
    LSM();
    virtual ~LSM();

    arma::vec computeLSM(Option*  myOption, const arma::mat &stock, const arma::mat &beta);
};


#endif // LSM_H
