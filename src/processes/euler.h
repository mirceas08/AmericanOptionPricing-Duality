#ifndef EULER_H
#define EULER_H

#include <armadillo>
#include <cmath>
#include "processes.h"
#include "../option/option.h"

class Euler: public Processes
{
public:
    Euler(Option* _myOption);
    virtual ~Euler();

    virtual arma::vec calculateStockPath(const double S0, const double numIntervals);
};

#endif // EULER_H
