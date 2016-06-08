#ifndef PROCESSES_H
#define PROCESSES_H

#include <armadillo>
#include <cmath>
#include "../option/option.h"


// base class
class Processes
{
protected:
    Option* myOption;
public:
    Processes(Option* _myOption);
    virtual ~Processes();

    virtual arma::vec calculateStockPath(const double S0, const double numIntervals) = 0;
};


#endif // PROCESSES_H
