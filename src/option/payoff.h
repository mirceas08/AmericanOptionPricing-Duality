#ifndef PAYOFF_H
#define PAYOFF_H

#include <algorithm>
#include <armadillo>

// base payoff class
class PayOff
{
public:
    PayOff();
    virtual ~PayOff() {};

    // turn PayOff into an abstract functor
    virtual arma::mat operator () (const arma::mat &S) const = 0;
};


// derived put option class
class PayOffPut: public PayOff
{
private:
    double K;
public:
    PayOffPut(const double &_K);
    virtual ~PayOffPut() {};

    virtual arma::mat operator () (const arma::mat &S) const;
};

// derived call option class
class PayOffCall: public PayOff
{
private:
    double K;
public:
    PayOffCall(const double &_K);
    virtual ~PayOffCall() {};

    virtual arma::mat operator () (const arma::mat &S) const;
};

// derived Asian call option class
class AsianCall: public PayOff
{
private:
    double K;
public:
    AsianCall(const double &_K);
    virtual ~AsianCall() {};

    virtual arma::mat operator () (const arma::mat &S) const;
};

// derived Asian put option class
class AsianPut: public PayOff
{
private:
    double K;
public:
    AsianPut(const double &_K);
    virtual ~AsianPut() {};

    virtual arma::mat operator () (const arma::mat &S) const;
};


#endif // PAYOFF_H
