#ifndef PAYOFF_CPP
#define PAYOFF_CPP

#include "payoff.h"

/* ------------------- PayOff ----------------------- */

PayOff::PayOff() {}


/* ------------------- PayOffPut  ----------------------- */

PayOffPut::PayOffPut(const double &_K) : K(_K) {}

arma::mat PayOffPut::operator() (const arma::mat &S) const
{
    double numSims = S.n_rows;
    double numIntervals = S.n_cols-1;

    arma::mat payoff(numSims, numIntervals);
    for (int i = 0; i < numIntervals; i++) {
        payoff.col(i) = arma::max(K - S.col(i+1), arma::zeros<arma::vec>(numSims));
    }

    return payoff;
}

/* ------------------- PayOffCall  ----------------------- */

PayOffCall::PayOffCall(const double &_K) : K(_K) {}

arma::mat PayOffCall::operator() (const arma::mat &S) const
{
    double numSims = S.n_rows;
    double numIntervals = S.n_cols-1;

    arma::mat payoff(numSims, numIntervals);
    for (int i = 0; i < numIntervals; i++) {
        payoff.col(i) = arma::max(S.col(i+1) - K, arma::zeros<arma::vec>(numSims));
    }

    return payoff;
}


/* ------------------- Asian Call  ----------------------- */

AsianCall::AsianCall(const double &_K) : K(_K) {}

arma::mat AsianCall::operator() (const arma::mat &S) const
{
    double numSims = S.n_rows;
    double numIntervals = S.n_cols-1;

    arma::mat average(numSims, numIntervals+1);
    for (int i = 0; i < average.n_rows; i++) {
        for (int j = 0; j < average.n_cols; j++) {
            average(i,j) = arma::mean(S.row(i).head(j+2));
        }
    }

    arma::mat payoff(numSims, numIntervals);
    for (int i = 0; i < numIntervals; i++) {
        payoff.col(i) = arma::max(average.col(i+1) - K, arma::zeros<arma::vec>(numSims));
    }

    return payoff;
}

/* ------------------- Asian Put  ----------------------- */

AsianPut::AsianPut(const double &_K) : K(_K) {}

arma::mat AsianPut::operator() (const arma::mat &S) const
{
    double numSims = S.n_rows;
    double numIntervals = S.n_cols-1;

    arma::mat average(numSims, numIntervals+1);
    for (int i = 0; i < average.n_rows; i++) {
        for (int j = 0; j < average.n_cols; j++) {
            average(i,j) = arma::mean(S.row(i).head(j+2));
        }
    }

    arma::mat payoff(numSims, numIntervals);
    for (int i = 0; i < numIntervals; i++) {
        payoff.col(i) = arma::max(K - average.col(i+1), arma::zeros<arma::vec>(numSims));
    }

    return payoff;
}


#endif // PAYOFF_CPP
