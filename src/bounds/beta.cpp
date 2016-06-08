#ifndef BETA_CPP
#define BETA_CPP

#include "beta.h"
#include "../util.h"

Beta::Beta(double _J):
    J(_J) {}

Beta::~Beta() {}


arma::mat Beta::computeBeta(Option* myOption, const arma::mat &stock)
{
    double numSims = stock.n_rows;
    double numIntervals = stock.n_cols-1;
    double dt = myOption->T / numIntervals;

    arma::mat payoff = myOption->payoff->operator()(stock);
    discountPayoff(payoff, myOption);

    arma::mat beta(J, numIntervals);
    arma::mat cont(numSims, numIntervals);
    arma::vec value = payoff.col(numIntervals-1);

    for (int i = numIntervals-2; i >= 0; i--) {
        arma::mat basis = basis_laguerre(stock.col(i+1).t(), J);
        arma::mat beta1 = arma::zeros<arma::mat>(J,J);
        arma::vec beta2 = arma::zeros<arma::vec>(J);

        for (int j = 0; j < numSims; j++) {
            beta1 = beta1 + basis.col(j) * basis.col(j).t();
            beta2 = beta2 + basis.col(j) * value(j);
        }

        beta1 = beta1 / numSims;
        beta2 = beta2 / numSims;

        beta.col(i) = solve(beta1,beta2);
        cont.col(i) = beta.col(i).t() * basis;
        value = arma::max(payoff.col(i), cont.col(i));
    }

    return beta;
}


//arma::mat Beta::computeBeta(Option* myOption, const arma::mat &stock)
//{
//    double numSims = stock.n_rows;
//    double numIntervals = stock.n_cols-1;
//    double dt = myOption->T / numIntervals;
//
////    arma::vec time(numIntervals);
////    for (int i = 0; i < numIntervals; i++) {
////        double myTime = (i+1)*dt;
////        time(i) = std::exp(-(myOption->r)*myTime);
////    }
//
//    arma::mat payoff(numSims, numIntervals);
//    for (int i = 0; i < numIntervals; i++) {
//        double time = (i+1)*dt;
//        payoff.col(i) = std::exp(-(myOption->r)*time) * myOption->payoff->operator()(stock.col(i+1));
//    }
////    for (int i = 0; i < numSims; i++) {
////        payoff.row(i) = time % myOption->payoff->operator()(stock.row(i).tail(numIntervals));
////    }
//
//    arma::mat beta(J, numIntervals);
//    arma::mat cont(numSims, numIntervals);
//    arma::vec value = payoff.col(numIntervals-1);
//
//    for (int i = numIntervals-2; i >= 0; i--) {
//        arma::mat basis = basis_laguerre(stock.col(i+1).t(), J);
//        arma::mat beta1 = arma::zeros<arma::mat>(J,J);
//        arma::vec beta2 = arma::zeros<arma::vec>(J);
//
//        for (int j = 0; j < numSims; j++) {
//            beta1 = beta1 + basis.col(j) * basis.col(j).t();
//            beta2 = beta2 + basis.col(j) * value(j);
//        }
//
//        beta1 = beta1 / numSims;
//        beta2 = beta2 / numSims;
//
//        beta.col(i) = solve(beta1,beta2);
//        cont.col(i) = beta.col(i).t() * basis;
//        value = arma::max(payoff.col(i), cont.col(i));
//    }
//
//    return beta;
//}

#endif // BETA_CPP
