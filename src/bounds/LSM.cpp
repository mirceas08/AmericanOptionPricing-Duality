#ifndef LSM_CPP
#define LSM_CPP

#include "LSM.h"
#include "../util.h"

LSM::LSM() {}

LSM::~LSM() {}

arma::vec LSM::computeLSM(Option* myOption, const arma::mat &stock, const arma::mat &beta)
{
    double numSims = stock.n_rows;
    double numIntervals = stock.n_cols-1;
    double dt = myOption->T / numIntervals;
    double J = beta.n_rows;

    arma::mat payoff = myOption->payoff->operator()(stock);
    discountPayoff(payoff, myOption);

    arma::mat value = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat continuation = arma::zeros<arma::mat>(numSims, numIntervals);

    for (int i = 0; i < numIntervals; i++) {
        if (i == numIntervals-1) {
            value.col(i) = payoff.col(i);
        }
        else {
            arma::mat basis = basis_laguerre(stock.col(i+1).t(), J);
            continuation.col(i) = beta.col(i).t() * basis;
            value.col(i) = arma::max(payoff.col(i), continuation.col(i));
        }
    }

    arma::vec LSMvec(numSims);
    for (int k = 0; k < numSims; k++) {
        double spot = arma::as_scalar(find(payoff.row(k) >= continuation.row(k), 1));
        LSMvec(k) = payoff.row(k)(spot);
    }

    return LSMvec;
}


//arma::vec LSM::computeLSM(Option* myOption, const arma::mat &stock, const arma::mat &beta)
//{
//    double numSims = stock.n_rows;
//    double numIntervals = stock.n_cols-1;
//    double dt = myOption->T / numIntervals;
//    double J = beta.n_rows;
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
//    arma::mat value = arma::zeros<arma::mat>(numSims, numIntervals);
//    arma::mat continuation = arma::zeros<arma::mat>(numSims, numIntervals);
//
//    for (int i = 0; i < numIntervals; i++) {
//        if (i == numIntervals-1) {
//            value.col(i) = payoff.col(i);
//        }
//        else {
//            arma::mat basis = basis_laguerre(stock.col(i+1).t(), J);
//            continuation.col(i) = beta.col(i).t() * basis;
//            value.col(i) = arma::max(payoff.col(i), continuation.col(i));
//        }
//    }
//
//    arma::vec LSMvec(numSims);
//    for (int k = 0; k < numSims; k++) {
//        double spot = arma::as_scalar(find(payoff.row(k) >= continuation.row(k), 1));
//        LSMvec(k) = payoff.row(k)(spot);
//    }
//
//    return LSMvec;
//}

#endif // LSM_CPP
