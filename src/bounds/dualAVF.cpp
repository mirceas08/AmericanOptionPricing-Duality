#ifndef DUALAVF_CPP
#define DUALAVF_CPP

#include "dualAVF.h"
#include "../util.h"
#include "../processes/euler.h"

DualAVF::DualAVF(double _numSuccessors):
    numSuccessors(_numSuccessors) {}

DualAVF::~DualAVF() {}

arma::vec DualAVF::computeDualAVF(Option* myOption, const arma::mat &stock, const arma::mat &beta)
{
    double numSims = stock.n_rows;
    double numIntervals = stock.n_cols-1;
    double dt = myOption->T / numIntervals;
    double J = beta.n_rows;

    double r = myOption->r;
    double sigma = myOption->sigma;
    double K = myOption->K;
    Processes* scheme = new Euler(myOption);

    arma::mat payoff = myOption->payoff->operator()(stock);
    discountPayoff(payoff, myOption);

    arma::mat value = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat continuation = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat martingale = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat dual = arma::zeros<arma::mat>(numSims, numIntervals);

    for (int i = 0; i < numIntervals; i++) {
        double time = (i+1)*dt;

        if (i == numIntervals-1) {
            value.col(i) = payoff.col(i);
        }
        else {
            arma::mat basis = basis_laguerre(stock.col(i+1).t(), J);
            continuation.col(i) = beta.col(i).t() * basis;
            value.col(i) = arma::max(payoff.col(i), continuation.col(i));
        }

        arma::mat Ss = arma::zeros<arma::mat>(numSims, numSuccessors);
        arma::mat Vs = arma::zeros<arma::mat>(numSims, numSuccessors);
        arma::mat Cs = arma::zeros<arma::mat>(numSims, numSuccessors);
        arma::mat Zs = arma::zeros<arma::mat>(numSims, numSuccessors);

        for (int k = 0; k < numSuccessors; k++) {
            arma::vec dW = std::sqrt(dt) * arma::randn(numSims);
            Ss.col(k) = stock.col(i) % arma::exp((r - 0.5*sigma*sigma)*dt + sigma*dW);
        }
        arma::vec myZero = arma::zeros(numSims);
        arma::mat temp = myOption->payoff->operator()(Ss);
        temp = arma::join_horiz(myZero, temp);
        Zs = std::exp(-r*time) * temp;

        for (int k = 0; k < numSuccessors; k++) {
            if (i == numIntervals-1) {
                Vs.col(k) = Zs.col(k);
            }
            else {
                arma::mat basis = basis_laguerre(Ss.col(k).t(), J);
                Cs.col(k) = beta.col(i).t() * basis;
                Vs.col(k) = arma::max(Zs.col(k), Cs.col(k));
            }
        }

        arma::vec diff = value.col(i) - arma::sum(Vs,1) / numSuccessors;

        if (i == 0)
            martingale.col(i) = diff;
        else
            martingale.col(i) = martingale.col(i-1) + diff;


        dual.col(i) = payoff.col(i) - martingale.col(i);
    }

    delete scheme;

    return arma::max(dual, 1);
}


arma::vec DualAVF::computeDualAVF_asian(Option* myOption, const arma::mat &stock, const arma::mat &beta)
{
    double numSims = stock.n_rows;
    double numIntervals = stock.n_cols-1;
    double dt = myOption->T / numIntervals;
    double J = beta.n_rows;

    double r = myOption->r;
    double sigma = myOption->sigma;
    double K = myOption->K;
    Processes* scheme = new Euler(myOption);

    arma::mat payoff = myOption->payoff->operator()(stock);
    discountPayoff(payoff, myOption);

    arma::mat value = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat continuation = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat martingale = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat dual = arma::zeros<arma::mat>(numSims, numIntervals);

    for (int i = 0; i < numIntervals; i++) {
        double time = (i+1)*dt;

        if (i == numIntervals-1) {
            value.col(i) = payoff.col(i);
        }
        else {
            arma::mat basis = basis_laguerre(stock.col(i+1).t(), J);
            continuation.col(i) = beta.col(i).t() * basis;
            value.col(i) = arma::max(payoff.col(i), continuation.col(i));
        }

        arma::mat Ss = arma::zeros<arma::mat>(numSims, numSuccessors);
        arma::mat Vs = arma::zeros<arma::mat>(numSims, numSuccessors);
        arma::mat Cs = arma::zeros<arma::mat>(numSims, numSuccessors);
        arma::mat Zs_temp = arma::zeros<arma::mat>(numSims, numSuccessors);
        arma::mat Zs = arma::zeros<arma::mat>(numSims, numSuccessors);

        for (int k = 0; k < numSuccessors; k++) {
            arma::vec dW = std::sqrt(dt) * arma::randn(numSims);
            Ss.col(k) = stock.col(i) % arma::exp((r - 0.5*sigma*sigma)*dt + sigma*dW);
        }


        // ================================================

        // build payoff matrix

        arma::mat tempStock = stock.head_cols(i+1);
        for (int k = 0; k < numSuccessors; k++) {
            arma::mat temp = arma::join_horiz(tempStock, Ss.col(k));
            temp = arma::join_horiz(arma::zeros(numSims), temp);
            arma::mat temporaryPayoff = myOption->payoff->operator()(temp);
            Zs_temp.col(k) = temporaryPayoff.tail_cols(1);
        }
        Zs = std::exp(-r*time) * Zs_temp;

        // ================================================

        for (int k = 0; k < numSuccessors; k++) {
            if (i == numIntervals-1) {
                Vs.col(k) = Zs.col(k);
            }
            else {
                arma::mat basis = basis_laguerre(Ss.col(k).t(), J);
                Cs.col(k) = beta.col(i).t() * basis;
                Vs.col(k) = arma::max(Zs.col(k), Cs.col(k));
            }
        }

        arma::vec diff = value.col(i) - arma::sum(Vs,1) / numSuccessors;

        if (i == 0)
            martingale.col(i) = diff;
        else
            martingale.col(i) = martingale.col(i-1) + diff;


        dual.col(i) = payoff.col(i) - martingale.col(i);
    }

    delete scheme;

    return arma::max(dual, 1);
}


//arma::vec DualAVF::computeDualAVF(Option* myOption, const arma::mat &stock, const arma::mat &beta)
//{
//    double numSims = stock.n_rows;
//    double numIntervals = stock.n_cols-1;
//    double dt = myOption->T / numIntervals;
//    double J = beta.n_rows;
//
//    double r = myOption->r;
//    double sigma = myOption->sigma;
//    double K = myOption->K;
//    Processes* scheme = new Euler(myOption);
//
//    arma::mat payoff(numSims, numIntervals);
//    for (int i = 0; i < numIntervals; i++) {
//        double time = (i+1)*dt;
//        payoff.col(i) = std::exp(-r*time) * myOption->payoff->operator()(stock.col(i+1));
//    }
//
//    arma::mat value = arma::zeros<arma::mat>(numSims, numIntervals);
//    arma::mat continuation = arma::zeros<arma::mat>(numSims, numIntervals);
//    arma::mat martingale = arma::zeros<arma::mat>(numSims, numIntervals);
//    arma::mat dual = arma::zeros<arma::mat>(numSims, numIntervals);
//
//    for (int i = 0; i < numIntervals; i++) {
//        double time = (i+1)*dt;
//
//        if (i == numIntervals-1) {
//            value.col(i) = payoff.col(i);
//        }
//        else {
//            arma::mat basis = basis_laguerre(stock.col(i+1).t(), J);
//            continuation.col(i) = beta.col(i).t() * basis;
//            value.col(i) = arma::max(payoff.col(i), continuation.col(i));
//        }
//
//        arma::mat Ss = arma::zeros<arma::mat>(numSims, numSuccessors);
//        arma::mat Vs = arma::zeros<arma::mat>(numSims, numSuccessors);
//        arma::mat Cs = arma::zeros<arma::mat>(numSims, numSuccessors);
//        arma::mat Zs = arma::zeros<arma::mat>(numSims, numSuccessors);
//
//        for (int k = 0; k < numSuccessors; k++) {
//            arma::vec dW = std::sqrt(dt) * arma::randn(numSims);
//            Ss.col(k) = stock.col(i) % arma::exp((r - 0.5*sigma*sigma)*dt + sigma*dW);
//            Zs.col(k) = std::exp(-r*time) * myOption->payoff->operator()(Ss.col(k));
//
//            if (i == numIntervals-1) {
//                Vs.col(k) = Zs.col(k);
//            }
//            else {
//                arma::mat basis = basis_laguerre(Ss.col(k).t(), J);
//                Cs.col(k) = beta.col(i).t() * basis;
//                Vs.col(k) = arma::max(Zs.col(k), Cs.col(k));
//            }
//        }
//
//        arma::vec diff = value.col(i) - arma::sum(Vs,1) / numSuccessors;
//
//        if (i == 0)
//            martingale.col(i) = diff;
//        else
//            martingale.col(i) = martingale.col(i-1) + diff;
//
//
//        dual.col(i) = payoff.col(i) - martingale.col(i);
//    }
//
//    delete scheme;
//
//    return arma::max(dual, 1);
//}

#endif // DUALAVF_CPP
