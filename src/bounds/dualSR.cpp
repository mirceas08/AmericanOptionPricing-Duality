#ifndef DUALSR_CPP
#define DUALSR_CPP

#include "dualSR.h"
#include "../util.h"
#include "../processes/euler.h"

DualSR::DualSR(double _subSims):
    subSims(_subSims) {}

DualSR::~DualSR() {}

arma::vec DualSR::computeDualSR(Option* myOption, const arma::mat &stock, const arma::mat &beta)
{
    double numSims = stock.n_rows;
    double numIntervals = stock.n_cols-1;
    double dt = myOption->T / numIntervals;
    double J = beta.n_rows;

    double r = myOption->r;
    double sigma = myOption->sigma;
    double K = myOption->K;
    Processes* scheme = new Euler(myOption);

    double cumStopping = 0.0;
    double firstTerm;
    double secondTerm;
    double payoffRecorded;
    double payoffTemp;
    double diff;
    double time;
    double condExp;
    arma::vec vecDual(numSims);

    for (int k = 0; k < numSims; k++) {
        arma::vec stockPath = stock.row(k);

        arma::vec payoff(numIntervals);
        arma::vec continuation(numIntervals);
        arma::vec martingale(numIntervals);
        arma::vec dual(numIntervals);
        arma::vec stoppingTime(numIntervals+1);
        arma::vec stockSub;

        for (int i = numIntervals-1; i >= 0; i--) {
            // payoff
            time = (i+1)*dt;
            payoff(i) = std::exp(-r*time) * std::max(K - stockPath(i+1), 0.0);

            // continuation
            arma::vec basis = basis_laguerre(stockPath(i+1), J);
            continuation(i) = arma::as_scalar((beta.col(i)).t() * basis);

            bool exercise = payoff(i) >= continuation(i);

            if (i == numIntervals-1)
                stoppingTime(i+1) = numIntervals-1;
            else
                stoppingTime(i+1) = (i+1)*exercise + stoppingTime(i+2)*(1-exercise);
        }
        stoppingTime(0) = stoppingTime(1);


        for (int i = 0; i < numIntervals; i++) {
            time = (i+1)*dt;

            if (i == 0) {
                payoffRecorded = 0.0;
                for (int j = 0; j < subSims; j++) {
                    stockSub = scheme->calculateStockPath(stockPath(i), stoppingTime(i)-i+1);
                    payoffRecorded += std::exp(-r*((stoppingTime(i))*dt)) * std::max(K - arma::as_scalar(stockSub.tail(1)), 0.0);
                }
                secondTerm = payoffRecorded / subSims;
            }
            else
                secondTerm = condExp;

            payoffRecorded = 0.0;
            for (int j = 0; j < subSims; j++) {
                double dtTime = (stoppingTime(i+1)-i)*dt;
                double dw = std::sqrt(dtTime) * arma::randn();
                double Ssub = stockPath(i+1) * std::exp((myOption->r-0.5*sigma*sigma)*dtTime + sigma*dw);
                payoffRecorded += std::exp(-r*((stoppingTime(i+1))*dt)) * std::max(K - Ssub, 0.0);
            }
            condExp = payoffRecorded / subSims;

            bool larger = payoff(i) >= continuation(i);
            if (larger)
                firstTerm = payoff(i);
            else
                firstTerm = condExp;

            diff = firstTerm - secondTerm;

            if (i == 0)
                martingale(i) = diff;
            else
                martingale(i) = martingale(i-1) + diff;

        }

        dual = payoff - martingale;
        vecDual(k) = arma::as_scalar(arma::max(dual));
    }

    delete scheme;

    return vecDual;
}



arma::vec DualSR::computeDualSR_matrixForm(Option* myOption, const arma::mat &stock, const arma::mat &beta)
{
    double numSims = stock.n_rows;
    double numIntervals = stock.n_cols-1;
    double dt = myOption->T / numIntervals;
    double J = beta.n_rows;

    double r = myOption->r;
    double sigma = myOption->sigma;
    double K = myOption->K;
    Processes* scheme = new Euler(myOption);

    arma::mat payoff(numSims, numIntervals);
    arma::mat continuation(numSims, numIntervals);
    arma::mat stoppingTime(numSims, numIntervals+1);
    for (int i = numIntervals-1; i >= 0; i--) {
        double time = (i+1)*dt;
        payoff.col(i) = std::exp(-r*time) * myOption->payoff->operator()(stock.col(i+1));

        arma::mat basis = basis_laguerre(stock.col(i+1).t(), J);
        continuation.col(i) = beta.col(i).t() * basis;

        arma::uvec exercise = payoff.col(i) >= continuation.col(i);

        if (i == numIntervals-1)
            stoppingTime.col(i+1).fill(numIntervals-1);
        else
            stoppingTime.col(i+1) = (i+1)*exercise + stoppingTime.col(i+2) % (1-exercise);
    }
    stoppingTime.col(0) = stoppingTime.col(1);

    arma::mat value = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat martingale = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat dual = arma::zeros<arma::mat>(numSims, numIntervals);


    arma::vec firstTerm(numSims);
    arma::vec secondTerm(numSims);
    arma::vec condExp(numSims);
    arma::mat payoffRecorded(numSims, subSims);
    arma::mat stockSub(numSims, subSims);
    for (int i = 0; i < numIntervals; i++) {
        double time = (i+1)*dt;

        if (i == 0) {
            for (int k = 0; k < subSims; k++) {
                arma::vec dtTime = (stoppingTime.col(i)-i+1)*dt;
                arma::vec dW = arma::sqrt(dtTime) * arma::randn(numSims);
                stockSub.col(k) = stock.col(i) % arma::exp((r - 0.5*sigma*sigma)*dtTime + sigma*dW);
                payoffRecorded.col(k) = arma::exp(-r*(stoppingTime.col(i+1)*dt)) % myOption->payoff->operator()(stockSub.col(k));
            }
            secondTerm = arma::sum(payoffRecorded,1) / subSims;
        }
        else
            secondTerm = condExp;

        arma::mat Ss = arma::zeros<arma::mat>(numSims, subSims);
        arma::mat Vs = arma::zeros<arma::mat>(numSims, subSims);
        arma::mat Cs = arma::zeros<arma::mat>(numSims, subSims);
        arma::mat Zs = arma::zeros<arma::mat>(numSims, subSims);

        for (int k = 0; k < subSims; k++) {
            arma::vec dtTime = (stoppingTime.col(i+1)-i)*dt;
            arma::vec dW = arma::sqrt(dtTime) * arma::randn(numSims);
            Ss.col(k) = stock.col(i+1) % arma::exp((r - 0.5*sigma*sigma)*dtTime + sigma*dW);
            Zs.col(k) = arma::exp(-r*(stoppingTime.col(i+1)*dt)) % myOption->payoff->operator()(Ss.col(k));
        }
        condExp = arma::sum(Zs,1) / subSims;

        arma::uvec larger = payoff.col(i) >= continuation.col(i);

        for (int j = 0; j < numSims; j++) {
            if (larger(j))
                firstTerm(j) = payoff.col(i)(j);
            else
                firstTerm(j) = condExp(j);
        }

        arma::vec diff = firstTerm - secondTerm;

        if (i == 0)
            martingale.col(i) = diff;
        else
            martingale.col(i) = martingale.col(i-1) + diff;

        dual.col(i) = payoff.col(i) - martingale.col(i);
    }

    delete scheme;

    return arma::max(dual, 1);
}

#endif // DUALSR_CPP
