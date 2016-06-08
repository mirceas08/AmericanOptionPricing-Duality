#include <iostream>
#include <algorithm>
#include "option/option.h"
#include "option/payoff.h"
#include "gbm/gbmDiscr.h"
#include "gbm/gbmEuler.h"
#include "util.h"

#include <armadillo>

int main(int argc, char **argv)
{
    arma::arma_rng::set_seed_random();

    double numSims;                            // number of simulations
    double numSimsBeta;
    double numIntervals;                       // number of time steps
    double S0;                              // initial spot price
    double K;                               // strike
    double r;                               // risk-free rate
    double sigma;                              // initial variance
    double T;                               // maturity
    std::string discretizationScheme;       // discretization scheme
    double J;
    double numSuccessors;
    double subSims;

    std::string dataFile = argv[1];
    std::ifstream fIN(dataFile.c_str());
    std::string line;

    if (fIN.is_open()) {
        while (std::getline(fIN, line)) {
        std::stringstream stream(line);
        std::string variable;
        std::string value;

        stream >> variable >> value;

        if (variable == "numSims")
            numSims = atoi(value.c_str());
        else if (variable == "timeSteps")
            numIntervals = atoi(value.c_str());
        else if (variable == "numSimsBeta")
            numSimsBeta = atof(value.c_str());
        else if (variable == "S0")
            S0 = atof(value.c_str());
        else if (variable == "strike")
            K = atof(value.c_str());
        else if (variable == "r")
            r = atof(value.c_str());
        else if (variable == "J")
            J = atof(value.c_str());
        else if (variable == "sigma")
            sigma = atof(value.c_str());
        else if (variable == "maturity")
            T = atof(value.c_str());
        else if (variable == "numSuccessors")
            numSuccessors = atof(value.c_str());
        else if (variable == "subSims")
            subSims = atof(value.c_str());
        else if (variable == "discretizationScheme")
            discretizationScheme = value;
        }
    }
    else {
        std::cout << "Error opening file" << std::endl;
        return -1;
    }

    transform(discretizationScheme.begin(), discretizationScheme.end(), discretizationScheme.begin(), ::toupper);

    /* ------------------------ Set option, payoff and discretization scheme objects ------------------------ */
    PayOff* myPayoff = new PayOffPut(K);
    Option* myOption = new Option(K, r, sigma, T, myPayoff);
    gbmDiscretization* scheme;
    scheme = new gbmEuler(myOption);
    double dt = myOption->T / numIntervals;


    std::cout << "=================== Option parameters ===================" << std::endl;
    std::cout << "Spot price: " << S0 << std::endl;
    std::cout << "Strike: " << K << std::endl;
    std::cout << "Risk free rate: " << r << std::endl;
    std::cout << "Volatility: " << sigma << std::endl;
    std::cout << "Maturity: " << T << std::endl;
    std::cout << "================= Simulation parameters =================" << std::endl;
    std::cout << "Number of simulations: " << numSims << std::endl;
    std::cout << "Number of successors for Duality AVF: " << numSuccessors << std::endl;
    std::cout << "Number of subsimulations for Duality SR: " << subSims << std::endl;
    std::cout << "Number of time steps: " << numIntervals << std::endl;
    std::cout << "Number of basis functions: " << J << std::endl;


// ========================================================================================================================
/*
    // binomial for american options
    double treeN = 5000;

	// Matrices for the stock price evolution and option price
	arma::mat S(treeN+1,treeN+1);
	arma::mat Op(treeN+1,treeN+1);
	double treeDT,u,d,p;

	// Quantities for the tree
	treeDT = T/ treeN;
	u = std::exp(sigma*sqrt(treeDT));
	d = 1.0/u;
	p = (std::exp(r*treeDT)-d) / (u-d);

	// Build the binomial tree
	for (int j=0; j<=treeN; j++)
		for (int i=0; i<=j; i++)
	 		 S(i,j) = S0*std::pow(u,j-i)*std::pow(d,i);

	// Compute terminal payoffs
	for (int i=0; i<=treeN; i++)
        Op(i,treeN) = std::max(K - S(i,treeN), 0.0);

	// Backward recursion through the tree
	for (int j=treeN-1; j>=0; j--)
		for (int i=0; i<=j; i++)
			Op(i,j) = std::max(K - S(i,j), std::exp(-r*treeDT)*(p*(Op(i,j+1)) + (1.0-p)*(Op(i+1,j+1))));


	// Return the option price
	std::cout << "============================================" << std::endl;
	std::cout << "Price from binomial: " << Op(0,0) << std::endl;
	std::cout << "============================================" << std::endl;
*/

// ========================================================================================================================

    // coefficient estimation

    // Timer
    arma::wall_clock timer;
    timer.tic();

    arma::mat stockBeta(numSimsBeta, numIntervals+1);
    for (int i = 0; i < numSimsBeta; i++) {
        stockBeta.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
    }

    arma::mat payoffBeta(numSimsBeta, numIntervals);
    for (int i = 0; i < numIntervals; i++) {
        double time = (i+1)*dt;
        payoffBeta.col(i) = std::exp(-r*time) * arma::max(K-stockBeta.col(i+1), arma::zeros<arma::vec>(numSimsBeta));
    }

    arma::mat beta(J, numIntervals);
    arma::mat cont(numSimsBeta, numIntervals);
    arma::vec value = payoffBeta.col(numIntervals-1);

    for (int i = numIntervals-2; i >= 0; i--) {
        arma::mat basis = vec_basis_laguerre(stockBeta.col(i+1).t(), J);
        arma::mat beta1 = arma::zeros<arma::mat>(J,J);
        arma::vec beta2 = arma::zeros<arma::vec>(J);

        for (int j = 0; j < numSimsBeta; j++) {
            beta1 = beta1 + basis.col(j) * basis.col(j).t();
            beta2 = beta2 + basis.col(j) * value(j);
        }

        beta1 = beta1 / numSims;
        beta2 = beta2 / numSims;

        beta.col(i) = solve(beta1,beta2);
        cont.col(i) = beta.col(i).t() * basis;
        value = arma::max(payoffBeta.col(i), cont.col(i));
    }

    double timeElapsed = timer.toc();

// ========================================================================================================================

    timer.tic();
    // Longstaff-Schwartz in matrix form

    arma::mat stockPathMAT(numSims, numIntervals+1);
    stockPathMAT.col(0) = S0 * arma::ones<arma::vec>(numSims);

    for (int i = 1; i < numIntervals+1; i++) {
        arma::vec dW = std::sqrt(dt) * arma::randn(numSims);
        stockPathMAT.col(i) = stockPathMAT.col(i-1) % arma::exp((r-0.5*sigma*sigma)*dt + sigma*dW);
    }

    arma::mat payoffMAT(numSims, numIntervals);
    for (int i = 0; i < numIntervals; i++) {
        double time = (i+1)*dt;
        payoffMAT.col(i) = std::exp(-r*time) * arma::max(K-stockPathMAT.col(i+1), arma::zeros<arma::vec>(numSims));
    }

    arma::mat vMAT = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat continuationMAT = arma::zeros<arma::mat>(numSims, numIntervals);

    for (int i = 0; i < numIntervals; i++) {
        if (i == numIntervals-1) {
            vMAT.col(i) = payoffMAT.col(i);
        }
        else {
            arma::mat basis = vec_basis_laguerre(stockPathMAT.col(i+1).t(), J);
            continuationMAT.col(i) = beta.col(i).t() * basis;
            vMAT.col(i) = arma::max(payoffMAT.col(i), continuationMAT.col(i));
        }
    }

    arma::vec vecRule(numSims);
    for (int k = 0; k < numSims; k++) {
        double spot = arma::as_scalar(find(payoffMAT.row(k) >= continuationMAT.row(k), 1));
        vecRule(k) = payoffMAT.row(k)(spot);
    }

    timeElapsed = timer.toc();
    std::cout << "====================== LSM ======================" << std::endl;
    std::cout << "LSM - lower bound: " << arma::mean(vecRule) << std::endl;
    std::cout << "LSM - variance: " << arma::var(vecRule) / numSims << std::endl;
    std::cout << "LSM - elapsed time: " << timeElapsed << std::endl;

// ========================================================================================================================

    timer.tic();
    // Duality in matrix form

    arma::mat stockPathDUAL(numSims, numIntervals+1);
    stockPathDUAL.col(0) = S0 * arma::ones<arma::vec>(numSims);

    for (int i = 1; i < numIntervals+1; i++) {
        arma::vec dW = std::sqrt(dt) * arma::randn(numSims);
        stockPathDUAL.col(i) = stockPathDUAL.col(i-1) % arma::exp((r-0.5*sigma*sigma)*dt + sigma*dW);
    }

    arma::mat payoffDUAL(numSims, numIntervals);
    for (int i = 0; i < numIntervals; i++) {
        double time = (i+1)*dt;
        payoffDUAL.col(i) = std::exp(-r*time) * arma::max(K-stockPathDUAL.col(i+1), arma::zeros<arma::vec>(numSims));
    }

    arma::mat vDUAL = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat continuationDUAL = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat martingaleDUAL = arma::zeros<arma::mat>(numSims, numIntervals);
    arma::mat dualDUAL = arma::zeros<arma::mat>(numSims, numIntervals);

    for (int i = 0; i < numIntervals; i++) {
        double time = (i+1)*dt;


        if (i == numIntervals-1) {
            vDUAL.col(i) = payoffDUAL.col(i);
        }
        else {
            arma::mat basis = vec_basis_laguerre(stockPathDUAL.col(i+1).t(), J);
            continuationDUAL.col(i) = beta.col(i).t() * basis;
            vDUAL.col(i) = arma::max(payoffDUAL.col(i), continuationDUAL.col(i));
        }

        arma::mat Ss = arma::zeros<arma::mat>(numSims, numSuccessors);
        arma::mat Vs = arma::zeros<arma::mat>(numSims, numSuccessors);
        arma::mat Cs = arma::zeros<arma::mat>(numSims, numSuccessors);
        arma::mat Zs = arma::zeros<arma::mat>(numSims, numSuccessors);

        for (int k = 0; k < numSuccessors; k++) {
            arma::vec dW = std::sqrt(dt) * arma::randn(numSims);
            Ss.col(k) = stockPathDUAL.col(i) % arma::exp((r-0.5*sigma*sigma)*dt + sigma*dW);
            Zs.col(k) = std::exp(-r*time) * arma::max(K - Ss.col(k), arma::zeros<arma::vec>(numSims));

            if (i == numIntervals-1) {
                Vs.col(k) = Zs.col(k);
            }
            else {
                arma::mat basis = vec_basis_laguerre(Ss.col(k).t(), J);
                Cs.col(k) = beta.col(i).t() * basis;
                Vs.col(k) = arma::max(Zs.col(k), Cs.col(k));
            }
        }

        arma::vec diff = vDUAL.col(i) - arma::sum(Vs,1) / numSuccessors;

        if (i == 0)
            martingaleDUAL.col(i) = diff;
        else
            martingaleDUAL.col(i) = martingaleDUAL.col(i-1) + diff;


        dualDUAL.col(i) = payoffDUAL.col(i) - martingaleDUAL.col(i);
    }

    arma::vec vecDual = arma::max(dualDUAL, 1);

    timeElapsed = timer.toc();
    std::cout << "====================== Duality AVF ======================" << std::endl;
    std::cout << "Duality AVF - upper bound: " << arma::mean(vecDual) << std::endl;
    std::cout << "Duality AVF - variance: " << arma::var(vecDual) / numSims << std::endl;
    std::cout << "Duality AVF - elapsed time: " << timeElapsed << std::endl;


// ========================================================================================================================


    // Martingales from stopping rules

    timer.tic();

    double cumStopping = 0.0;
    double firstTerm;
    double secondTerm;
    double payoffRecorded;
    double payoffTemp;
    double diff;
    double time;
    double condExp;
    arma::vec vecDualSR(numSims);

    for (int k = 0; k < numSims; k++) {
        arma::vec stockPath = scheme->calculateStockPath(S0, numIntervals+1);


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
            arma::vec basis = phi(stockPath(i+1), J);
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
                double Ssub = stockPath(i+1) * std::exp((r-0.5*sigma*sigma)*dtTime + sigma*dw);
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
        vecDualSR(k) = arma::as_scalar(arma::max(dual));
    }

    timeElapsed = timer.toc();
    std::cout << "====================== Duality SR ======================" << std::endl;
    std::cout << "Duality SR - upper bound: " << arma::mean(vecDualSR) << std::endl;
    std::cout << "Duality SR - variance: " << arma::var(vecDualSR) / numSims << std::endl;
    std::cout << "Duality SR - elapsed time: " << timeElapsed << std::endl;
    std::cout << "=================================================" << std::endl;

// ========================================================================================================================

    delete myPayoff;
    delete myOption;
    delete scheme;

    return 0;
}
