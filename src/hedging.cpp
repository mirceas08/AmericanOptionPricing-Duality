#include <iostream>
#include <algorithm>
#include <armadillo>
#include <cmath>
#include <iomanip>

#include "option/option.h"
#include "option/payoff.h"
#include "processes/processes.h"
#include "processes/euler.h"
#include "util.h"

#include "bounds/beta.h"
#include "bounds/dualAVF.h"


int main(int argc, char **argv)
{
    arma::arma_rng::set_seed_random();

    double numIntervals;
    double numSims;
    double numSimsBeta;
    double numSuccessors;
    double S0;                              // initial spot price
    double K;                               // strike
    double r;                               // risk-free rate
    double sigma;                              // initial variance
    double T;                               // maturity
    double J;
    double kappa;
    double epsilon;

    std::string dataFile = argv[1];
    std::ifstream fIN(dataFile.c_str());
    std::string line;

    if (fIN.is_open()) {
        while (std::getline(fIN, line)) {
        std::stringstream stream(line);
        std::string variable;
        std::string value;

        stream >> variable >> value;

        if (variable == "numSimsBeta")
            numSimsBeta = atof(value.c_str());
        else if (variable == "S0")
            S0 = atof(value.c_str());
        else if (variable == "timeSteps")
            numIntervals = atof(value.c_str());
        else if (variable == "numSims")
            numSims = atof(value.c_str());
        else if (variable == "numSuccessors")
            numSuccessors = atof(value.c_str());
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
        else if (variable == "kappa")
            kappa = atof(value.c_str());
        else if (variable == "epsilon")
            epsilon = atof(value.c_str());
        }
    }
    else {
        std::cout << "Error opening file" << std::endl;
        return -1;
    }



// ==============================================================================================================================================================================

    double n = 100;
	// Matrices for the stock price evolution and option price
	arma::mat binS(n+1,n+1);
	arma::mat Op(n+1,n+1);

	// Quantities for the tree
	double bindt = T/n;
	double u = exp(sigma*sqrt(bindt));
	double d = 1.0/u;
	double p = (std::exp(r*bindt)-d) / (u-d);

	// Build the binomial tree
	for (int j=0; j<=n; j++)
		for (int i=0; i<=j; i++)
	 		 binS(i,j) = S0*std::pow(u,j-i)*std::pow(d,i);

	// Compute terminal payoffs
	for (int i=0; i<=n; i++)
		Op(i,n) = std::max(K - binS(i,n), 0.0);

	// Backward recursion through the tree
	for (int j=n-1; j>=0; j--)
		for (int i=0; i<=j; i++)
			Op(i,j) = std::max(K - binS(i,j), std::exp(-r*bindt)*(p*(Op(i,j+1)) + (1.0-p)*(Op(i+1,j+1))));

	// Return the option price
	std::cout << "Binomial price: " << Op(0,0) << std::endl;
	arma::mat hedging1(n+1, n+1);
	for (int j=0; j<=n; j++)
		for (int i=0; i<=j; i++)
	 		 hedging1(i,j) = (Op(i,j+1) - Op(i+1,j+1)) / (binS(i,j+1) - binS(i+1,j+1));

    std::cout << "Hedging: " << hedging1(0,0) << std::endl;



// ==============================================================================================================================================================================









    /* ------------------------ Set option, payoff and discretization scheme objects ------------------------ */
    PayOff* myPayoff = new PayOffPut(K);
    Option* myOption = new Option(K, r, sigma, T, myPayoff);
    Processes* scheme = new Euler(myOption);

    std::cout << "=================== Option parameters ===================" << std::endl;
    std::cout << "Spot price: " << S0 << std::endl;
    std::cout << "Strike: " << K << std::endl;
    std::cout << "Risk free rate: " << r << std::endl;
    std::cout << "Volatility: " << sigma << std::endl;
    std::cout << "Maturity: " << T << std::endl;
    std::cout << "================= LSM parameters =================" << std::endl;
    std::cout << "Number of basis functions: " << J << std::endl;
    std::cout << "================= Monte Carlo parameters =================" << std::endl;
    std::cout << "Numer of time steps: " << numIntervals << std::endl;
    std::cout << "Number of simulations for beta estimation: " << numSimsBeta << std::endl;
    std::cout << "Number of simulations: " << numSims << std::endl;
    std::cout << "Number of successors: " << numSuccessors << std::endl;


    // coefficient estimation

    // Timer
    arma::wall_clock timer;
    timer.tic();
    double dt = myOption->T / numIntervals;

    /* ***************** Beta estimation ******************** */
    Beta* betaEstimation = new Beta(J);
    arma::mat stock(numSimsBeta, numIntervals+1);
    for (int i = 0; i < numSimsBeta; i++) {
        stock.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
    }
    arma::mat beta = betaEstimation->computeBeta(myOption, stock);

    /* ***************** MLMC pilot run ******************** */
    DualAVF* myDualAVF = new DualAVF(numSuccessors);

    stock.resize(numSims, numIntervals+1);
    for (int i = 0; i < numSims; i++) {
        stock.row(i) = scheme->calculateStockPath(S0, numIntervals+1);
    }

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

    arma::vec AVFprice = arma::max(dual, 1);

    double priceMLMC = arma::mean(AVFprice);
    double variance = arma::var(AVFprice) / numSims;

    // hedging
    arma::mat discountedStock = stock;
    discountStock(discountedStock, myOption);
    arma::mat diffStock(numSimsBeta, numIntervals);
    for (int i = 0; i < numIntervals; i++) {
        diffStock.col(i) = discountedStock.col(i+1) - discountedStock.col(i);
    }

    arma::mat diffValue(numSims, numIntervals);
    for (int i = 1; i < numIntervals; i++) {
        diffValue.col(i) = value.col(i) - value.col(i-1);
    }
    diffValue.col(0) = value.col(0) - priceMLMC;

    arma::mat theta(3, numIntervals);
    for (int i = numIntervals-1; i >= 0; i--) {
        if (i == 0) {
            theta(0, i) = arma::as_scalar(arma::cov(value.col(i), diffStock.col(i)) / arma::var(diffStock.col(i)));
            theta(1, i) = priceMLMC - theta(0, i) * arma::mean(discountedStock.col(i));
            theta(2, i) = -arma::mean(diffValue.col(i));
        }
        else {
            theta(0, i) = arma::as_scalar(arma::cov(value.col(i), diffStock.col(i)) / arma::var(diffStock.col(i)));
            theta(1, i) = arma::mean(value.col(i-1)) - theta(0, i) * arma::mean(discountedStock.col(i));
            theta(2, i) = -arma::mean(diffValue.col(i));
        }
    }

    stock.save("stock.dat", arma::raw_ascii);
    payoff.save("payoff.dat", arma::raw_ascii);
    martingale.save("martingale.dat", arma::raw_ascii);
    theta.save("theta.dat", arma::raw_ascii);

    std::cout << "=================== Standard Monte Carlo ===================" << std::endl;
    std::cout << "Standard Monte Carlo estimator: " << priceMLMC << std::endl;
    std::cout << "Standard Monte Carlo variance: " << variance << std::endl;

    delete myPayoff;
    delete myOption;
    delete scheme;
    delete betaEstimation;

    return 0;
}
