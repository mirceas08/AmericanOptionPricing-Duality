#ifndef UTIL_H
#define UTIL_H

#include <armadillo>
#include <iomanip>
using namespace std;

// display data in tabular format
template<typename T> void printElement(T t, const int& width)
{
    const char separator = ' ';
    cout << left << setw(width) << setfill(separator) << setprecision(8) << t;
}

// ========================================================================================================================

inline void discountPayoff(arma::mat &payoff, Option* myOption)
{
    double numSims = payoff.n_rows;
    double numIntervals = payoff.n_cols;
    double dt = myOption->T / numIntervals;

    arma::vec time(numIntervals);
    for (int i = 0; i < numIntervals; i++) {
        double myTime = (i+1)*dt;
        time(i) = std::exp(-(myOption->r)*myTime);
    }

    for (int i = 0; i < numSims; i++)
        payoff.row(i) = time % payoff.row(i);
}

// ========================================================================================================================

inline void discountStock(arma::mat &stock, Option* myOption)
{
    double numSims = stock.n_rows;
    double numIntervals = stock.n_cols;
    double dt = myOption->T / numIntervals;

    arma::vec time(numIntervals);
    for (int i = 0; i < numIntervals; i++) {
        double myTime = i*dt;
        time(i) = std::exp(-(myOption->r)*myTime);
    }

    for (int i = 0; i < numSims; i++)
        stock.row(i) = time % stock.row(i);
}

// ========================================================================================================================

inline double laguerre(double x, double k)
{
    if (k == 0)
        return 1.0;
    else if (k == 1)
        return 1.0 - x;
    else
        return (1/k) * ((2*k - 1 - x) * laguerre(x, k-1) - (k-1) * laguerre(x, k-2));
}

inline arma::vec basis_laguerre(double x, double J)
{
    arma::vec basis(J);

    for (int i = 0; i < J; i++)
    {
        basis(i) = laguerre(x, i);
    }

    return basis;
}

// ========================================================================================================================

inline arma::vec laguerre(arma::vec x, double k)
{
    int vecSize = x.size();
    arma::vec y(vecSize);

    if (k == 0) {
        y.fill(1);
    }
    else if (k == 1)
        y = 1 - x;
    else
        y = (1/k) * ((2*k - 1 - x) % laguerre(x, k-1) - (k-1) * laguerre(x, k-2));

    return y;
}

inline arma::mat basis_laguerre(arma::vec x, double J)
{
    arma::mat y(J, x.size());

    for (int i = 1; i <= J; i++) {
        y.row(i-1) = laguerre(x.t(), i-1);
    }

    return y;
}

// ========================================================================================================================


#endif // UTIL_H
