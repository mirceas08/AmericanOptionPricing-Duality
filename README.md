Description
===========
This is the repository for the code used throughout my MSc thesis in Quantitative Finance. The analysis focused on the dual formulation of the American option pricing problem that provides an upper bound on the price on the option. The method makes use of Monte Carlo techniques and complements lower bounds produced by primal formulations. The main feature of option pricing through duality is the construction of a martingale vanishing at zero, achieved here by nested simulations. To overcome the growing computational complexity, the thesis discusses the Multilevel Monte Carlo technique which allowed a parallel implementation of the code.

The code is written in C++ in an object-oriented manner and MPI is used as a message passing system for what concerns parallel runs. Below is some information on code dependencies and modules available.

Dependencies
===========
* C++ and MPI compiler with C++11 support
* Armadillo: high quality C++ linear algebra library used throughout the whole code. It integrates with LAPACK and BLAS. Use Armadillo without installation and link against BLAS and LAPACK instead
* OpenBLAS: multi-threaded replacement of traditional BLAS. Recommended for signicantly higher performance. Otherwise link with traditional BLAS and LAPACK

Primal-dual executables
===========
* LSM (Longstaff-Schwartz algorithm)
* dualAVF (Dual martingale constructed through Approximate Value Functions)
* dualSR (Dual martingale constructed through Stopping Rules)

To compile the code type

>make

Run the code with

>./executable datafile

Acknowledgments
===========
* My supervisor, Professor Anna Battauz (Bocconi University), for the guidance and help throughout the thesis project
* Professor Francesca Maggioni (University of Bergamo) and CINECA Computing Centre for providing me computing hours on Cineca's distributed cluster Galileo

References
===========
[1] M.B. Giles. Multilevel Monte Carlo path simulation. Operations Research, 56(3),
2008.

[2] M.B. Haugh and L. Kogan. Pricing American options: a duality approach. Operations
Research, 52(2), 2004.

[3] L.C.G. Rogers. Monte Carlo valuation of American options. Mathematical Finance,
12(3), 2002.

[4] C. Sanderson. Armadillo: an open source C++ linear algebra library for fast prototyping
and computationally intensive experiments. Technical report, NICTA, 2010.

License
===========
Code freely available under the GNU General Public License.
