#pragma once

#include <cstdio>
#include <cmath>
#include <ctime>
#include <cstring>
#include <conio.h>

#include <io.h>//access
#include <direct.h>//getcwd

const double pi = 3.14159265359;
// using SI! --------------------------------------------------------------------------------------------------------------
const double hbar = 1.054571817e-34; // J*s
const double e = 1.602176634e-19; // elementary charge, coulombs
const double eps0 = 8.85418781e-12; // permittivity of free space
const double eps = 12.53; // static dielectric constant

const double m0 = 9.109383561e-31; // free electron mass, kg
const double m_e = 0.067 * m0; // eff e mass in GaAs
const double m_hh = 0.377 * m0; // eff mass of heavy holes in GaAs
const double m_lh = 0.082 * m0; // light holes in GaAs
const double mu_hh = 1.0 / (1.0 / m_e + 1.0 / m_hh); // reduced mass of e-hh
const double mu_lh = 1.0 / (1.0 / m_e + 1.0 / m_lh); // e-lh
const double M_h = m_e + m_hh;
const double M_l = m_e + m_lh;

const double a0_hh = 4 * pi * eps0 * eps * (hbar * hbar) / mu_hh / (e * e); // Bohr radius of Xhh
const double a0_lh = 4 * pi * eps0 * eps * (hbar * hbar) / mu_lh / (e * e); // Bohr radius of Xlh

const double Rx = mu_hh * pow(e, 4) / (pow(4 * pi * eps0 * eps, 2) * 2 * pow(hbar, 2));
const double lambda_2d = 4 * pi * eps0 * eps * (hbar * hbar) / mu_hh / (e * e) / 2;
// QW wavefunction parameters ---------------------------------------------------------------------------------------

typedef struct { // struct type for loading and storing exciton wf parameters + fix, S_real and V_MC
	double L;
	double dZ;
	int sizeRho;
	int sizeZe;
	int sizeZh;
	double fix;
	double S_real;
	double V_MC;
	double V_MC_Ex;
} wf_parameter_struct;

// MC integration parameters ----------------------------------------------------------------------------------------------
const int numThrowsNorm = 1e7; // num points used to normalize

const int dim = 10;
const long long int numRun = 5e7;
const long long int N = 1000 * 1024;// NUM_SM* MAX_THREADS_PER_SM;//CUDA_CORES * dim * 10; // total number of points (~random numbers) to be thrown; preferably about numSM * maxThreadsPerSM
const int numPoints = N / dim; // number of dim-dimensional points

const double tol = 0.15; // tolerance for integral error; program terminates once tol is reached

// Set value of transferred momentum q:
const double q[] = { 0.0, 0.1 / a0_hh, 0.2 / a0_hh, 0.3 / a0_hh, 0.4 / a0_hh, 0.5 / a0_hh, 0.6 / a0_hh, 0.7 / a0_hh, 0.8 / a0_hh, 0.9 / a0_hh, 1.0 / a0_hh,
					1.5 / a0_hh, 2.0 / a0_hh, 3.0 / a0_hh , 4.0 / a0_hh, 5.0 / a0_hh , 6.0 / a0_hh , 7.0 / a0_hh, 8.0 / a0_hh }; // num_q = 19
//const double q = 0.0 / a0_hh;
//const double q[] = { 9.0 / a0_hh, 10.0 / a0_hh , 11.0 / a0_hh };