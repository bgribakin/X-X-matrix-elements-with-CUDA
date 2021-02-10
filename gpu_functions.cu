#include "gpu_functions.h"


/* ---------------------------------- __device__ functions (helper funcs, wave funcs, potential etc.) ------------------------------------------- */

/* returns the closest discrete wavefunction value for given continuous arguments*/
__device__ double psi_1s_QW(wf_parameter_struct* gpu_X_wf_params, double* wf, double rho, double z_e, double z_h) {

	double dZ = gpu_X_wf_params->dZ;
	int sizeRho = gpu_X_wf_params->sizeRho;
	int sizeZe = gpu_X_wf_params->sizeZe;
	int sizeZh = gpu_X_wf_params->sizeZh;

	int iRho, iZe, iZh;
	double psi;

	iRho = floor(rho / dZ + 0.5) - 1;
	iZe = floor((z_e + dZ * (sizeZe + 1) / 2) / dZ + 0.5);
	iZh = floor((z_h + dZ * (sizeZe + 1) / 2) / dZ + 0.5);

	int index = iRho + iZe * sizeRho + iZh * sizeZe * sizeRho;
	if (iRho < 0 || iZe < 0 || iZh < 0) {// || index > sizeRho * sizeZe * sizeZh) {
//		printf("\n Illegal memory access: index = %d, iRho = %d, iZe = %d, iZh = %d,   wf[index] set to 0                                                      \n ", index, iRho, iZe, iZh);
		psi = 0;
	}
	else if (iRho >= sizeRho || iZe >= sizeZe || iZh >= sizeZh)
		psi = 0;
	else
		psi = wf[index];

	return psi;
}

__device__ double psi_1s_QW_analytical(double a0, double L, double S, double rho, double z_e, double z_h) {
	const double pi = 3.14159265359;
	double psi;

	if (abs(z_e) <= L / 2 && abs(z_h) <= L / 2)
		psi = 4 / (a0 * L) / sqrt(2 * pi) / sqrt(S) * exp(-rho / (a0)) * cos(pi / L * z_e) * cos(pi / L * z_h);
	else
		psi = 0;

	return psi;
}

/*  calculates the additional potential between in X-e system (excluding e1-h1)
fix is the 'nonzeroness': V = const / (r + fix) */
__device__ double V_I_pot(double r_e1h2, double r_e2h1, double r_e1e2, double r_h1h2) {
	const double pi = 3.14159265359;
	const double e2eps = 1.8412430591e-29; // in SI (J*m); e2eps = e^2/(4pi * eps * eps0), so V(r) = e2eps/r
	// 1.8412430591e-29 ~ eps = 12.53, 1.78843221e-29 ~ eps = 12.9
	const double a0_hh = 1.152714e-08;// m

	double V_I;
	// we introduce 'fix' to never worry about NaNs
	/*if (r_e1h2 == 0 || r_e2h1 == 0 || r_e1e2 == 0 || r_h1h2 == 0)
		V_I = e2eps * (1 / (r_e1e2 + fix) + 1 / (r_h1h2 + fix) - 1 / (r_e1h2 + fix) - 1 / (r_e1h2 + fix)); // sum of coulomb potentials
	else*/
	V_I = e2eps * (1 / r_e1e2 + 1 / r_h1h2 - 1 / r_e1h2 - 1 / r_e2h1);

	return V_I;
}

/* e-h attraction potential
fix is the 'nonzeroness': V = const / (r + fix) */
__device__ double V_eh_pot(double r_eh, double fix) {
	const double pi = 3.14159265359;
	const double e2eps = 1.8412430591e-29; // in SI (J*m); e2eps = e^2/(4pi * eps * eps0), so V(r) = e2eps/r
	// 1.8412430591e-29 ~ eps = 12.53, 1.78843221e-29 ~ eps = 12.9
	const double a0_hh = 1.152714e-08;// m

	double V_eh = e2eps * 1 / (r_eh + fix);

	return V_eh;
}

/* ------------------------------------------------ __global__ kernel functions ------------------------------------------------------------------  */

/* used to initialize the random states */
__global__ void initRand(unsigned int seed, int runCounter, curandState_t* states) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	// nvidia recommends using the same seed but monotonically increasing sequence numbers when dealing with multiple kernel launches
	// but that is much much slower, so best to leave runCounter=0
	for (int i = index; i < N; i += stride)
		curand_init(seed, N * runCounter + i, 0, &states[i]);
}

/* calculates J_exch^{e-e}(q) using MC method
   stores numPoints function values in gpu_f and as much squares in gpu_f2 */
__global__ void intMC_J_xx_exch(curandState_t* states, double* gpu_f, double* gpu_f2, double* gpu_wf, wf_parameter_struct* gpu_X_wf_params, double L, int dim, double q) {
	const double pi = 3.14159265359;

	const double m0 = 9.109383561e-31; // free electron mass, kg
	const double m_e = 0.067 * m0; // eff e mass in GaAs
	const double m_hh = 0.35 * m0; // eff mass of heavy holes in GaAs along z
	const double mu_hh = 0.0417 * m0; // 1.0 / (1.0 / m_e + 1.0 / m_hh); // reduced mass of e-hh with in-plane m_hh
	const double M = m_e + m_hh;
	const double a0_hh = 1.152714e-08; // Xhh Bohr radius, m

	const double e = 1.602176634e-19; // elementary charge, coulombs	
	const double e2eps = 1.8412430591e-29; // in SI (J*m); e2eps = e^2/(4pi * eps * eps0), so V(r) = e2eps/r
	// 1.8412430591e-29 ~ eps = 12.53, 1.78843221e-29 ~ eps = 12.9

	double dZ = gpu_X_wf_params->dZ;
	int sizeRho = gpu_X_wf_params->sizeRho;
	int sizeZe = gpu_X_wf_params->sizeZe;
	int sizeZh = gpu_X_wf_params->sizeZh;
	double S_real = gpu_X_wf_params->S_real;

	// 2d polar relative coordinates give detTheta = 1; 2d centre-mass coords are integrated and give an S multiplier
	// we are left with (rho_eh, phi_eh)x2 + (xi, phi_xi) + (z_e1, z_e2, z_h1, z_h2) -- 10 coords in total

	double rho_e1h1, phi_e1h1, z_e1, z_h1;
	double rho_e2h2, phi_e2h2, z_e2, z_h2;
	double xi, phi_xi;

	double rho_e1h2, rho_e2h1; // 2d distances for psi_e1h2, psi_e2h1
	double r_e1h2 = 0, r_e2h1 = 0, r_e1e2 = 0, r_h1h2 = 0; // 3d distances for potential V_I

	double psi_e1h1, psi_e2h2, psi_e1h2, psi_e2h1; // exciton wavefunctions
	double V_I; // value of potential V_I
	double q_factor_real, q_factor_im, q_factor_arg; // contain q-dependency assuming Q = Q' = 0

	double detTheta; // Jacobian

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double rands[10];

	for (int i = index; i < numPoints; i += stride) {
		rands[0] = curand_uniform_double(&states[dim * i + 0]);
		rands[1] = curand_uniform_double(&states[dim * i + 1]);
		rands[2] = curand_uniform_double(&states[dim * i + 2]);
		rands[3] = curand_uniform_double(&states[dim * i + 3]);
		rands[4] = curand_uniform_double(&states[dim * i + 4]);
		rands[5] = curand_uniform_double(&states[dim * i + 5]);
		rands[6] = curand_uniform_double(&states[dim * i + 6]);
		rands[7] = curand_uniform_double(&states[dim * i + 7]);
		rands[8] = curand_uniform_double(&states[dim * i + 8]);
		rands[9] = curand_uniform_double(&states[dim * i + 9]);

		rho_e1h1 = dZ * (1.0 + rands[0] * (sizeRho + 1));
		phi_e1h1 = 2 * pi * rands[1];
		z_e1 = dZ * (rands[2] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);
		z_h1 = dZ * (rands[3] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);

		rho_e2h2 = dZ * (1.0 + rands[4] * (sizeRho + 1));
		phi_e2h2 = 2 * pi * rands[5];
		z_e2 = dZ * (rands[6] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);
		z_h2 = dZ * (rands[7] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);

		// theoretically '2 * (sizeRho + 1)' is the largest upper bound for xi, larger ones lead to gpu_f[i] = 0
		// investigate!
		xi = dZ + dZ * rands[8] * (2 * (sizeRho + 1));
		phi_xi = 2 * pi * rands[9];

		// now let's calculate other necessary distances: rho/r_e1h2, rho/r_e2h1, r_e1e2, r_h1h2 -------------------------------------------------

		double rho_1_sq = pow(rho_e1h1, 2);
		double rho_2_sq = pow(rho_e2h2, 2);
		double xi_sq = pow(xi, 2);
		double rho_e1h2_sq, rho_e2h1_sq, rho_e1e2_sq, rho_h1h2_sq;

		// doubled scalar products:
		double t_xi_rho_1 = 2 * xi * rho_e1h1 * cos(phi_xi - phi_e1h1);
		double t_xi_rho_2 = 2 * xi * rho_e2h2 * cos(phi_xi - phi_e2h2);
		double t_rho_1_rho_2 = 2 * rho_e1h1 * rho_e2h2 * cos(phi_e1h1 - phi_e2h2);

		// assemble necessary 2d vector squares:
		rho_e1h2_sq = (xi_sq + pow(m_hh / M, 2) * rho_1_sq + pow(m_e / M, 2) * rho_2_sq // for w.f.and potential
			+ m_hh / M * t_xi_rho_1
			+ m_e / M * t_xi_rho_2
			+ m_e * m_hh / pow(M, 2) * t_rho_1_rho_2);

		rho_e2h1_sq = (xi_sq + pow(m_e / M, 2) * rho_1_sq + pow(m_hh / M, 2) * rho_2_sq // for w.f.and potential
			- m_e / M * t_xi_rho_1
			- m_hh / M * t_xi_rho_2
			+ m_e * m_hh / pow(M, 2) * t_rho_1_rho_2);

		rho_e1e2_sq = (xi_sq + pow(m_hh / M, 2) * (rho_1_sq + rho_2_sq) // only for the potential
			+ m_hh / M * t_xi_rho_1
			- m_hh / M * t_xi_rho_2
			- pow(m_hh / M, 2) * t_rho_1_rho_2);

		rho_h1h2_sq = (xi_sq + pow(m_e / M, 2) * (rho_1_sq + rho_2_sq) // only for the potential
			- m_e / M * t_xi_rho_1
			+ m_e / M * t_xi_rho_2
			- pow(m_e / M, 2) * t_rho_1_rho_2);

		// assemble 3d distances for V_I potential:
		r_e1h2 = sqrt(rho_e1h2_sq + pow(z_e1 - z_h2, 2));
		r_e2h1 = sqrt(rho_e2h1_sq + pow(z_e2 - z_h1, 2));
		r_e1e2 = sqrt(rho_e1e2_sq + pow(z_e1 - z_e2, 2));
		r_h1h2 = sqrt(rho_h1h2_sq + pow(z_h1 - z_h2, 2));

		// get 2d vector lengths for psi_e1h2, psi_e2h1:
		rho_e1h2 = sqrt(rho_e1h2_sq);
		rho_e2h1 = sqrt(rho_e2h1_sq);

		// now, calculate wavefunctions:
		psi_e1h1 = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_e1h1, z_e1, z_h1);
		psi_e2h2 = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_e2h2, z_e2, z_h2);
		psi_e1h2 = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_e1h2, z_e1, z_h2);
		psi_e2h1 = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_e2h1, z_e2, z_h1);

		// calculate potential:
		V_I = V_I_pot(r_e1h2, r_e2h1, r_e1e2, r_h1h2);

		// evalute q-factor
		double q_factor_arg = q * (m_hh - m_e) / M * xi * cos(phi_xi) 
							+ q * (2 * mu_hh / M * (rho_e2h2 * cos(phi_e2h2) - rho_e1h1 * cos(phi_e1h1)));
		q_factor_real = cos(q_factor_arg);
		q_factor_im = sin(q_factor_arg);

		// finally, calculate the Jacobian:
		detTheta = rho_e1h1 * rho_e2h2 * xi;

		// now we simply evaluate the complete integrand
		gpu_f[i] = S_real * detTheta * q_factor_real * psi_e1h2 * psi_e2h1 * V_I * psi_e1h1 * psi_e2h2 / e * 1e6 * (S_real * 1e12); // in micro eV * micro m
		//gpu_f[i] = S_real * detTheta * psi_e1h1 * psi_e1h1 * psi_e2h2 * psi_e2h2; // double norm
		//gpu_f[i] = S_real * detTheta * psi_e1h1 * psi_e2h2 * psi_e1h2 * psi_e2h1; // overlap integral
		//printf("\ngpu_f[%d] = S_real * detTheta * psi_e1h1 * psi_e2 * (V_I / e * 1e6) * psi_e2h1 * psi_e1 = %e * %e * %e *  %e * %e * %e * %e", i, S_real, detTheta, psi_e1h1, psi_e2, (V_I / e * 1e6), psi_e2h1, psi_e1);

		gpu_f2[i] = gpu_f[i] * gpu_f[i]; // here we store their squares to get <f^2> -> int error
	}
}

/* calculates J_exch^{e-e}(q) using MC method
   stores numPoints function values in gpu_f and as much squares in gpu_f2 */
__global__ void intMC_J_xx_exch_hh(curandState_t* states, double* gpu_f, double* gpu_f2, double* gpu_wf, wf_parameter_struct* gpu_X_wf_params, double L, int dim, double q) {
	const double pi = 3.14159265359;

	const double m0 = 9.109383561e-31; // free electron mass, kg
	const double m_e = 0.067 * m0; // eff e mass in GaAs
	const double m_hh = 0.35 * m0; // eff mass of heavy holes in GaAs along z
	const double mu_hh = 0.0417 * m0; // 1.0 / (1.0 / m_e + 1.0 / m_hh); // reduced mass of e-hh with in-plane m_hh
	const double M = m_e + m_hh;
	const double a0_hh = 1.152714e-08; // Xhh Bohr radius, m

	const double e = 1.602176634e-19; // elementary charge, coulombs	
	const double e2eps = 1.8412430591e-29; // in SI (J*m); e2eps = e^2/(4pi * eps * eps0), so V(r) = e2eps/r
	// 1.8412430591e-29 ~ eps = 12.53, 1.78843221e-29 ~ eps = 12.9

	double dZ = gpu_X_wf_params->dZ;
	int sizeRho = gpu_X_wf_params->sizeRho;
	int sizeZe = gpu_X_wf_params->sizeZe;
	int sizeZh = gpu_X_wf_params->sizeZh;
	double S_real = gpu_X_wf_params->S_real;

	// 2d polar relative coordinates give detTheta = 1; 2d centre-mass coords are integrated and give an S multiplier
	// we are left with (rho_eh, phi_eh)x2 + (xi, phi_xi) + (z_e1, z_e2, z_h1, z_h2) -- 10 coords in total

	double rho_e1h1, phi_e1h1, z_e1, z_h1;
	double rho_e2h2, phi_e2h2, z_e2, z_h2;
	double xi, phi_xi;

	double rho_e1h2, rho_e2h1; // 2d distances for psi_e1h2, psi_e2h1
	double r_e1h2 = 0, r_e2h1 = 0, r_e1e2 = 0, r_h1h2 = 0; // 3d distances for potential V_I

	double psi_e1h1, psi_e2h2, psi_e1h2, psi_e2h1; // exciton wavefunctions
	double V_I; // value of potential V_I
	double q_factor_real, q_factor_im, q_factor_arg; // contain q-dependency assuming Q = Q' = 0

	double detTheta; // Jacobian

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double rands[10];

	for (int i = index; i < numPoints; i += stride) {
		rands[0] = curand_uniform_double(&states[dim * i + 0]);
		rands[1] = curand_uniform_double(&states[dim * i + 1]);
		rands[2] = curand_uniform_double(&states[dim * i + 2]);
		rands[3] = curand_uniform_double(&states[dim * i + 3]);
		rands[4] = curand_uniform_double(&states[dim * i + 4]);
		rands[5] = curand_uniform_double(&states[dim * i + 5]);
		rands[6] = curand_uniform_double(&states[dim * i + 6]);
		rands[7] = curand_uniform_double(&states[dim * i + 7]);
		rands[8] = curand_uniform_double(&states[dim * i + 8]);
		rands[9] = curand_uniform_double(&states[dim * i + 9]);

		rho_e1h1 = dZ * (1.0 + rands[0] * (sizeRho + 1));
		phi_e1h1 = 2 * pi * rands[1];
		z_e1 = dZ * (rands[2] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);
		z_h1 = dZ * (rands[3] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);

		rho_e2h2 = dZ * (1.0 + rands[4] * (sizeRho + 1));
		phi_e2h2 = 2 * pi * rands[5];
		z_e2 = dZ * (rands[6] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);
		z_h2 = dZ * (rands[7] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);

		// theoretically '2 * (sizeRho + 1)' is the largest upper bound for xi, larger ones lead to gpu_f[i] = 0
		// investigate!
		xi = dZ + dZ * rands[8] * (2 * (sizeRho + 1));
		phi_xi = 2 * pi * rands[9];

		// now let's calculate other necessary distances: rho/r_e1h2, rho/r_e2h1, r_e1e2, r_h1h2 -------------------------------------------------

		double rho_1_sq = pow(rho_e1h1, 2);
		double rho_2_sq = pow(rho_e2h2, 2);
		double xi_sq = pow(xi, 2);
		double rho_e1h2_sq, rho_e2h1_sq, rho_e1e2_sq, rho_h1h2_sq;

		// doubled scalar products:
		double t_xi_rho_1 = 2 * xi * rho_e1h1 * cos(phi_xi - phi_e1h1);
		double t_xi_rho_2 = 2 * xi * rho_e2h2 * cos(phi_xi - phi_e2h2);
		double t_rho_1_rho_2 = 2 * rho_e1h1 * rho_e2h2 * cos(phi_e1h1 - phi_e2h2);

		// assemble necessary 2d vector squares:
		rho_e1h2_sq = (xi_sq + pow(m_hh / M, 2) * rho_1_sq + pow(m_e / M, 2) * rho_2_sq // for w.f.and potential
			+ m_hh / M * t_xi_rho_1
			+ m_e / M * t_xi_rho_2
			+ m_e * m_hh / pow(M, 2) * t_rho_1_rho_2);

		rho_e2h1_sq = (xi_sq + pow(m_e / M, 2) * rho_1_sq + pow(m_hh / M, 2) * rho_2_sq // for w.f.and potential
			- m_e / M * t_xi_rho_1
			- m_hh / M * t_xi_rho_2
			+ m_e * m_hh / pow(M, 2) * t_rho_1_rho_2);

		rho_e1e2_sq = (xi_sq + pow(m_hh / M, 2) * (rho_1_sq + rho_2_sq) // only for the potential
			+ m_hh / M * t_xi_rho_1
			- m_hh / M * t_xi_rho_2
			- pow(m_hh / M, 2) * t_rho_1_rho_2);

		rho_h1h2_sq = (xi_sq + pow(m_e / M, 2) * (rho_1_sq + rho_2_sq) // only for the potential
			- m_e / M * t_xi_rho_1
			+ m_e / M * t_xi_rho_2
			- pow(m_e / M, 2) * t_rho_1_rho_2);

		// assemble 3d distances for V_I potential:
		r_e1h2 = sqrt(rho_e1h2_sq + pow(z_e1 - z_h2, 2));
		r_e2h1 = sqrt(rho_e2h1_sq + pow(z_e2 - z_h1, 2));
		r_e1e2 = sqrt(rho_e1e2_sq + pow(z_e1 - z_e2, 2));
		r_h1h2 = sqrt(rho_h1h2_sq + pow(z_h1 - z_h2, 2));

		// get 2d vector lengths for psi_e1h2, psi_e2h1:
		rho_e1h2 = sqrt(rho_e1h2_sq);
		rho_e2h1 = sqrt(rho_e2h1_sq);

		// now, calculate wavefunctions:
		psi_e1h1 = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_e1h1, z_e1, z_h1);
		psi_e2h2 = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_e2h2, z_e2, z_h2);
		psi_e1h2 = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_e1h2, z_e1, z_h2);
		psi_e2h1 = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_e2h1, z_e2, z_h1);

		// calculate potential:
		V_I = V_I_pot(r_e1h2, r_e2h1, r_e1e2, r_h1h2);

		// evalute q-factor
		double q_factor_arg = q * (m_hh - m_e) / M * xi * cos(phi_xi)
			+ q * (2 * mu_hh / M * (rho_e2h2 * cos(phi_e2h2) - rho_e1h1 * cos(phi_e1h1)));
		q_factor_real = cos(q_factor_arg);
		q_factor_im = sin(q_factor_arg);

		// finally, calculate the Jacobian:
		detTheta = rho_e1h1 * rho_e2h2 * xi;

		// now we simply evaluate the complete integrand
		gpu_f[i] = S_real * detTheta * q_factor_real * psi_e1h2 * psi_e2h1 * V_I * psi_e1h1 * psi_e2h2 / e * 1e6 * (S_real * 1e12); // in micro eV * micro m
		//gpu_f[i] = S_real * detTheta * psi_e1h1 * psi_e1h1 * psi_e2h2 * psi_e2h2; // double norm
		//gpu_f[i] = S_real * detTheta * psi_e1h1 * psi_e2h2 * psi_e1h2 * psi_e2h1; // overlap integral
		//printf("\ngpu_f[%d] = S_real * detTheta * psi_e1h1 * psi_e2 * (V_I / e * 1e6) * psi_e2h1 * psi_e1 = %e * %e * %e *  %e * %e * %e * %e", i, S_real, detTheta, psi_e1h1, psi_e2, (V_I / e * 1e6), psi_e2h1, psi_e1);

		gpu_f2[i] = gpu_f[i] * gpu_f[i]; // here we store their squares to get <f^2> -> int error
	}
}

/* calculates J_dir(q) using MC method
   stores numPoints function values in gpu_f and as much squares in gpu_f2 */
__global__ void intMC_J_xx_dir(curandState_t* states, double* gpu_f, double* gpu_f2, double* gpu_wf, wf_parameter_struct* gpu_X_wf_params, double L, int dim, double q) {
	const double pi = 3.14159265359;

	const double m0 = 9.109383561e-31; // free electron mass, kg
	const double m_e = 0.067 * m0; // eff e mass in GaAs
	const double m_hh = 0.35 * m0; // eff mass of heavy holes in GaAs along z
	const double mu_hh = 0.0417 * m0; // 1.0 / (1.0 / m_e + 1.0 / m_hh); // reduced mass of e-hh with in-plane m_hh 
	const double M = m_e + m_hh;
	const double a0_hh = 1.152714e-08; // Xhh Bohr radius, m

	const double e = 1.602176634e-19; // elementary charge, coulombs	
	const double e2eps = 1.8412430591e-29; // in SI (J*m); e2eps = e^2/(4pi * eps * eps0), so V(r) = e2eps/r
	// 1.8412430591e-29 ~ eps = 12.53, 1.78843221e-29 ~ eps = 12.9

	double dZ = gpu_X_wf_params->dZ;
	int sizeRho = gpu_X_wf_params->sizeRho;
	int sizeZe = gpu_X_wf_params->sizeZe;
	int sizeZh = gpu_X_wf_params->sizeZh;
	double S_real = gpu_X_wf_params->S_real;

	// 2d polar relative coordinates give detTheta = 1; 2d centre-mass coords are integrated and give an S multiplier
	// we are left with (rho_eh, phi_eh)x2 + (xi, phi_xi) + (z_e1, z_e2, z_h1, z_h2) -- 10 coords in total

	double rho_e1h1, phi_e1h1, z_e1, z_h1;
	double rho_e2h2, phi_e2h2, z_e2, z_h2;
	double xi, phi_xi;

	double r_e1h2 = 0, r_e2h1 = 0, r_e1e2 = 0, r_h1h2 = 0; // 3d distances for potential V_I

	double psi_e1h1, psi_e2h2; // exciton wavefunctions
	double V_I; // value of potential V_I
	double q_factor_real, q_factor_im, q_factor_arg; // contains q-dependency assuming Q = Q' = 0

	double detTheta; // Jacobian

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double rands[10];

	for (int i = index; i < numPoints; i += stride) {
		rands[0] = curand_uniform_double(&states[dim * i + 0]);
		rands[1] = curand_uniform_double(&states[dim * i + 1]);
		rands[2] = curand_uniform_double(&states[dim * i + 2]);
		rands[3] = curand_uniform_double(&states[dim * i + 3]);
		rands[4] = curand_uniform_double(&states[dim * i + 4]);
		rands[5] = curand_uniform_double(&states[dim * i + 5]);
		rands[6] = curand_uniform_double(&states[dim * i + 6]);
		rands[7] = curand_uniform_double(&states[dim * i + 7]);
		rands[8] = curand_uniform_double(&states[dim * i + 8]);
		rands[9] = curand_uniform_double(&states[dim * i + 9]);

		rho_e1h1 = dZ * (1.0 + rands[0] * (sizeRho + 1));
		phi_e1h1 = 2 * pi * rands[1];
		z_e1 = dZ * (rands[2] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);
		z_h1 = dZ * (rands[3] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);

		rho_e2h2 = dZ * (1.0 + rands[4] * (sizeRho + 1));
		phi_e2h2 = 2 * pi * rands[5];
		z_e2 = dZ * (rands[6] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);
		z_h2 = dZ * (rands[7] * (sizeZe + 1) - (double)(sizeZe + 1) / 2);

		// theoretically '2 * (sizeRho + 1)' is the largest upper bound for xi, larger ones lead to gpu_f[i] = 0
		// investigate!
		xi = dZ + dZ * rands[8] * (2 * (sizeRho + 1));
		phi_xi = 2 * pi * rands[9];

		// now let's calculate other necessary distances: rho/r_e1h2, rho/r_e2h1, r_e1e2, r_h1h2 -------------------------------------------------

		double rho_1_sq = pow(rho_e1h1, 2);
		double rho_2_sq = pow(rho_e2h2, 2);
		double xi_sq = pow(xi, 2);
		double rho_e1h2_sq, rho_e2h1_sq, rho_e1e2_sq, rho_h1h2_sq;

		// doubled scalar products:
		double t_xi_rho_1 = 2 * xi * rho_e1h1 * cos(phi_xi - phi_e1h1);
		double t_xi_rho_2 = 2 * xi * rho_e2h2 * cos(phi_xi - phi_e2h2);
		double t_rho_1_rho_2 = 2 * rho_e1h1 * rho_e2h2 * cos(phi_e1h1 - phi_e2h2);

		// assemble necessary 2d vector squares:
		rho_e1h2_sq = (xi_sq + pow(m_hh / M, 2) * rho_1_sq + pow(m_e / M, 2) * rho_2_sq // for w.f.and potential
			+ m_hh / M * t_xi_rho_1
			+ m_e / M * t_xi_rho_2
			+ m_e * m_hh / pow(M, 2) * t_rho_1_rho_2);

		rho_e2h1_sq = (xi_sq + pow(m_e / M, 2) * rho_1_sq + pow(m_hh / M, 2) * rho_2_sq // for w.f.and potential
			- m_e / M * t_xi_rho_1
			- m_hh / M * t_xi_rho_2
			+ m_e * m_hh / pow(M, 2) * t_rho_1_rho_2);

		rho_e1e2_sq = (xi_sq + pow(m_hh / M, 2) * (rho_1_sq + rho_2_sq) // only for the potential
			+ m_hh / M * t_xi_rho_1
			- m_hh / M * t_xi_rho_2
			- pow(m_hh / M, 2) * t_rho_1_rho_2);

		rho_h1h2_sq = (xi_sq + pow(m_e / M, 2) * (rho_1_sq + rho_2_sq) // only for the potential
			- m_e / M * t_xi_rho_1
			+ m_e / M * t_xi_rho_2
			- pow(m_e / M, 2) * t_rho_1_rho_2);

		// assemble 3d distances for V_I potential:
		r_e1h2 = sqrt(rho_e1h2_sq + pow(z_e1 - z_h2, 2));
		r_e2h1 = sqrt(rho_e2h1_sq + pow(z_e2 - z_h1, 2));
		r_e1e2 = sqrt(rho_e1e2_sq + pow(z_e1 - z_e2, 2));
		r_h1h2 = sqrt(rho_h1h2_sq + pow(z_h1 - z_h2, 2));

		// now, calculate wavefunctions:
		psi_e1h1 = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_e1h1, z_e1, z_h1);
		psi_e2h2 = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_e2h2, z_e2, z_h2);

		// calculate potential:
		V_I = V_I_pot(r_e1h2, r_e2h1, r_e1e2, r_h1h2);

		// evalute q-factor
		double q_factor_arg = q * xi * cos(phi_xi);
		q_factor_real = cos(q_factor_arg);
		q_factor_im = sin(q_factor_arg);

		// finally, calculate the Jacobian:
		detTheta = rho_e1h1 * rho_e2h2 * xi;

		// now we simply evaluate the complete integrand
		gpu_f[i] = S_real * detTheta * q_factor_real * psi_e1h1 * psi_e2h2 * V_I * psi_e1h1 * psi_e2h2 / e * 1e6 * (S_real * 1e12); // in micro eV * micro m
		//gpu_f[i] = S_real * detTheta * psi_e1h1 * psi_e1h1 * psi_e2h2 * psi_e2h2; // double norm
		//gpu_f[i] = S_real * detTheta * psi_e1h1 * psi_e2h2 * psi_e1h2 * psi_e2h1; // overlap integral
		//printf("\ngpu_f[%d] = S_real * detTheta * psi_e1h1 * psi_e2 * (V_I / e * 1e6) * psi_e2h1 * psi_e1 = %e * %e * %e *  %e * %e * %e * %e", i, S_real, detTheta, psi_e1h1, psi_e2, (V_I / e * 1e6), psi_e2h1, psi_e1);

		gpu_f2[i] = gpu_f[i] * gpu_f[i]; // here we store their squares to get <f^2> -> int error
	}
}

/* same, but calculates Ex: <psi | V_eh | psi>*/
__global__ void intMC_Ex(curandState_t* states, double* gpu_f, double* gpu_f2, double* gpu_wf, wf_parameter_struct* gpu_X_wf_params, double L) {
	const double pi = 3.14159265359;

	const double m0 = 9.109383561e-31; // free electron mass, kg
	const double m_e = 0.067 * m0; // eff e mass in GaAs
	const double m_hh = 0.51 * m0; // eff mass of heavy holes in GaAs    
	const double mu_hh = 1.0 / (1.0 / m_e + 1.0 / m_hh); // reduced mass of e-hh   
	const double M_h = m_e + m_hh;
	const double a0_hh = 1.152714e-08; // Xhh Bohr radius, m

	const double e = 1.602176634e-19; // elementary charge, coulombs
	const double e2eps = 1.7884322117e-29; // in SI (J*m); e2eps = e^2/(4pi * eps * eps0), so V(r) = e2eps/r

	double dZ = gpu_X_wf_params->dZ;
	int sizeRho = gpu_X_wf_params->sizeRho;
	int sizeZe = gpu_X_wf_params->sizeZe;
	int sizeZh = gpu_X_wf_params->sizeZh;
	double S_real = gpu_X_wf_params->S_real;
	double fix = gpu_X_wf_params->fix;

	// 2d polar relative coordinates give detTheta = 1; 2d centre-mass coords are integrated and give an S multiplier
	// we are left with (rho_eh, phi_eh)x2 + (z_e1, z_e2, z_h) -- 7 coords in total
	double rho_eh, phi_eh, z_e, z_h;
	double r_eh = 0;// distances for potential V_I

	double psi_eh; // exc. wavefunc of exciton
	double V_eh; // value of potential V_eh_pot
	double detTheta;

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < numPoints; i += stride) {
		rho_eh = curand_uniform(&states[dim * i + 0]) * dZ * (sizeRho - 1);
		phi_eh = curand_uniform(&states[dim * i + 2]) * 2 * pi;
		z_e = curand_uniform(&states[dim * i + 4]) * dZ * (sizeZe - 1);
		z_h = curand_uniform(&states[dim * i + 6]) * dZ * (sizeZh - 1);

		// calc distances for potential (simple to check expressions)
		r_eh = sqrt(pow(rho_eh, 2) + pow(z_e - z_h, 2));

		// evaluate the V_I potential
		V_eh = V_eh_pot(r_eh, fix);

		// evaluate wave functions
		psi_eh = psi_1s_QW(gpu_X_wf_params, gpu_wf, rho_eh, z_e, z_h);

		// don't forget about the jacobian
		detTheta = rho_eh;// jacobi determinant of new symmetrical double cylindrical relative coordinates

		// now we simply evaluate the complete integrand
		gpu_f[i] = S_real * detTheta * psi_eh * V_eh * psi_eh / e * 1e6; // in micro eV
		//printf("\ngpu_f[%d] = S_real * detTheta * psi_e1h1 * psi_e2 * (V_I / e * 1e6) * psi_e2h1 * psi_e1 = %e * %e * %e *  %e * %e * %e * %e", i, S_real, detTheta, psi_e1h1, psi_e2, (V_I / e * 1e6), psi_e2h1, psi_e1);
		gpu_f2[i] = gpu_f[i] * gpu_f[i]; // here we store their squares to get <f^2> -> int error
	}
}
