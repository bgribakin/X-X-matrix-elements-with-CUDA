#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include "constants_types.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "\n\nGPUassert: %s %s %d\n\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ double psi_1s_QW(wf_parameter_struct* gpu_X_wf_params, double* wf, double rho, double z_e, double z_h);
__device__ double psi_1s_QW_analytical(double a0, double L, double S, double rho, double z_e, double z_h);
__device__ double V_I_pot(double r_e1h2, double r_e2h1, double r_e1e2, double r_h1h2);
__device__ double V_eh_pot(double r_eh, double fix);

__global__ void initRand(unsigned int seed, int runCounter, curandState_t* states);
__global__ void intMC_J_xx_exch(curandState_t* states, double* gpu_f, double* gpu_f2, double* gpu_wf, wf_parameter_struct* gpu_X_wf_params, double L, int dim, double q);
__global__ void intMC_J_xx_dir(curandState_t* states, double* gpu_f, double* gpu_f2, double* gpu_wf, wf_parameter_struct* gpu_X_wf_params, double L, int dim, double q);
__global__ void intMC_Ex(curandState_t* states, double* gpu_f, double* gpu_f2, double* gpu_wf, wf_parameter_struct* gpu_X_wf_params, double L);