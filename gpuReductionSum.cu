#include "gpuReductionSum.h"

using namespace std;
using namespace std::chrono;

#define DIM 1024

__global__ void reductionDouble(double* vect, double* vecOut, int size)
{
	__shared__ double block[DIM];
	unsigned int globalIndex = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int i = threadIdx.x;
	if (globalIndex < size)
		block[i] = vect[globalIndex];
	else
		block[i] = 0;

	__syncthreads();

	for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1)
	{
		if (i < j)
			block[i] += block[i + j];

		__syncthreads();
	}
	if (i == 0)
		vecOut[blockIdx.x] = block[0];
}

void sumGPUDouble(double* vector, double* vectorOutput, int vec_size)
{
	int numInputElements = vec_size;
	int numOutputElements;
	int threadsPerBlock = DIM;
	double* dev_vec;
	double* dev_vecOut;

	cudaSetDevice(0);
	cudaMalloc((double**)&dev_vec, vec_size * sizeof(double));
	cudaMalloc((double**)&dev_vecOut, vec_size * sizeof(double));
	cudaMemcpy(dev_vec, vector, vec_size * sizeof(double), cudaMemcpyHostToDevice);

	do
	{
		numOutputElements = numInputElements / (threadsPerBlock);
		if (numInputElements % (threadsPerBlock))
			numOutputElements++;
		reductionDouble << < numOutputElements, threadsPerBlock >> > (dev_vec, dev_vecOut, numInputElements);
		numInputElements = numOutputElements;
		if (numOutputElements > 1)
			reductionDouble << < numOutputElements, threadsPerBlock >> > (dev_vecOut, dev_vec, numInputElements);

	} while (numOutputElements > 1);

	cudaDeviceSynchronize();
	cudaMemcpy(vector, dev_vec, vec_size * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(vectorOutput, dev_vecOut, vec_size * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(dev_vec);
	cudaFree(dev_vecOut);
}
