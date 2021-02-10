#pragma once
// this code is taken from https://github.com/mswiniars/Optimizing-Parallel-Reduction
// seems to work accurately and it's VERY fast
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

__global__ void reductionDouble(double* vect, double* vecOut, int size);
void sumGPUDouble(double* vector, double* vectorOutput, int vec_size);