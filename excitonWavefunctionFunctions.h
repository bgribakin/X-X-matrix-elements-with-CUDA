#pragma once

#include "fileFunctions.h"
#include "constants_types.h"

int loadExcitonWfParams(char* filename, wf_parameter_struct* X_wf_params);
void load(char wavefunction_filenames[], double* wf, unsigned long long* file_size, wf_parameter_struct* X_wf_params);
void normalize(double* wf, int numThrowsNorm, wf_parameter_struct* X_wf_params);
void checkNormalizationMC(double* wf, int numThrowsCheckNorm, wf_parameter_struct* X_wf_params);