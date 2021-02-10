/*
  This version of the program calculates the X-X exchange constant. Coordinates: (z_e1, z_h1, z_e2, z_h2) + (r_e1h1, r_e2h2, xi)^(2d), xi = R1 - R2

  In this version, the parameters dZ, sizeRho, sizeZe, sizeZh are retrieved from text files (made by the 'run_one_direct_calc' program) automatically

  The paths must be rewritten for each machine's folder structure and project location
  The wf names are recommended to only contain the width of the QW (e.g. "WQW=Xnm.bin")

*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include "constants_types.h"
#include "fileFunctions.h"
#include "excitonWavefunctionFunctions.h"
#include "gpu_functions.h"
#include "gpuReductionSum.h"

int main()
{
	time_t tic;
	time_t toc;
	time_t seed;

	FILE* filenameFile; // file with filenames to process
	filenameFile = fopen("filenames.txt", "r");

	int bufferLength = 255;
	char* buffer = new char[bufferLength];
	char* X_wf_params_filename = new char[bufferLength];
	char* X_wf_filename = new char[bufferLength];

	int counter = 0;
	while (fgets(buffer, bufferLength, filenameFile)) {
		if (strcmp(buffer, "end")) {
			counter++;
		}
	}
	printf("Calculations to process: %d\n\n", counter);
	rewind(filenameFile);
	printFile("filenames.txt", filenameFile);
	printf("==================================================================================================\n");

	while (fgets(buffer, bufferLength, filenameFile)) {
		if (strcmp(buffer, "end")) { // returns 0 when buffer == "end"

			/* create filenames for *.txt and *.bin files */
			strncpy(X_wf_params_filename, buffer, strlen(buffer) - 1); // a filename shouldn't end with \n			
			X_wf_params_filename[strlen(buffer) - 1] = '\0'; // strncpy doesn't append 'filename' with a proper end-symbol, do it manually
			strcat(X_wf_params_filename, ".txt");

			strncpy(X_wf_filename, buffer, strlen(buffer) - 1);
			X_wf_filename[strlen(buffer) - 1] = '\0';
			strcat(X_wf_filename, ".bin");

			/* load X wf parameters --------------------------------------------------------------------------------------------------------------------------------     */
			printf("  Exciton wavefunction parameters *.txt file:\n\t'%s'\n", X_wf_params_filename);

			wf_parameter_struct* X_wf_params;
			wf_parameter_struct* gpu_X_wf_params;
			X_wf_params = new wf_parameter_struct();
			if (loadExcitonWfParams(X_wf_params_filename, X_wf_params) == 0) {// loaded exc. wf parameters into cpu memory
				delete X_wf_params;
				printf("\nSkipping to next file...\n==================================================================================================\n");
				continue;
			}

			gpuErrchk(cudaMalloc((void**)&gpu_X_wf_params, sizeof(wf_parameter_struct)));
			gpuErrchk(cudaMemcpy(gpu_X_wf_params, X_wf_params, sizeof(wf_parameter_struct), cudaMemcpyHostToDevice)); // copied the params into gpu memory as well

			/* load X wavefunction ---------------------------------------------------------------------------------------------------------------------------------		*/
			double* wf, * gpu_wf;
			wf = new double[X_wf_params->sizeRho * X_wf_params->sizeZe * X_wf_params->sizeZh + 1];
			unsigned long long file_size;

			printf("  Exciton wavefunction *.bin file:\n\t'%s'\n", X_wf_filename);
			load(X_wf_filename, wf, &file_size, X_wf_params);
			normalize(wf, numThrowsNorm, X_wf_params);
			checkNormalizationMC(wf, numThrowsNorm / 10, X_wf_params); // check normalization using MC integration in (rho, phi, ze, zh) coordinates	

			// copy the normalized wave function to device memory
			gpuErrchk(cudaMalloc((void**)&gpu_wf, (X_wf_params->sizeRho * X_wf_params->sizeZe * X_wf_params->sizeZh + 1) * sizeof(double)));
			gpuErrchk(cudaMemcpy(gpu_wf, wf, (X_wf_params->sizeRho * X_wf_params->sizeZe * X_wf_params->sizeZh + 1) * sizeof(double), cudaMemcpyHostToDevice));
			delete[] wf;
			/* done loading X wf --------------------------------------------------------------------------------------------------------------------------------------      */

			int blockSize = 512;
			int numBlocks = (N + blockSize - 1) / blockSize;

			for (int num_q = 0; num_q < 19; num_q++) { // loop for q dependency

				curandState_t* states;
				gpuErrchk(cudaMalloc((void**)&states, N * sizeof(curandState_t))); // space for random states

				double cpu_f; // variable for sum of integrand function values inside a run on cpu
				double cpu_f2; // var for the sum of it's squares (to calculate error later) on cpu

				double intValue = 0.0; // vars to accumulate final values across all runs
				double intError = 0.0;

				double temp_res = 0; // vars for storing int. estimates in real-time 
				double temp_err = 0;

				double* gpu_f; // array for integrand function values at N random points on gpu
				double* gpu_f2; // array for it's squares (to calculate error later) on gpu
				gpuErrchk(cudaMalloc((void**)&gpu_f, numPoints * sizeof(double)));
				gpuErrchk(cudaMalloc((void**)&gpu_f2, numPoints * sizeof(double)));
				double* gpu_f_out;
				double* gpu_f2_out;
				gpuErrchk(cudaMalloc((void**)&gpu_f_out, numPoints * sizeof(double)));
				gpuErrchk(cudaMalloc((void**)&gpu_f2_out, numPoints * sizeof(double)));

				//printf("\n Rx = %e meV\n\n", Rx / e * 1e3);
				//printf("\n a0_hh = %e m", a0_hh);
				//printf("\n lambda_2d = %e m\n\n", lambda_2d);
				printf("--------------------------------------------------------------------------------------------\n");
				printf("  Calc parameters:\n\tnumPointsNorm = %.3e\n\tnumPoints = %.3e, numRun = %.1e, max total points = %.3e, \n\ttol = %.1e\tq = %3.1f * a0_hh = %.3e\n", (double)numThrowsNorm, (double)numPoints, (double)numRun, (double)numPoints * numRun, tol, q[num_q] * a0_hh, q[num_q]);
				printf("--------------------------------------------------------------------------------------------\n");
				printf("  Calculation controls:\n");
				printf("  \t'p' -- pause\n");
				printf("  \t'n' -- skip to next filename\n");
				printf("  \t'b' -- break\n");

				printf("       ______________________________________________________________________\n");
				printf("       J_X-X   &   error, mueV*mum^2  |  total points  |   elapsed time\n");

				char filename[] = "integral__.dat";
				FILE* F = fopen(filename, "a");
				if (F == NULL)
					printf("Failed opening file \"%s\"! \n", filename);
				fprintf(F, "\n============================================================================================\n");
				fprintf(F, "'%s'\n", X_wf_params_filename);
				fprintf(F, "  Calc parameters:\n\tnumPointsNorm = %.3e\n\tnumPoints = %.3e, numRun = %.1e, max total points = %.3e, \n\ttol = %.1e\tq = %3.1f * a0_hh = %.3e\n", (double)numThrowsNorm, (double)numPoints, (double)numRun, (double)numPoints * numRun, tol, q[num_q] * a0_hh, q[num_q]);
				//fprintf(F, "--------------------------------------------------------------------------------------------\n");
				fprintf(F, "       ______________________________________________________________________\n");
				fprintf(F, "       J_X-X   &   error, mueV*mum^2  |  total points  |   elapsed time\n");
				fclose(F);

				tic = clock();
				seed = tic + time(0);
				initRand << <numBlocks, blockSize >> > (seed, 0, states); // invoke the GPU to initialize all of the random states
				gpuErrchk(cudaDeviceSynchronize());

				printf("\t              ");
				long long int runCounter;
				for (runCounter = 0; runCounter < numRun; runCounter++) {
					//        initRand << <numBlocks, blockSize >> > (time(0)+clock(), 0, states); // invoke the GPU to initialize all of the random states
					//        gpuErrchk(cudaDeviceSynchronize());

					// calculate exciton coulomb energy for testing:
					//intMC_J_xx_exch << <numBlocks, blockSize >> > (states, gpu_f, gpu_f2, gpu_wf, gpu_X_wf_params, X_wf_params->L, dim, q[num_q]); // accumulate func and func^2 evaluations in gpu_f and gpu_f2
					intMC_J_xx_dir << <numBlocks, blockSize >> > (states, gpu_f, gpu_f2, gpu_wf, gpu_X_wf_params, X_wf_params->L, dim, q[num_q]); // accumulate func and func^2 evaluations in gpu_f and gpu_f2
					gpuErrchk(cudaPeekAtLastError());
					gpuErrchk(cudaDeviceSynchronize());

					sumGPUDouble(gpu_f, gpu_f_out, numPoints);
					sumGPUDouble(gpu_f2, gpu_f2_out, numPoints);

					/* copy back */
					gpuErrchk(cudaMemcpy(&cpu_f, gpu_f_out, sizeof(double), cudaMemcpyDeviceToHost));
					gpuErrchk(cudaMemcpy(&cpu_f2, gpu_f2_out, sizeof(double), cudaMemcpyDeviceToHost));

					intValue += cpu_f;
					intError += cpu_f2;

					// real-time output
					if (runCounter % 100 == 99) { //  we lose speed if we printf on every run
						for (int bCount = 0; bCount < (150); bCount++) // erase old line
							printf("\b");

						temp_res = X_wf_params->V_MC * intValue / ((runCounter + 1) * numPoints);
						temp_err = 3 * X_wf_params->V_MC / sqrt((runCounter + 1) * numPoints) * sqrt(intError / ((runCounter + 1) * numPoints) - intValue * intValue / ((runCounter + 1) * numPoints) / ((runCounter + 1) * numPoints));
						printf("\t%13e\t%12e", temp_res, temp_err);
						printf("\t %9e", (double)(runCounter + 1) * numPoints);
						toc = clock();
						printf("\t  %7e s", double(toc - tic) / CLOCKS_PER_SEC);
						//printf("\tJ_ex = %13e pm %12e mueV ", X_wf_params->V_MC_Ex * intValue / ((runCounter + 1) * numPoints), 3 * X_wf_params->V_MC_Ex / sqrt((runCounter + 1) * numPoints) * sqrt(intError / ((runCounter + 1) * numPoints) - intValue * intValue / ((runCounter + 1) * numPoints) / ((runCounter + 1) * numPoints)));

						if (temp_err < tol) {
							printf("\n--------------------------------------------------------------------------------------------\n");
							printf("\n\tDesired tolerance reached: temp_err < %.4f\n\n", tol);
							printf("=============================================================================================================================\n\n\n");
							break; // skip to end of this calculation
						}

						// keyboard control
						if (_kbhit()) {
							char kb = _getch(); // consume the char from the buffer, otherwise _kbhit remains != 0

							if (kb == 'p') {
								char filename[] = "integral__.dat";
								FILE* F = fopen(filename, "a");
								if (F == NULL)
									printf("Failed opening file \"%s\"! \n", filename);

								fprintf(F, "       ______________________________________________________________________\n");
								fprintf(F, "       J_X-X   &   error, mueV*mum^2  |  total points  |   elapsed time\n");
								fprintf(F, "\t%13e\t %12e", temp_res, temp_err);
								fprintf(F, "\t %9e", (double)(runCounter + 1) * numPoints);
								fprintf(F, "\t  %7e s\n", double(toc - tic) / CLOCKS_PER_SEC);
								fprintf(F, "--------------------------------------------------------------------------------------------\n");
								fclose(F);

								printf("\n\n Program paused: intermediate results appended to file \"%s\".\n", filename);
								printf(" To continue, press any key.\n\n");

								_getch(); // wait for a second key press to continue calculation

								printf("       ______________________________________________________________________\n");
								printf("       J_X-X   &   error, mueV*mum^2  |  total points  |   elapsed time\n");
							}

							else if (kb == 'n') {
								printf("\n=============================================================================================================================\n\n");
								printf(" Skipping to next calculation...\n\n");
								printf("=============================================================================================================================\n\n\n");
								break;// skip to end of this calculation
							}

							else if (kb == 'b') {
								printf("\n=============================================================================================================================\n\n");
								printf(" Program stopped.\n\n");
								printf("=============================================================================================================================\n\n\n");
								exit(10);

							}
						}
					}
				}

				F = fopen(filename, "a");
				if (F == NULL)
					printf("Failed opening file \"%s\"! \n", filename);

				fprintf(F, "Final value:\n");
				//fprintf(F, "       ______________________________________________________________________\n");
				//fprintf(F, "       J_X-X   &   error, mueV*mum^2  |  total points  |   elapsed time\n");
				fprintf(F, "\t%13e\t %12e", temp_res, temp_err);
				fprintf(F, "\t %9e", (double)(runCounter + 1) * numPoints);
				fprintf(F, "\t  %7e s\n", double(toc - tic) / CLOCKS_PER_SEC);
				fprintf(F, "====================================================================================================\n");
				fclose(F);
				
				gpuErrchk(cudaFree(states));
				gpuErrchk(cudaFree(gpu_f));
				gpuErrchk(cudaFree(gpu_f2));
				gpuErrchk(cudaFree(gpu_f_out));
				gpuErrchk(cudaFree(gpu_f2_out));
			}			
			delete[] X_wf_params;
			gpuErrchk(cudaFree(gpu_wf));
			gpuErrchk(cudaFree(gpu_X_wf_params));
		}
	}
	printf("\n\n\n\t\tAll calculations processed.\n\n");

	fclose(filenameFile);
	delete[] buffer;
	delete[] X_wf_params_filename;
	delete[] X_wf_filename;

	return 0;
}