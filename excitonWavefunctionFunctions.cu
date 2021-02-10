#include "excitonWavefunctionFunctions.h"

//  random num (0; 1)
double randf() {
	double ans = double(rand()) / double(RAND_MAX);
	return ans;
}

/* ---------------------------------- functions for loading parameters and wavefunctions and such -------------------------------------------------- */

// reads dZ, sizeRho, sizeZe and sizeZh from file F and into the specified structure
// also calculates S_real, V_MC and fix
int loadExcitonWfParams(char* filename, wf_parameter_struct* X_wf_params) {

	double WQW;
	double dZ;
	int sizeRho;
	int sizeZe;
	int sizeZh;

	if (get_val_from_line(filename, "double", 3, NULL, &WQW) == 0)
		return 0;
	WQW = WQW * 1e-9; // nm -> m
	get_val_from_line(filename, "double", 4, NULL, &dZ);
	dZ = dZ * 1e-9; 
	get_val_from_line(filename, "int", 5, &sizeRho, NULL);
	get_val_from_line(filename, "int", 6, &sizeZe, NULL);
	get_val_from_line(filename, "int", 7, &sizeZh, NULL);

	X_wf_params->L = WQW;
	X_wf_params->dZ = dZ;
	X_wf_params->sizeRho = sizeRho;
	X_wf_params->sizeZe = sizeZe;
	X_wf_params->sizeZh = sizeZh;
	X_wf_params->S_real = pi * pow(2 * (sizeRho + 1) * dZ, 2);
	//	X_wf_params->V_MC = (2 * pi * (X_wf_params->sizeRho - 1) * (X_wf_params->sizeZe - 1) * (X_wf_params->sizeZh - 1)) * (2 * pi * (X_wf_params->sizeRho - 1) * (X_wf_params->sizeZe - 1)) * pow(X_wf_params->dZ, 5); // MC volume of integrated area 
	X_wf_params->V_MC = pow(2 * pi * (sizeRho + 1) * (sizeZe + 1) * (sizeZh + 1), 2)
		* (2 * pi * 2 * (sizeRho + 1)) * pow(dZ, 7); // MC volume of integrated area 
	X_wf_params->V_MC_Ex = (2 * pi * X_wf_params->sizeRho * (X_wf_params->sizeZe - 1) * (X_wf_params->sizeZh - 1)) * pow(X_wf_params->dZ, 3);
	X_wf_params->fix = 1e-17;

	printf("\t   Exciton wave function parameters loaded:\n");
	printf("\t   * L = %.0f nm\n", X_wf_params->L * 1e9);
	printf("\t   * dZ = %.4f nm\n", X_wf_params->dZ * 1e9);
	printf("\t   * sizeRho = %d\n", X_wf_params->sizeRho);
	printf("\t   * sizeZe  = %d\n", X_wf_params->sizeZe);
	printf("\t   * sizeZh  = %d\n", X_wf_params->sizeZh);
	printf("\t   * S_real = %e mum^2\n", X_wf_params->S_real);
	printf("\t   * V_MC = (2pi * Rho * Z * Z)^2 * (2pi * Xi) = %e, [V_MC] = L^7\n", X_wf_params->V_MC);
	
	return 1;
}

// loads wavefunc with given parameters from a .bin file
void load(char wavefunction_filenames[], double* wf, unsigned long long* file_size, wf_parameter_struct* X_wf_params) {
	//https://stackoverflow.com/questions/22059189/read-a-file-as-byte-array
	FILE* fileptr;
	//double *wf;
	unsigned long long filelen;

	FILE* F;
	fopen_s(&F, "wavefunc_text.txt", "w");

	fopen_s(&fileptr, wavefunction_filenames, "rb");  // Open the file in binary mode
	if (fileptr != NULL) {
		fseek(fileptr, 0, SEEK_END);          // Jump to the end of the file
		filelen = ftell(fileptr);             // Get the current byte offset in the file
		rewind(fileptr);                      // Jump back to the beginning of the file
		unsigned long long numbers_count = filelen / sizeof(double);
		//wf = (double *)malloc((filelen + 1)); // Enough memory for file + \0
		//*sizeof(double)
		unsigned long long read_size = 0;
		unsigned long long read_size_add;
		//double read_value;
		char bytes[8];
		//		printf("filelen in mb = %f\n", (float)filelen / 1024.0 / 1024.0);
//				printf("numbers_count = %d\n", numbers_count);
	//			printf("sizes product = %d\n", sizeRho * sizeZe * sizeZh);
		unsigned long long i;
		for (i = 0; i < (numbers_count); i++) {
			read_size_add = fread(&bytes, 8, 1, fileptr);
			//			if (read_size_add != 1) { printf("Reading error!\n"); fputs("Reading error!\n", stderr); exit(1); }
						//read_size_add = fread(&read_value, 1, 1, fileptr); // Read in the entire file
			if (i > X_wf_params->sizeRho * X_wf_params->sizeZe * X_wf_params->sizeZh) {
				printf("\n\t   Failed to load wave function! Not enough memory allocated for wf[].\n  Possible cause: wavefunction dimensions are larger than specified.\n");
				fclose(fileptr); // Close the file
				fclose(F);
				exit(1);
			}
			else {
				wf[i] = *((double*)bytes);
				fprintf_s(F, "%e ", wf[i]);
				read_size = read_size + read_size_add;
			}
			//if  (1)	printf("numbers_count = %d, i = %d\n", numbers_count, i);
		}
		//printf("read_size = %d\n", read_size);
		*(file_size) = read_size * sizeof(double);
		fclose(fileptr); // Close the file
		fclose(F);
		if (i == X_wf_params->sizeRho * X_wf_params->sizeZe * X_wf_params->sizeZh)
			printf("\t   Exciton wave function loaded successfully!\n");
		else {
			printf("\n\t   Failed to load wave function! Wavefunction dimensions are larger than specified:\n\tnumbers_count = %d\n\ti = %d\n\tsizeRho * sizeZe * sizeZh = %d\n", numbers_count, i, X_wf_params->sizeRho * X_wf_params->sizeZe * X_wf_params->sizeZh);
			exit(2);
		}
	}
	else {
		printf("\n\t   Failed to open file '%s'!\n", wavefunction_filenames);
	}
}

// ensures the wavefunction is properly normalized before integration
void normalize(double* wf, int numThrowsNorm, wf_parameter_struct* X_wf_params) {
	double res = 0;
	double res2 = 0;

	double dZ = X_wf_params->dZ;
	int sizeRho = X_wf_params->sizeRho;
	int sizeZe = X_wf_params->sizeZe;
	int sizeZh = X_wf_params->sizeZh;
	double S_real = X_wf_params->S_real;

	// initially, wf = \psi * \rho, so we fix that	
	for (int iRho = 0; iRho < sizeRho; iRho++)
		for (int iZe = 0; iZe < sizeZe; iZe++)
			for (int iZh = 0; iZh < sizeZh; iZh++)
				wf[iRho + iZe * sizeRho + iZh * sizeZe * sizeRho] = wf[iRho + iZe * sizeRho + iZh * sizeZe * sizeRho] / (dZ * (iRho + 1));

	// now we integrate it to find the normalization constant
/*	for (int iRho = 0; iRho < sizeRho; iRho++)
		for (int iZe = 0; iZe < sizeZe; iZe++)
			for (int iZh = 0; iZh < sizeZh; iZh++)
				//				for (int iPhi = 0; iPhi < 100; iPhi++)
				res += pow(dZ, 3) * pow(wf[iRho + iZe * sizeRho + iZh * sizeZe * sizeRho], 2) * dZ * (iRho + 1);// * 1.0 / 100 * 2 * pi;
*/

	printf("\t   Calculating normalization constant using MC integration @%.1e points...      ", (float)numThrowsNorm);
	for (int i = 0; i < numThrowsNorm; i++) {
		if (i % 99 == 0)
			printf("\b\b\b\b%3.2f", (float)i / numThrowsNorm);
		double rho = randf() * (sizeRho + 1) + 1;
		int iRho = floor(0.5 + rho - 1);

		double ze = randf() * (sizeZe + 1) - 0.5;
		int iZe = floor(0.5 + ze);

		double zh = randf() * (sizeZe + 1) - 0.5;
		int iZh = floor(0.5 + zh);

		double tmp;

		int index = iRho + iZe * sizeRho + iZh * sizeZe * sizeRho;
		if (index < 0) {
			printf("\n Illegal memory access: index = %d, iRho = %d, iZe = %d, iZh = %d\n ", index, iRho, iZe, iZh);
			tmp = 0;
		}
		else if (iRho >= sizeRho || iZe >= sizeZe || iZh >= sizeZh) {
			tmp = 0;
		}
		else
			tmp = pow(wf[iRho + iZe * sizeRho + iZh * sizeZe * sizeRho], 2) * dZ * rho;
		res += tmp;
		res2 += pow(tmp, 2);
	}
	double V = 2 * pi * pow(dZ, 3) * (sizeZe + 1) * (sizeZh + 1) * (sizeRho + 1);

	res2 = 3 * V * sqrt((res2 / numThrowsNorm - res * res / numThrowsNorm / numThrowsNorm)) / sqrt(numThrowsNorm);
	res = res * V / numThrowsNorm;
	printf("\n\t\tNormalization integral = %e pm %e \t rel. error = %.3f \n", res, res2, res2 / res);

	// make normalization unity
	for (int iRho = 0; iRho < sizeRho; iRho++)
		for (int iZe = 0; iZe < sizeZe; iZe++)
			for (int iZh = 0; iZh < sizeZh; iZh++) {
				wf[iRho + iZe * sizeRho + iZh * sizeZe * sizeRho] = wf[iRho + iZe * sizeRho + iZh * sizeZe * sizeRho] / sqrt(res) / sqrt(S_real);
				//				printf("	wf[iRho + iZe * sizeRho + iZh * sizeZe * sizeRho] = %e\n", wf[iRho + iZe * sizeRho + iZh * sizeZe * sizeRho]);
			}
}

void checkNormalizationMC(double* wf, int numThrowsCheckNorm, wf_parameter_struct* X_wf_params) {

	double dZ = X_wf_params->dZ;
	int sizeRho = X_wf_params->sizeRho;
	int sizeZe = X_wf_params->sizeZe;
	int sizeZh = X_wf_params->sizeZh;
	double S_real = X_wf_params->S_real;

	printf("\t   Checking normalization using MC integration @%.1e points...      ", (float)numThrowsCheckNorm);
	srand(time(0));
	double res = 0;
	double res2 = 0;
	for (int i = 0; i < numThrowsCheckNorm; i++) {
		if (i % 99 == 0)
			printf("\b\b\b\b%3.2f", (float)i / numThrowsCheckNorm);
		double rho = randf() * (sizeRho + 1) + 1;
		int iRho = floor(0.5 + rho - 1);

		double ze = randf() * (sizeZe + 1) - 0.5;
		int iZe = floor(0.5 + ze);

		double zh = randf() * (sizeZe + 1) - 0.5;
		int iZh = floor(0.5 + zh);

		double tmp;

		int index = iRho + iZe * sizeRho + iZh * sizeZe * sizeRho;
		if (index < 0) {
			printf("\n Illegal memory access: index = %d, iRho = %d, iZe = %d, iZh = %d\n ", index, iRho, iZe, iZh);
			tmp = 0;
		}
		else if (iRho >= sizeRho || iZe >= sizeZe || iZh >= sizeZh) {
			tmp = 0;
		}
		else
			tmp = pow(wf[iRho + iZe * sizeRho + iZh * sizeZe * sizeRho], 2) * dZ * rho;
		res += tmp;
		res2 += pow(tmp, 2);
	}
	double V = 2 * pi * S_real * pow(dZ, 3) * (sizeZe + 1) * (sizeZh + 1) * (sizeRho + 1);
	printf("\n\t\tNormalization integral = %6.4f", res * V / numThrowsCheckNorm);
	printf(" pm %6.4f\n\n", sqrt((res2 / numThrowsCheckNorm - res * res / numThrowsCheckNorm / numThrowsCheckNorm)) * V / sqrt(numThrowsCheckNorm));
}