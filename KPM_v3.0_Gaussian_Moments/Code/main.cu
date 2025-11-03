#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <curand_kernel.h>
#include "./book.h"
#include "./system.h"
#include "./complex.cu"
#include "./math_functions.cu"
#include "./random.cu"
#include "./functions.cu"
#include "./kpm_functions.cu"
#include "./kpm_algorithm.cu"
#include "./fmt-master/include/fmt/core.h"

// -----------------------------------------------------------------------
// define here datatype to use

typedef double myType;

// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// Main

int main(int argc, char **argv)
{
	// -------------------------------------------------------------------
	// What we calculating?

	int DOS  = 0;
	int LDOS = 0;
	int StoLDOS = 1;

	int ran_vec = 128;
	int dis_rea = 1;   // do not increase it yet

	double E = 0.5;
	// -------------------------------------------------------------------
	// Loading necessary arrays

	curandStateMRG32k3a *devMRGStates;

	LOAD_CPU(myType, M, m);
	LOAD_CPU(myType, M, m_gaussian);
	LOAD_CPU(myType, HamiltonianSize, test);

	LOAD_CPU(myType, HamiltonianSize, a_dev);
	LOAD_CPU(myType, HamiltonianSize, b_dev);
	LOAD_CPU(double, HamiltonianSize, D_dev);

	LOAD_GPU(curandStateMRG32k3a, dimBlock*dimGrid, devMRGStates);
	LOAD_GPU(myType, HamiltonianSize, a_dev);
	LOAD_GPU(myType, HamiltonianSize, b_dev);
	LOAD_GPU(double, HamiltonianSize, D_dev);

	computeGaussianMoments(100000, E, 0.00001, m_gaussian);
	
	// -------------------------------------------------------------------
	// Normal calculations

	if (DOS == 1 or LDOS == 1)
	{
		// ---------------------------------------------------------------
		// Further Necessary Arrays for this case

		LOAD_CPU(myType, M, m_avg);
		LOAD_CPU(myType, M, m_dev);
		LOAD_CPU(myType, dimBlock, m1_red);
		LOAD_CPU(myType, dimBlock, m2_red);

		LOAD_GPU(myType, M, m_avg);
		LOAD_GPU(myType, M, m_dev);
		LOAD_GPU(myType, dimBlock, m1_red);
		LOAD_GPU(myType, dimBlock, m2_red);

		setZero<<<dimGrid, dimBlock>>>(M, m_avg);

		// ---------------------------------------------------------------
		// Calculation
		// Does not work for different disorders, yet, only because of the inner loop

		if (DOS == 1)
		{
			for (int dis_idx = 0; dis_idx/2 < dis_rea; dis_idx += 2)
			{
				for (int ran_idx = 1; (ran_idx - 1)/2 < ran_vec; ran_idx += 2)
				{
					int avg_idx  = (ran_idx  - 1)/2 + (dis_idx/2)*dis_rea;

					getAndersonDisorder<<<dimGrid, dimBlock>>>(devMRGStates, dis_idx, HamiltonianSize, W, D_dev);
					getRandomVector<<<dimGrid, dimBlock>>>(devMRGStates, ran_idx, HamiltonianSize, a_dev);
					getMoments(D_dev, a_dev, b_dev, m_dev, m1_red, m2_red);

					addAverage<<<dimGrid, dimBlock>>>(M, avg_idx, m_dev, m_avg);
				}
			}
			
			// -----------------------------------------------------------
			// Printing

			TRANSFER_GPU_CPU(m_avg, m, M, myType);
				
			for (int k = 0; k < M; k++)
				printf("%.8f %.8f\n", m[k], m[k]);
		}
	}
	
	// -------------------------------------------------------------------
	// LDOS Stochastic Approach
	// Only works for real valued functions fow now
	
	if (StoLDOS == 1)
	{
		// ---------------------------------------------------------------
		// Further Necessary Arrays for this case
		
		LOAD_CPU(myType, HamiltonianSize, test);
		LOAD_CPU(myType, HamiltonianSize, rand_dev);
		LOAD_CPU(myType, HamiltonianSize, LDOS_dev);
		LOAD_CPU(myType, HamiltonianSize, LDOS_AVG_dev);
		LOAD_CPU(myType, HamiltonianSize, LDOS_VAR_dev);

		LOAD_GPU(myType, HamiltonianSize, rand_dev);
		LOAD_GPU(myType, HamiltonianSize, LDOS_dev);
		LOAD_GPU(myType, HamiltonianSize, LDOS_AVG_dev);
		LOAD_GPU(myType, HamiltonianSize, LDOS_VAR_dev);

		// ---------------------------------------------------------------
		// Calculation

		for (int dis_idx = 0; dis_idx/2 < dis_rea; dis_idx += 2)
		{
			for (int ran_idx = 1; (ran_idx - 1)/2 < ran_vec; ran_idx += 2)
			{
				int avg_idx  = (ran_idx  - 1)/2;

				getAndersonDisorder<<<dimGrid, dimBlock>>>(devMRGStates, dis_idx, HamiltonianSize, W, D_dev);
				getRandomVectorStoLDOS<<<dimGrid, dimBlock>>>(devMRGStates, ran_idx, HamiltonianSize, a_dev);
				copyArray<<<dimGrid, dimBlock>>>(HamiltonianSize, a_dev, rand_dev);

				stochasticLDOS(E, D_dev, a_dev, b_dev, LDOS_dev, m_gaussian);
				hadamardProduct<<<dimGrid, dimBlock>>>(HamiltonianSize, LDOS_dev, rand_dev, LDOS_dev);
				addVectorFactor<<<dimGrid, dimBlock>>>(HamiltonianSize, 1.0, 1.0/(Scale)*1.0/(HamiltonianSize), LDOS_dev, LDOS_dev);

				// -----------------------------------------------------------
				// Statistics

				// copy old average
				copyArray<<<dimGrid, dimBlock>>>(HamiltonianSize, LDOS_AVG_dev, rand_dev);

				// new average and variance
				addAverage<<<dimGrid, dimBlock>>>(HamiltonianSize, avg_idx, LDOS_dev, LDOS_AVG_dev);
				addVariance<<<dimGrid, dimBlock>>>(HamiltonianSize, avg_idx, LDOS_dev, rand_dev, LDOS_AVG_dev, LDOS_VAR_dev);
			}
		}

		// ---------------------------------------------------------------
		// Printing

		TRANSFER_GPU_CPU(LDOS_VAR_dev, test, HamiltonianSize, myType);

		for (int k = 0; k < HamiltonianSize; k++)
			printf("%.32f\n", test[k]);
	}
	
	//for (int k = 0; k < M; k++)
	//	printf("%.8f %.8f\n", m[k].real, m[k].imag);
}