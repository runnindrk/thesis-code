#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <curand_kernel.h>
#include "./book.h"
#include "./system.h"
#include "./complex.cu"
#include "./random.cu"
#include "./functions.cu"
#include "./kpm_functions.cu"
#include "./kpm_algorithm.cu"

// -----------------------------------------------------------------------
// define here datatype to use
// thread and block size
// definitions to clear code

typedef float myType;

#define LOAD_CPU(arg1, arg2, arg3) arg1 *arg3 = (arg1 *)malloc(arg2*sizeof(arg1));
#define LOAD_GPU(arg1, arg2, arg3) HANDLE_ERROR( cudaMalloc((void**)&arg3, (arg2)*sizeof(arg1)) );

// arg1 GPU; arg2 CPU; arg3 size, arg4 type
#define TRANSFER_GPU_CPU(arg1, arg2, arg3, arg4) HANDLE_ERROR( cudaMemcpy(arg2, arg1, arg3*sizeof(arg4), cudaMemcpyDeviceToHost) );


// -----------------------------------------------------------------------
// -----------------------------------------------------------------------
// Main

int main(int argc, char **argv)
{
	// -------------------------------------------------------------------
	// Loading necessary arrays

	curandStateMRG32k3a *devMRGStates;

	LOAD_CPU(myType, M, m);
	LOAD_CPU(myType, HamiltonianSize, test);

	LOAD_CPU(myType, HamiltonianSize, a_dev);
	LOAD_CPU(myType, HamiltonianSize, b_dev);
	LOAD_CPU(double, HamiltonianSize, D_dev);
	LOAD_CPU(myType, M, m_dev);
	LOAD_CPU(myType, dimBlock, m1_red);
	LOAD_CPU(myType, dimBlock, m2_red);

	LOAD_GPU(curandStateMRG32k3a, dimBlock*dimGrid, devMRGStates);
	LOAD_GPU(myType, HamiltonianSize, a_dev);
	LOAD_GPU(myType, HamiltonianSize, b_dev);
	LOAD_GPU(double, HamiltonianSize, D_dev);
	LOAD_GPU(myType, M, m_dev);
	LOAD_GPU(myType, dimBlock, m1_red);
	LOAD_GPU(myType, dimBlock, m2_red);

	// -------------------------------------------------------------------
	
	getRandomVector<<<dimGrid, dimBlock>>>(devMRGStates, 2, HamiltonianSize, a_dev);
	getAndersonDisorder<<<dimGrid, dimBlock>>>(devMRGStates, 1, HamiltonianSize, W, D_dev);

	getMoments(D_dev, a_dev, b_dev, m_dev, m1_red, m2_red);

	TRANSFER_GPU_CPU(m_dev, m, M, myType);

	//for (int k = 0; k < M; k++)
	//	printf("%.8f %.8f\n", m[k].real, m[k].imag);

	for (int k = 0; k < M; k++)
		printf("%.8f %.8f\n", m[k], m[k]);
	
	//printf("%.8f %.8f\n", conj(a).real, conj(a).imag);
	//printf("%.8f\n", Scale);
	//printf("RAN WITHOUT ERRORS\n");
}