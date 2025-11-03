#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <curand_kernel.h>
#include "./book.h"
#include "./system.h"

// -----------------------------------------------------------------------
// Complex Random Functions

template<typename T>
__global__ void getRandomVectorComplex(curandStateMRG32k3a *state, int seed, int N, T *a)
{	
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	double phase = 0;
    T rphase(phase, 0);
    T I(0, 1);
	
	curand_init(seed, tid, 0, &state[tid%(dimBlock*dimGrid)]);
	curandStateMRG32k3a localState = state[tid%(dimBlock*dimGrid)];
		
	while (tid < N)
	{	
		phase = curand_uniform(&localState)*2*PI;
		rphase.real = phase;

        a[tid]   = cuExp(I*rphase); 
        
        a[tid].real /= sqrtf(N); 
        a[tid].imag /= sqrtf(N);
		
		tid += blockDim.x*gridDim.x; 
	}
}

// -----------------------------------------------------------------------
// Real Random Functions

//template<typename T>
__global__ void getRandomVector(curandStateMRG32k3a *state, int seed, int N, double *a)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	curand_init(seed, tid, 0, &state[tid%(dimBlock*dimGrid)]);
	curandStateMRG32k3a localState = state[tid%(dimBlock*dimGrid)];
		
	while (tid < N)
	{	
		a[tid] = curand_normal_double(&localState)/sqrtf(N);
		tid += blockDim.x*gridDim.x; 
	}
}

//template<typename T>
__global__ void getRandomVectorStoLDOS(curandStateMRG32k3a *state, int seed, int N, double *a)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	double mu    = 1.0/(sqrtf(sqrtf(2.0)));
	double sigma = sqrtf(1.0 - 1.0/sqrtf(2.0));

	curand_init(seed, tid, 0, &state[tid%(dimBlock*dimGrid)]);
	curandStateMRG32k3a localState = state[tid%(dimBlock*dimGrid)];
	
	while (tid < N)
	{	
		double uni = curand_uniform(&localState);

		a[tid] = (uni < 0.5)?(curand_normal_double(&localState)*sigma + mu):(curand_normal_double(&localState)*sigma - mu);
		tid += blockDim.x*gridDim.x; 
	}
}

// -----------------------------------------------------------------------
// Disorder Models

template<typename T, typename V>
__global__ void getAndersonDisorder(curandStateMRG32k3a *state, int seed, int N, V strW, T *a)
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	curand_init(seed, tid, 0, &state[tid%(dimBlock*dimGrid)]);
	curandStateMRG32k3a localState = state[tid%(dimBlock*dimGrid)];
		
	while (tid < N)
	{	
		a[tid] = curand_uniform_double(&localState)*strW - strW/2;
		tid += blockDim.x*gridDim.x; 
	}
}