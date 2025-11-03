#include <iostream>
#include <type_traits>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <curand_kernel.h>
#include "./book.h"
#include "./system.h"

// -----------------------------------------------------------------------

__host__ __device__ int mod(int a, int b)
{
    int r = a%b;
    return r < 0 ? r + b : r;
}

template<typename T>
__host__ __device__ inline T conj(T a)
{
    return a;
}

template<typename T>
__host__ __device__ inline cuComplex<T> conj(cuComplex<T> a)
{
    a.imag = - a.imag;
    return a;
}

// -----------------------------------------------------------------------

template<typename T>
__device__ T cuExp(const T& a)
{
    T b(0, 0);

    b.real = expf(a.real)*cosf(a.imag);
    b.imag = expf(a.real)*sinf(a.imag);

    return b;
}

template<typename T>
__global__ void setZero(int N, T *a)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    while (tid < N)
    {
        a[tid] = a[tid]*0; 
        tid += blockDim.x*gridDim.x; 
    }
}

// -----------------------------------------------------------------------
// KPM Auxiliary Functions

template<typename T>
__global__ void addToArray(int MomentIndex, int N, T *a, T *b)
{
    for (int i = 0; i < N; i++)
        b[MomentIndex] = b[MomentIndex] + a[i];
}

template<typename T>
__global__ void addMomentToArray(int MomentIndex, int MomentIndexSubtract, int N, T *a, T *b)
{
    for (int i = 0; i < N; i++)
        b[MomentIndex] = b[MomentIndex] + a[i];

    b[MomentIndex] = b[MomentIndex]*2.0 - b[MomentIndexSubtract];
}