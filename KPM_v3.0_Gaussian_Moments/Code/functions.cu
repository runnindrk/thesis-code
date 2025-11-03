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

template<typename T>
__global__ void copyArray(int N, T *a, T *b)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
    
    while (tid < N)
    {
        b[tid] = a[tid]; 
        tid += blockDim.x*gridDim.x; 
    }
}

// -----------------------------------------------------------------------
// KPM Auxiliary Functions

template<typename T, typename U>
__global__ void addMomentToArray(U flag, int MomentIndex, int MomentIndexSubtract, int N, T *a, T *b)
{
    for (int i = 0; i < N; i++)
        b[MomentIndex] = b[MomentIndex] + a[i];

    b[MomentIndex] = b[MomentIndex]*(1.0 + flag) - b[MomentIndexSubtract]*flag;
}