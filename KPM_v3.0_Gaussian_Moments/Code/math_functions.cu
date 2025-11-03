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
// Simple Math

__host__ __device__ int mod(int a, int b)
{
    int r = a%b;
    return r < 0 ? r + b : r;
}

template<typename T>
__device__ T cuExp(const T& a)
{
    T b(0, 0);

    b.real = expf(a.real)*cosf(a.imag);
    b.imag = expf(a.real)*sinf(a.imag);

    return b;
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
// Chebyshev Math

template<typename T>
__host__ __device__ T chebyshev_factor(int n, T x)
{	
    return cosf(n*acosf(x));
}

// -----------------------------------------------------------------------
// Vector Math

template<typename T, typename U>
__global__ void addVectorFactor(int N, U flag, U factor, T *v, T *l)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < N) // generalize here for dimensions
    {
        l[tid] = l[tid]*(1.0 - flag) + v[tid]*factor;
        
        tid += blockDim.x*gridDim.x; 
    }
}

template<typename T>
__global__ void hadamardProduct(int N, T *r, T *l, T *s)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
    while (tid < N)
    {
        s[tid] = r[tid]*l[tid]; // normalize LDOS back to normal here
            
        tid += blockDim.x*gridDim.x; 
    }
}

// -----------------------------------------------------------------------
// Statistics

template<typename T>
__global__ void addAverage(int N, int num_av, T *r, T *l)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < N) // generalize here for dimensions
    {
		l[tid] = l[tid] + (r[tid] - l[tid])/double((num_av + 1));
			
        tid += blockDim.x*gridDim.x; 
    }
}

template<typename T>
__global__ void addVariance(int N, int num_av, T *r, T *r_mean_0, T *r_mean_1, T *l)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    while (tid < N) // generalize here for dimensions
    {
		l[tid] = l[tid] + ((conj(r[tid] - r_mean_0[tid]))*(r[tid] - r_mean_1[tid]) - l[tid])/double((num_av + 1));
			
        tid += blockDim.x*gridDim.x; 
    }
}

// -----------------------------------------------------------------------
// Correction to Coeficients (Gaussian)

__host__ __device__ double f(int n, double x, double mu, double sigma)
{
    return cosf(n*acosf(x))*expf(-0.5*(x - mu)*(x - mu)/(sigma*sigma))/(sqrtf(2*PI)*sigma)*sqrtf(1 - x*x);
}

__host__ __device__ double N(int n, double x, double mu, double sigma)
{
    return expf(-0.5*(x - mu)*(x - mu)/(sigma*sigma))/(sqrtf(2*PI)*sigma)*sqrtf(1 - x*x);
}

double chebyshevIntegration(int n, int num_int_points, double mu, double sigma)
{
    double m = 0;
    double normalization = 0;

    for (int i = 1; i < num_int_points + 1; i++)
    {
        double xi = cosf(double(2*i - 1)/double(2*num_int_points)*PI);
        m += PI/double(num_int_points)*f(n, xi, mu, sigma);
        normalization += PI/double(num_int_points)*N(n, xi, mu, sigma);
    }

    return m/normalization;
}

void computeGaussianMoments(int num_int_points, double mu, double sigma, double *m)
{
    for (int i = 0; i < M; i++)
        m[i] = chebyshevIntegration(i, num_int_points, mu, sigma);
}



