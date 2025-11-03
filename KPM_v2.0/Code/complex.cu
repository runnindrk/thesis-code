#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <curand_kernel.h>
#include "./book.h"
#include "./system.h"

// -----------------------------------------------------------------------
// T Complex

template<typename T>
struct cuComplex
{
    T real;
    T imag;

    cuComplex() = default;

    __host__ __device__ cuComplex<T>(T a, T b ) : real(a), imag(b) {}

    // -------------------------------------------------------------------
    // Multiplication

    __host__ __device__ cuComplex<T> operator*(const T& a)
    {
        return cuComplex<T>(real*a, imag*a);
    }

    __host__ __device__ cuComplex<T> operator*(const cuComplex<T>& a)
    {
        return cuComplex<T>(real*a.real - imag*a.imag, imag*a.real + real*a.imag);
    }

    // -------------------------------------------------------------------
    // Division

    __host__ __device__ cuComplex<T> operator/(const T& a)
    {
        return cuComplex<T>(real/a, imag/a);
    }
    
    // -------------------------------------------------------------------
    // Addition

    __host__ __device__ cuComplex<T> operator+(const cuComplex<T>& a)
    {
        return cuComplex<T>(real + a.real, imag + a.imag);
    }

    __host__ __device__ cuComplex<T> operator+=(const cuComplex<T>& a)
    {
        return cuComplex<T>(real + a.real, imag + a.imag);
    }

    __host__ __device__ cuComplex<T> operator+(const T& a)
    {
        return cuComplex<T>(real + a, imag);
    }

    __host__ __device__ cuComplex<T> operator+()
    {
        return cuComplex<T>(real, imag);
    }

    // -------------------------------------------------------------------
    // Subtraction

    __host__ __device__ cuComplex<T> operator-(const cuComplex<T>& a)
    {
        return cuComplex<T>(real - a.real, imag - a.imag);
    }

    __host__ __device__ cuComplex<T> operator-=(const cuComplex<T>& a)
    {
        return cuComplex<T>(real - a.real, imag - a.imag);
    }

    __host__ __device__ cuComplex<T> operator-(const T& a)
    {
        return cuComplex<T>(real - a, imag);
    }

    __host__ __device__ cuComplex<T> operator-()
    {
        return cuComplex<T>(-real, -imag);
    }
    
    // -------------------------------------------------------------------
    // Conjungation

    __host__ __device__ cuComplex<T> conj(void)
    {
    	return cuComplex<T>(real, -imag);
    }

    // -------------------------------------------------------------------
    // Magnitude
    
    __host__ __device__ T magn(void)
    {
    	return real*real + imag*imag;
    }
};

