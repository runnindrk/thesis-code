#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <curand_kernel.h>
#include "./book.h"
#include "./system.h"

// -----------------------------------------------------------------------
// KPM Functions

template<typename T, typename U>
__global__ void getTwoReductions(U flag, U *D, T *v, T *a, T *m1, T *m2)
{
    __shared__ T cache1r[dimBlock];
    __shared__ T cache2r[dimBlock];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    
    int z = 0; 
    int y = 0; 
    int x = 0; 

    // we must set it to a random value, after that we set it to zero so it works

    T temp1 = a[0]; T temp2 = a[0];
    T tempV = a[0]; T tempR = a[0];
    U Dis   = D[0]; T DElem = a[0];

    temp1 = temp1*0; temp2 = temp2*0;
    tempV = tempV*0; tempR = tempR*0;
    Dis   =   Dis*0; DElem = DElem*0;

    while (tid < SystemSize) // generalize here for dimensions
    {
    	// ---------------------------------------------------------------
    	// MatrixVectorMul
    	
        z = tid/(Y*X);
        y = (tid - z*X*Y)/X;
    	x = (tid - z*X*Y)%X; 
        
        tempV = v[tid];
        Dis   = D[tid];
        DElem = tempV*Dis;

        tempR = ( - v[tid - x + mod(x + 1, X)] - v[tid - x + mod(x - 1, X)]
                  - v[tid - X*(y - mod(y + 1, Y))] - v[tid - X*(y - mod(y - 1, Y))]
                  - v[tid - X*Y*(z - mod(z + 1, Z))] - v[tid - X*Y*(z - mod(z - 1, Z))]
                  + DElem - Shift)/(Scale)*(2.0/(1.0 + flag)) - a[tid]*(1.0 - flag);
        
        // ---------------------------------------------------------------
        
        temp1 = temp1 + conj(tempV)*tempV;
        temp2 = temp2 + conj(tempR)*tempV;
        
        a[tid] = tempR;
        
        tid += blockDim.x*gridDim.x;
    }

    cache1r[cacheIndex] = temp1;
    cache2r[cacheIndex] = temp2;

    __syncthreads();

    int i = blockDim.x/2;
    
    
    while (i != 0) 
    {
        if (cacheIndex < i)
        {
            cache1r[cacheIndex] = cache1r[cacheIndex] + cache1r[cacheIndex + i];
            cache2r[cacheIndex] = cache2r[cacheIndex] + cache2r[cacheIndex + i];
        }

        __syncthreads();
        i /= 2;
    }
    
    __syncthreads();

    if (cacheIndex == 0)
    {   
        m1[blockIdx.x] = cache1r[0];
        m2[blockIdx.x] = cache2r[0];
    }
}

// -----------------------------------------------------------------------