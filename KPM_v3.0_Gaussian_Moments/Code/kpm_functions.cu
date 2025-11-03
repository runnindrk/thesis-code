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

    // -------------------------------------------------------------------
    // Dimension stuff

    #if (Dimension == 3)
        int z = 0;
        int y = 0;
        int x = 0;
    #endif

    #if (Dimension == 2)
        int y = 0;
        int x = 0;
    #endif

    #if (Dimension == 1)
        int x = 0;
    #endif
    
    // -------------------------------------------------------------------
    // we must set it to a random value, after that we set it to zero so it works

    T temp1 = a[0]*0; T temp2 = a[0]*0; 
    
    T tempV[Orbitals] = {a[0]*0.0}; T tempR[Orbitals] = {a[0]*0.0};
    U   Dis[Orbitals] = {D[0]*0.0}; T DElem[Orbitals] = {a[0]*0.0};

    cuComplex<double> I(0, 1);

    while (tid < SystemSize) // generalize here for dimensions
    {
    	// ---------------------------------------------------------------
    	// Initializes variables
    	
        #if (Dimension == 3)
            z = tid/(Y*X);
            y = (tid - z*X*Y)/X;
            x = (tid - z*X*Y)%X;
        #endif

        #if (Dimension == 2)
            y = tid/X;
            x = tid%X;
        #endif

        #if (Dimension == 1)
            x = tid;
        #endif
        
        for (int i = 0; i < Orbitals; i++)
        {
            tempV[i] = v[Orbitals*tid + i];
            Dis[i]   = D[Orbitals*tid + i];
            DElem[i] = tempV[i]*Dis[i];
        }
        
        // ---------------------------------------------------------------
        // This changes from model to model, it is in system.h

        Model
        
        // ---------------------------------------------------------------
        // Update

        for (int i = 0; i < Orbitals; i++)
        {
            temp1 = temp1 + conj(tempV[i])*tempV[i];
            temp2 = temp2 + conj(tempR[i])*tempV[i];
        }
        
        for (int i = 0; i < Orbitals; i++)
            a[Orbitals*tid + i] = tempR[i];
        
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
// Stochastic KPM Functions

template<typename T, typename U>
__global__ void addIteration(U flag, U *D, T *v, T *a)
{
    int tid = threadIdx.x + blockIdx.x*blockDim.x;

    // -------------------------------------------------------------------
    // Dimension stuff

    #if (Dimension == 3)
        int z = 0;
        int y = 0;
        int x = 0;
    #endif

    #if (Dimension == 2)
        int y = 0;
        int x = 0;
    #endif

    #if (Dimension == 1)
        int x = 0;
    #endif

    // -------------------------------------------------------------------
    // we must set it to a random value, after that we set it to zero so it works

    T tempV[Orbitals] = {a[0]*0.0}; T tempR[Orbitals] = {a[0]*0.0};
    U   Dis[Orbitals] = {D[0]*0.0}; T DElem[Orbitals] = {a[0]*0.0};

    cuComplex<double> I(0, 1);

    while (tid < SystemSize) // generalize here for dimensions
    {
    	// ---------------------------------------------------------------
    	// MatrixVectorMul
    	
        #if (Dimension == 3)
            z = tid/(Y*X);
            y = (tid - z*X*Y)/X;
            x = (tid - z*X*Y)%X;
        #endif

        #if (Dimension == 2)
            y = tid/X;
            x = tid%X;
        #endif

        #if (Dimension == 1)
            x = tid;
        #endif
        
        for (int i = 0; i < Orbitals; i++)
        {
            tempV[i] = v[Orbitals*tid + i];
            Dis[i]   = D[Orbitals*tid + i];
            DElem[i] = tempV[i]*Dis[i];
        }

        // ---------------------------------------------------------------
        // This changes from model to model

        Model
        
        // ---------------------------------------------------------------
        // Update

        for (int i = 0; i < Orbitals; i++)
            a[Orbitals*tid + i] = tempR[i];
        
        tid += blockDim.x*gridDim.x;
    }
}