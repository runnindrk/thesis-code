#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <curand_kernel.h>
#include "./book.h"
#include "./system.h"

// -----------------------------------------------------------------------
// DOS/LDOS KPM Algorithm

template<typename T, typename U>
void getMoments(U *D, T *a, T *b, T *m, T *m1_red, T *m2_red)
{
    setZero<<<dimGrid, dimBlock>>>(HamiltonianSize, b);
    setZero<<<dimGrid, dimBlock>>>(M, m);
    setZero<<<dimGrid, dimBlock>>>(dimBlock, m1_red);
    setZero<<<dimGrid, dimBlock>>>(dimBlock, m2_red);
    
    getTwoReductions<<<dimGrid, dimBlock>>>(1.0, D, a, b, m1_red, m2_red);
    
    addToArray<<<1, 1>>>(0, dimBlock, m1_red, m);
    addToArray<<<1, 1>>>(1, dimBlock, m2_red, m);
    
    for (int i = 0; i < M/2 - 1; ++i)
    {
		getTwoReductions<<<128, 128>>>(0.0, D, (i%2 == 0)?b:a, (i%2 == 0)?a:b, m1_red, m2_red);
		
        addMomentToArray<<<1, 1>>>(2*(i + 1), 0, dimBlock, m1_red, m);
        addMomentToArray<<<1, 1>>>(2*(i + 1) + 1, 1, dimBlock, m2_red, m);
    }
    
}

// -----------------------------------------------------------------------
// Stochastic LDOS KPM Algorithm