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
    
    addMomentToArray<<<1, 1>>>(0.0, 0, 0, dimBlock, m1_red, m);
    addMomentToArray<<<1, 1>>>(0.0, 1, 0, dimBlock, m2_red, m);
    
    for (int i = 0; i < M/2 - 1; ++i)
    {
		getTwoReductions<<<dimGrid, dimBlock>>>(0.0, D, (i%2 == 0)?b:a, (i%2 == 0)?a:b, m1_red, m2_red);
		
        addMomentToArray<<<1, 1>>>(1.0, 2*(i + 1), 0, dimBlock, m1_red, m);
        addMomentToArray<<<1, 1>>>(1.0, 2*(i + 1) + 1, 1, dimBlock, m2_red, m);
    }
}

// -----------------------------------------------------------------------
// Stochastic LDOS KPM Algorithm

template<typename T, typename U>
void stochasticLDOS(double E, U *D, T *a, T *b, T *l, double *m_gaussian)
{
    double factor = 0;
    setZero<<<dimGrid, dimBlock>>>(HamiltonianSize, b);
    setZero<<<dimGrid, dimBlock>>>(HamiltonianSize, l);
    
    addIteration<<<dimGrid, dimBlock>>>(1.0, D, a, b);

    factor = m_gaussian[0];
    //factor = chebyshev_factor(0, E);
    addVectorFactor<<<dimGrid, dimBlock>>>(HamiltonianSize, 0.0, factor, a, l);

    factor = 2*m_gaussian[1];
    //factor = 2*chebyshev_factor(1, E);
    addVectorFactor<<<dimGrid, dimBlock>>>(HamiltonianSize, 0.0, factor, b, l);
    
    for (int i = 0; i < M - 2; ++i)
    {
        addIteration<<<dimGrid, dimBlock>>>(0.0, D, (i%2 == 0)?b:a, (i%2 == 0)?a:b);
        
        factor = 2*m_gaussian[i + 2];
        //factor = 2*chebyshev_factor(i + 2, E);
        addVectorFactor<<<dimGrid, dimBlock>>>(HamiltonianSize, 0.0, factor, (i%2 == 0)?a:b, l);
    }

    factor = 1.0/((PI)*sqrtf(1 - E*E));
    addVectorFactor<<<dimGrid, dimBlock>>>(HamiltonianSize, 1.0, factor, l, l);
}