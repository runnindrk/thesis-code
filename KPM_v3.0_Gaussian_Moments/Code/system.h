#define dimGrid  128
#define dimBlock 128

#define PI acosf(-1)

// -------------------------------------------------------------------
// Definitions to avoid hard reading

#define LOAD_CPU(arg1, arg2, arg3) arg1 *arg3 = (arg1 *)malloc(arg2*sizeof(arg1));
#define LOAD_GPU(arg1, arg2, arg3) HANDLE_ERROR( cudaMalloc((void**)&arg3, (arg2)*sizeof(arg1)) );

// arg1 GPU; arg2 CPU; arg3 size, arg4 type
#define TRANSFER_GPU_CPU(arg1, arg2, arg3, arg4) HANDLE_ERROR( cudaMemcpy(arg2, arg1, arg3*sizeof(arg4), cudaMemcpyDeviceToHost) );

// definitions for easier sparse-matrix reading, arg1 is axis and arg2 is orbital
#define Xneigh1(arg1, arg2) (Orbitals*(tid - x + mod(x + arg1, X)) + arg2)
#define Yneigh1(arg1, arg2) (Orbitals*(tid - X*(y - mod(y + arg1, Y))) + arg2)
#define Zneigh1(arg1, arg2) (Orbitals*(tid - X*Y*(z - mod(z + arg1, Z))) + arg2)

// 2nd neighbours currently defined only in 2-dimensions
#define XYneigh1(arg1, arg2, arg3) (Orbitals*(tid - x + mod(x + arg1, X) - X*(y - mod(y + arg2, Y))) + arg3)

// -------------------------------------------------------------------
// -------------------------------------------------------------------
// In this we put the system dimensions XYZ
// and number of moments

#define Dimension 3   // dimensions
#define Orbitals  1   // number of orbitals

#define X 64
#define Y 64
#define Z 64

#define M 32768 // number of moments

#define SystemSize      X*Y*Z
#define HamiltonianSize Orbitals*SystemSize

#define W 18.0
#define lowerE -6 - W/2
#define upperE  6 + W/2

#define Scale ((upperE) - (lowerE))/(2.0 - 0.01)
#define Shift ((upperE) + (lowerE))/(2.0)

#define Model CubicModel 

// -------------------------------------------------------------------
// Models Predefined
// Orbitals = 1; Dimension = 3

#define CubicModel tempR[0] = ( - v[Xneigh1(1, 0)] - v[Xneigh1(-1, 0)] \
                                - v[Yneigh1(1, 0)] - v[Yneigh1(-1, 0)] \
                                - v[Zneigh1(1, 0)] - v[Zneigh1(-1, 0)] \
                                + DElem[0] - Shift)/(Scale)*(2.0/(1.0 + flag)) \
                                - a[tid]*(1.0 - flag); \

// -------------------------------------------------------------------
// Orbitals = 2; Dimension = 3

#define WeylModel tempR[0] =  (I*(v[Xneigh1(1, 1)] - v[Xneigh1(-1, 1)] \
        			      - I*(v[Yneigh1(1, 1)] - v[Yneigh1(-1, 1)]) \
        			      +    v[Zneigh1(1, 0)] - v[Zneigh1(-1, 0)])*1.0/2.0 \
                             + DElem[0] - Shift)/(Scale)*(2.0/(1.0 + flag)) \
                             - a[2*tid]*(1.0 - flag); \
                       \
                  tempR[1] =  (I*(v[Xneigh1(1, 0)] - v[Xneigh1(-1, 0)] \
                             + I*(v[Yneigh1(1, 0)] - v[Yneigh1(-1, 0)]) \
                             -    v[Zneigh1(1, 1)] + v[Zneigh1(-1, 1)])*1.0/2.0 \
                             + DElem[1] - Shift)/(Scale)*(2.0/(1.0 + flag)) \
                             - a[2*tid + 1]*(1.0 - flag); \

// -------------------------------------------------------------------
// Orbitals = 1; Dimension = 2

#define SquareModel tempR[0] = ( - v[Xneigh1(1, 0)] - v[Xneigh1(-1, 0)] \
                                 - v[Yneigh1(1, 0)] - v[Yneigh1(-1, 0)] \
                                 + DElem[0] - Shift)/(Scale)*(2.0/(1.0 + flag)) \
                                 - a[tid]*(1.0 - flag); \

// -------------------------------------------------------------------
// Orbitals = 2; Dimension = 2

#define NicoModel tempR[0] = ( + v[Xneigh1(1, 0)] + v[Xneigh1(-1, 0)] \
                               - v[Yneigh1(1, 0)] - v[Yneigh1(-1, 0)] \
                               + (v[XYneigh1(1, 1, 1)] + v[XYneigh1(-1, -1, 1)])*0.5 \
                               - (v[XYneigh1(1, -1, 1)] + v[XYneigh1(-1, 1, 1)])*0.5 \
                               + DElem[0] - Shift)/(Scale)*(2.0/(1.0 + flag)) \
                               - a[2*tid]*(1.0 - flag); \
                     \
                  tempR[1] = ( - v[Xneigh1(1, 1)] - v[Xneigh1(-1, 1)] \
                               + v[Yneigh1(1, 1)] + v[Yneigh1(-1, 1)] \
                               + (v[XYneigh1(1, 1, 0)] + v[XYneigh1(-1, -1, 0)])*0.5 \
                               - (v[XYneigh1(1, -1, 0)] + v[XYneigh1(-1, 1, 0)])*0.5 \
                               + DElem[1] - Shift)/(Scale)*(2.0/(1.0 + flag)) \
                               - a[2*tid + 1]*(1.0 - flag); \

// -------------------------------------------------------------------
// Orbitals = 2; Dimension = 2

#define GrapheneModel tempR[0] = ( - v[Xneigh1(-1, 1)] - v[Yneigh1(-1, 1)] \
                                   - v[Yneigh1(0, 1)] \
                                   + DElem[0] - Shift)/(Scale)*(2.0/(1.0 + flag)) \
                                   - a[2*tid]*(1.0 - flag); \
                        \
                      tempR[1] = ( - v[Xneigh1(0, 0)] - v[Xneigh1(1, 0)] \
                                   - v[Yneigh1(1, 0)] \
                                   + DElem[1] - Shift)/(Scale)*(2.0/(1.0 + flag)) \
                                   - a[2*tid + 1]*(1.0 - flag); \

// -------------------------------------------------------------------
// Orbitals = 1; Dimension = 1

#define LinearModel tempR[0] = ( - v[Xneigh1(1, 0)] - v[Xneigh1(-1, 0)] \
                                 + DElem[0] - Shift)/(Scale)*(2.0/(1.0 + flag)) \
                                 - a[tid]*(1.0 - flag); \

// -------------------------------------------------------------------