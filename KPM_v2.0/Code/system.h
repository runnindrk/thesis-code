#define dimGrid  128
#define dimBlock 128

// -------------------------------------------------------------------
// In this we put the system dimensions XYZ
// and number of moments

#define Dimension 3    // dimensions
#define Orbitals  1    // number of orbitals

#define X 1024
#define Y 1024
#define Z 256

#define M 2048 // number of moments

#define SystemSize      X*Y*Z
#define HamiltonianSize Orbitals*SystemSize

#define W 0.0
#define lowerE -6 - W/2
#define upperE  6 + W/2

#define Scale ((upperE) - (lowerE))/(2.0 - 0.01)
#define Shift ((upperE) + (lowerE))/(2.0)

// -------------------------------------------------------------------