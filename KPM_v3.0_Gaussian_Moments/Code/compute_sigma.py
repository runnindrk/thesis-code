import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from numba import njit

# --------------------------------------------------------------------------
# Arguments 

X = 64
Y = 64
Z = 64

# --------------------------------------------------------------------------
# Acess moments

@njit
def chebyshev_polynomials(x, M):

  return np.cos(M*np.arccos(x))

@njit
def kernal(n, N):

  a = (N - n + 1)*np.cos(n*np.pi/(N + 1))
  b = np.sin(n*np.pi/(N + 1))/np.tan(np.pi/(N + 1))
  k = 1/(N+1)*(a + b)

  return k

@njit
def dos(all_moments, N): # moments > 1

  Ep = 10000

  dos = np.zeros(Ep)
  energy = np.linspace(-1 + 0.001, 1 - 0.001, Ep)

  sqrt = 1/(np.pi*np.sqrt(1 - energy**2))
  dos += chebyshev_polynomials(energy, 0)*kernal(0, N)*all_moments[0]

  for i in range (N - 1):
    dos += 2*chebyshev_polynomials(energy, i + 1)*kernal(i + 1, N)*all_moments[i + 1]

  return energy, sqrt*dos

@njit
def necessary_dos(all_moments, Et, sigma, num_points): # moments > 1

  N = 512
  energy = np.arange(max(Et - 2*sigma, - 1 + 1e-5), min(Et + 2*sigma, 1 - 1e-5), sigma/num_points)
  dos = np.zeros(len(energy))

  sqrt = 1/(np.pi*np.sqrt(1 - energy**2))
  dos += chebyshev_polynomials(energy, 0)*kernal(0, N)*all_moments[0]

  for i in range (N - 1):
    dos += 2*chebyshev_polynomials(energy, i + 1)*kernal(i + 1, N)*all_moments[i + 1]

  return energy, sqrt*dos

# --------------------------------------------------------------------------

@njit
def compute_DOS_integral(all_moments, Et, sigma, num_points):
   
   E, dos = necessary_dos(all_moments, Et, sigma, num_points)

   return np.trapz(dos, E)

# --------------------------------------------------------------------------

def bissection_method(all_moments, Ms, Mf, Nk, Et):

  # initialize
  limit = 20
  i     = -1

  Ma  = Ms
  N_a = X*Y*Z*compute_DOS_integral(all_moments, Et, Ma, 10000)

  Mb  = Mf
  N_b = X*Y*Z*compute_DOS_integral(all_moments, Et, Mb, 10000)

  if N_b - Nk > 0:

    raise Exception("Chose a higher M for the Max DOS")

  while True:

    i += 1

    # midpoint
    print(Ma, N_a, Mb, N_b)

    Mc  = (Ma + Mb)/2
    N_c = X*Y*Z*compute_DOS_integral(all_moments, Et, Mc, 10000)
    
    if i == limit:

      return Mc, N_c

    if abs(N_c - Nk) < 1:

      return Mc, N_c

    else:

      if ((N_a - Nk)*(N_c - Nk) <= 0):

        Mb        = Mc
        #Eb, Dos_b = Ec, Dos_c
        N_b       = N_c

      elif ((N_b - Nk)*(N_c - Nk) <= 0):

        Ma        = Mc
        #Ea, Dos_a = Ec, Dos_c
        N_a       = N_c

      else:

        raise Exception("None of the Intervals Qualifies")

# --------------------------------------------------------------------------

moments = np.loadtxt("../Data/moments_DOS.dat")[:, 0]

M, N = bissection_method(moments, 1, 0.00001, 300, 0)
print(M, N)
