import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
from numba import njit, prange

# ----------------------------------------

@njit
def get_random_vector_dos(size):
    return np.exp(-2*np.pi*np.random.random(size)*1j).astype(np.cdouble)/np.sqrt(size)

@njit
def get_random_vector_ldos(size):
    return np.exp(-2*np.pi*np.random.random(size)*1j).astype(np.cdouble)

@njit
def get_handcrafted_random_vector_ldos(size):
    
    mu    = 1/2**(1/4)
    sigma = np.sqrt(1 - 1/np.sqrt(2))
    x     = np.random.normal(0, 1, size = size)*sigma + mu
    
    pos = np.where(np.random.random_sample(size) < 0.5)[0]
    x[pos] -= 2*mu
    
    return x + 0j
    
#num_med_fixed = 4
#M_fixed = 2048
size = int(sys.argv[1])
r_init = get_random_vector_dos(size)
W_strenght = 2.5
W_vector = np.random.random(size).astype(np.cdouble)*W_strenght - W_strenght/2

# full real space hamiltonian 
H = - np.eye(size,k=1) - np.eye(size,k=-1) + np.diag(W_vector.real)
H[0, -1] = -1
H[-1, 0] = -1

E, v = np.linalg.eigh(H)

@njit
def exact_1D_dos(E):
    
    return 1/(np.pi*np.sqrt(4 - E**2))

# -----------------------------------------
# KPM

@njit
def hamiltonian_vector_product(vector, W_vector, W_strenght):
    
    size = len(vector)
    a = np.zeros(size).astype(np.cdouble)
    
    for i in range (size):
        a[i] = (- vector[(i - 1)%size] - vector[(i + 1)%size] + vector[i]*W_vector[i])/(2 + 0.1 + W_strenght/2)
    
    return a

@njit
def generate_moments(M, r_init, W_vector, W_strenght):
    
    size = len(r_init)
    m = np.zeros(M).astype(np.cdouble)
    
    a = r_init
    b = hamiltonian_vector_product(r_init, W_vector, W_strenght)
    c = r_init
    
    m[0] = np.dot(a.conj(), a)
    m[1] = np.dot(b.conj(), a)
    
    for i in range (M - 2):
        
        c = 2*hamiltonian_vector_product(b, W_vector, W_strenght) - a
        m[i + 2] = np.dot(c.conj(), r_init)
        
        a = b.copy()
        b = c.copy()
        
    return m

@njit
def chebyshev_polynomial(x, n):
    return np.cos(n*np.arccos(x))

@njit
def jackson_kernel(M, n):
    
    a = (M - n + 1)*np.cos(np.pi*n/(M + 1))
    b = np.sin(np.pi*n/(M + 1))/np.tan(np.pi/(M + 1))
    
    return (a + b)/(M + 1)

@njit
def resum_moments(energy, moments, W_strenght):
    
    E = energy/(2 + 0.1 + W_strenght/2)
    M = len(moments)
    DOS = moments[0]
    
    for i in range (M - 1):
        DOS += 2*moments[i + 1]*chebyshev_polynomial(E, i + 1)*jackson_kernel(M, i + 1)
    
    return DOS*1/(np.pi*np.sqrt(1 - E**2))/(2 + 0.1 + W_strenght/2)

@njit
def DOS(moments, W_strenght):
    
    E = np.linspace(-(2 + W_strenght/2), (2 + W_strenght/2), 1000)
    dos = np.zeros(len(E))
    
    for i in range (len(E)):
        dos[i] = resum_moments(E[i], moments, W_strenght)
        
    return dos

@njit
def average_DOS(size, W_strenght, W_vector, M, n_av):
    
    r_init = get_random_vector_dos(size)
    moments = generate_moments(M, r_init, W_vector, W_strenght).real
    DOS_av = np.zeros(len(DOS(moments, W_strenght)))
    
    for i in range (n_av):
    
        r_init = get_random_vector_dos(size)
        moments = generate_moments(M, r_init, W_vector, W_strenght).real
        DOS_av += (DOS(moments, W_strenght) - DOS_av)/(i + 1)
        
    return DOS_av
    
# -----------------------------------------
# DOS TEST

#E = np.linspace(-(2 + W_strenght/2), (2 + W_strenght/2), 1000)

#plt.plot(E, exact_1D_dos(E))
#plt.plot(E, average_DOS(size, W_strenght, W_vector, 128, 50))
#plt.show()

# -----------------------------------------
# EXACT LDOS KPM

@njit
def get_position_vector(size, location):
    
    a = np.zeros(size)
    a[location] = 1
    
    return a.astype(np.cdouble)

@njit
def LDOS_position(E, M, size, location, W_vector, W_strenght):
    
    r_init = get_position_vector(size, location)
    
    moments = generate_moments(M, r_init, W_vector, W_strenght)
    LDOS = resum_moments(E, moments, W_strenght)
    
    return LDOS

@njit
def LDOS(E, M, size, W_vector, W_strenght):
    
    ldos = np.zeros(size).astype(np.cdouble)
    
    for i in range (size):
        ldos[i] = LDOS_position(E, M, size, i, W_vector, W_strenght)
        
    return ldos
    
# -----------------------------------------
# STOCHASTIC LDOS KPM

@njit
def iterate_vector(energy, M, r_init, W_vector, W_strenght):
    
    E = energy/(2 + 0.1 + W_strenght/2)
    
    size = len(r_init)
    V = np.zeros(size).astype(np.cdouble)
    
    a = r_init
    b = hamiltonian_vector_product(r_init, W_vector, W_strenght)
    
    V += a*(1/(np.pi*np.sqrt(1 - E**2))) \
         *chebyshev_polynomial(E, 0)*jackson_kernel(M, 0)
    
    V += 2*b*(1/(np.pi*np.sqrt(1 - E**2))) \
          *chebyshev_polynomial(E, 1)*jackson_kernel(M, 1)
    
    for i in range (M - 2):
        
        c = 2*hamiltonian_vector_product(b, W_vector, W_strenght) - a
        V += 2*c*(1/(np.pi*np.sqrt(1 - E**2))) \
              *chebyshev_polynomial(E, i + 2)*jackson_kernel(M, i + 2)
        
        a = np.copy(b)
        b = np.copy(c)
        
    return r_init.conj()*V/(2 + 0.1 + W_strenght/2)

@njit
def average_stochastic_ldos(num_av, E, M, W_vector, W_strenght):
    
    size = len(W_vector)
    ldos_av = np.zeros(size).astype(np.cdouble)
    ldos_va = np.zeros(size).astype(np.cdouble)
    
    for i in range (num_av):
        r_init = get_random_vector_ldos(size) # get_random_vector_ldos(size)
        result_iter = iterate_vector(E, M, r_init, W_vector, W_strenght)
        
        ldos_av_old = ldos_av
        ldos_av += (result_iter - ldos_av)/(i + 1)
        ldos_va += ((result_iter - ldos_av_old)*(result_iter.conj() - ldos_av.conj()) - ldos_va)/(i + 1) 
         
    return ldos_av, ldos_va

# -----------------------------------------
# EXACT LDOS WITH DIAGONALIZATION

@njit
def dirac_delta(x, M, E_alpha):

    x /= (2 + 0.1 + W_strenght/2)
    E_alpha /= (2 + 0.1 + W_strenght/2)

    delta = 1 # np.ones(len(x))

    for i in range (M - 1):
        delta += 2*chebyshev_polynomial(x, i + 1)*chebyshev_polynomial(E_alpha, i + 1)*jackson_kernel(M, i + 1)

    return (delta/(np.pi*np.sqrt(1 - x**2)))/(2 + 0.1 + W_strenght/2)

@njit 
def LDOS_exact(eigenvectors, eigenvalues, E, M): #index means eigenvector and corresponding eigenvalue

    size = len(eigenvalues)
    LDOS = np.zeros(size)

    for i in range (size):

        LDOS += np.abs(eigenvectors[:, i])**2*dirac_delta(E, M, eigenvalues[i])

    return LDOS

# -----------------------------------------
# EXACT ERROR

@njit 
def exact_variance(eigenvectors, eigenvalues, energy, M, num_med):

    size = len(eigenvalues)
    variance = np.zeros(size)

    for i in range (size):

        variance += np.abs(eigenvectors[:, i])**2*dirac_delta(energy, M, eigenvalues[i])**2

    #variance += - LDOS_exact(eigenvectors, eigenvalues, energy, M)**2

    return variance/num_med

index = np.argmin(np.abs(E))
energy = E[index]

# -----------------------------------------------------------------------------------------
moments          = int(sys.argv[2])
num_points       = int(sys.argv[3])
nn               = int(sys.argv[4])
nm               = 2**np.arange(nn)
num_medias_graph = int(sys.argv[5])

exactv = exact_variance(v, E, energy, moments, 1)

ldos   = LDOS_exact(v, E, energy, moments)
minimo = np.min(np.log(exactv))
maximo = np.max(np.log(exactv))
points = np.linspace(minimo, maximo, num_points)
points_index = np.zeros(num_points).astype(dtype=int)

for i in range (num_points):
    points_index[i] = int(np.argmin(np.abs(np.log(exactv) - points[i])))

# -----------------------------------------------------------------------------------------

@njit(parallel=True)
def looping():
    desvios = np.zeros((nn, num_points), dtype=float)

    for n in range(num_medias_graph):
        for i in prange(len(nm)):
            num_av = nm[i]
            m, v1 = average_stochastic_ldos(num_av, energy, moments, W_vector, W_strenght)
            for j in range (len(points_index)):
                r = points_index[j]
                x = np.abs(v1[r]/exactv[r] - 1.0)
                desvios[i,j] += (x - desvios[i,j])/(n + 1)

    return desvios
# -----------------------------------------------------------------------------------------

desvios = looping()

labels = np.zeros(num_points + 1)
labels[0] = 0

for i in range (num_points):
    labels[i + 1] = exactv[points_index[i]]

two_d_array = np.zeros((len(desvios) + 1, num_points + 1))
two_d_array[0, :] = labels 
two_d_array[1:, 0] = nm 
two_d_array[1:, 1:] = desvios

np.savetxt('dados.dat', two_d_array)

for j,r in enumerate(points_index):
    plt.plot(nm, desvios[:,j],label=str(exactv[r]))
    
plt.yscale('log')
plt.xscale('log')
plt.legend()

plt.show()