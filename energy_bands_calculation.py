import math
import numpy as np
import sympy
def energy_band_values(v1_3, v1_4, v1_11, v2_3, v2_4, v2_11, k_vec, energy_band_array, c, d_v):
    pseudopotential1_3 = v1_3
    pseudopotential2_3 = v2_3
    vector_magnitude3 = c*math.sqrt(3)
    pseudopotential1_4 = v1_4
    pseudopotential2_4 = v2_4
    #vector_magnitude8 = c*math.sqrt(8)
    vector_magnitude4 = c*math.sqrt(4)
    pseudopotential1_11 = v1_11
    pseudopotential2_11 = v2_11
    vector_magnitude11 = c*math.sqrt(11)
    atomic_basis_vector = d_v
    magnitude_threshold = .01
    pseudopotential1 = 0
    pseudopotential2 = 0
    matrix_size = energy_band_array.shape[0]
    energy_levels = np.zeros([8])
    k_squared = np.dot(k_vec, k_vec)
    matrix = np.zeros([matrix_size, matrix_size], dtype=complex)
    for i in range(0, matrix_size):
        h_i = energy_band_array[i,:]
        k_i = k_vec + h_i
        for j in range(0, matrix_size):
            delta = sympy.KroneckerDelta(i, j)
            h_j = energy_band_array[j,:]
            q_vec = h_i - h_j
            q_magnitude = math.sqrt(np.dot(q_vec, q_vec))
            if delta == 1:
                pseudopotential1 = np.dot(k_i, k_i)
                pseudopotential2 = np.dot(k_i, k_i)
            elif abs(q_magnitude - vector_magnitude3) < magnitude_threshold:
                pseudopotential1 = pseudopotential1_3
                pseudopotential2 = pseudopotential2_3
            elif abs(q_magnitude - vector_magnitude4) < magnitude_threshold:
                pseudopotential1 = pseudopotential1_4
                pseudopotential2 = pseudopotential2_4
            elif abs(q_magnitude - vector_magnitude11) < magnitude_threshold:
                pseudopotential1 = pseudopotential1_11
                pseudopotential2 = pseudopotential2_11
            else:
                pseudopotential1 = 0
                pseudopotential2 = 0
            matrix[i, j] = np.dot(k_i, k_i)*delta + pseudopotential1 + pseudopotential2*np.exp((-1j*np.dot(q_vec, atomic_basis_vector)))
    eigenvalues = np.linalg.eigvals(matrix)
    real_eigenvalues = eigenvalues.real
    real_eigenvalues = np.sort(real_eigenvalues, axis=0)
    real_eigenvalues = np.unique(real_eigenvalues.round(decimals=6))
    energy_levels = np.empty([8])
    for n in range(8):
        energy_levels[n] = real_eigenvalues[n]
    return energy_levels