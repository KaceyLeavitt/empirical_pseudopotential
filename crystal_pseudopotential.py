import math
import numpy as np
import sympy
import matplotlib.pyplot as plt
import timeit
import time
from mpl_toolkits.mplot3d import Axes3D
start_time = timeit.default_timer()
a = 10.26
c = 2*math.pi/a
b1 = np.array([-c, c ,c])
b2 = np.array([c, -c ,c])
b3 = np.array([c, c, -c])
radius = math.sqrt(11)*c
x_array = np.array([])
y_array = np.array([])
z_array = np.array([])
for m in range(-10, 10):
    for n in range(-10, 10):
        for l in range(-10, 10):
            h = m*b1 + n*b2 + l*b3
            if np.dot(h, h) <= radius**2:
                x_array = np.append(x_array, h[0])
                y_array = np.append(y_array, h[1])
                z_array = np.append(z_array, h[2])
"""
print (x_array)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_array, y_array, z_array)
plt.show()
"""
N = 11
x1 = np.linspace(0, 4, N, endpoint=True)
k_x = np.linspace(0, 4, N, endpoint=True)
k_y = np.linspace(0, 4, N, endpoint=True)
k_z = np.linspace(0, 4, N, endpoint=True)
for n in range(0, N):
    if (x1[n] >= 0) & (x1[n] < 1):
        k_x[n] = math.pi/a*(1 - x1[n])
        k_y[n] = math.pi/a*(1 - x1[n])
        k_z[n] = math.pi/a*(1 - x1[n])
    elif (x1[n] >= 1) & (x1[n] < 2):
        k_x[n] = 0
        k_y[n] = 2*math.pi/a*(x1[n] - 1)
        k_z[n] = 0
    elif (x1[n] >= 2) & (x1[n] < 3):
        k_x[n] = math.pi/(2*a)*(x1[n] - 2)
        k_y[n] = 2*math.pi/a
        k_z[n] = math.pi/(2*a)*(x1[n] - 2)
    elif (x1[n] >= 3) & (x1[n] <= 4):
        k_x[n] = 3*math.pi/(2*a)*(x1[n] - 8/3)
        k_y[n] = 2*math.pi/a
        k_z[n] = 3*math.pi/(2*a)*(x1[n] - 8/3)
    else:
        k_x[n] = 0
        k_y[n] = 0
        k_z[n] = 0
#plt.plot(x1, k_z)
#plt.show()

pseudopotential1 = -.21
vector_magnitude1 = c*math.sqrt(3)
pseudopotential2 = .04
vector_magnitude2 = c*math.sqrt(8)
pseudopotential3 = .08
vector_magnitude3 = c*math.sqrt(11)
atomic_basis_vector = np.array([a/4, a/4, a/4])
magnitude_threshold = .01
pseudopotential = 0

matrix_size = x_array.shape[0]
energy_levels = np.zeros([N, 8])

total_time = 0


def process(magnitude):
    if np.isclose(magnitude, vector_magnitude1):
        return pseudopotential1
    elif np.isclose(magnitude,vector_magnitude2):
        return pseudopotential2
    elif np.isclose(magnitude,vector_magnitude3):
        return pseudopotential3
    else:
        return 0

for n in range(N):
    k_vec = np.array([k_x[n], k_y[n], k_z[n]])
    k_squared = np.dot(k_vec, k_vec)
    matrix = np.zeros([matrix_size, matrix_size], dtype=complex)
    #h = np.vstack((x_array, np.vstack((y_array, z_array))))
    h = np.zeros((len(x_array),3))#figure out the mechanics of vstack + hstack in Python 3 relations? Is there dif?
    h[:,0] = x_array
    h[:,1] = y_array
    h[:,2] = z_array
    for i in range(0,matrix_size):
        k_temp = k_vec + h[i,:]
        q_vector = -(h - h[i])
        q_magnitudes = np.array([math.sqrt(np.dot(x,x))for x in q_vector])
        pre_result = np.array([process(q_mag)for q_mag in q_magnitudes])
        #print(pre_result)

        for j in range(0, matrix_size):
            matrix[i, j] = pre_result[j]*(1 + np.exp((-1j*np.dot(q_vector[j], atomic_basis_vector))))
        matrix[i, i] += np.dot(k_temp, k_temp)
        #print(q_vector,q_magnitudes)
        #results =
    #print(np.array([f(a) for a in x_array]))

    start1 = time.time()
    for i in range(0, matrix_size):
        h_i = np.array([x_array[i], y_array[i], z_array[i]])
        k_i = k_vec + h_i
        for j in range(0, matrix_size):
            delta = sympy.KroneckerDelta(i, j)
            h_j = np.array([x_array[j], y_array[j], z_array[j]])
            q_vec = h_i - h_j
            q_magnitude = math.sqrt(np.dot(q_vec, q_vec))
            if delta == 1:
                #pseudopotential = np.dot(k_i, k_i)
                pseudopotential = 0
            elif abs(q_magnitude - vector_magnitude1) < magnitude_threshold:
                pseudopotential = pseudopotential1
            elif abs(q_magnitude - vector_magnitude2) < magnitude_threshold:
                pseudopotential = pseudopotential2
            elif abs(q_magnitude - vector_magnitude3) < magnitude_threshold:
                pseudopotential = pseudopotential3
            else:
                pseudopotential = 0
            matrix[i, j] = np.dot(k_i, k_i)*delta + pseudopotential*(1 + np.exp((-1j*np.dot(q_vec, atomic_basis_vector))))
    print (matrix)
    end1 = time.time()
    eigenvalues = np.linalg.eigvals(matrix)#Use linalg.eigvalsh(np.real(matrix))
    end2 = time.time()
    print((end2-end1) > (end1 - start1))
    real_eigenvalues = eigenvalues.real
    real_eigenvalues = np.sort(real_eigenvalues, axis=0)
    for m in range(0, 8):
        energy_levels[n, m] = real_eigenvalues[m]
for k in range(0, 8):
    band_array = np.array([])
    for l in range(0, N):
        band_array = np.append(band_array, energy_levels[l, k])
    plt.plot(x1, band_array)
elapsed = timeit.default_timer() - start_time
print(elapsed)
plt.show()