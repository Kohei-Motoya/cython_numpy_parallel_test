import numpy as np
import convolve1
import convolve_py
import time

N = 500
f = np.arange(N*N, dtype=np.int).reshape((N,N))
g = np.arange(81, dtype=np.int).reshape((9, 9))

start_time = time.time()
convolve1.parallel_convolve(f, g)
end_time = time.time()
print('parallel cython: {0:.4f} s'.format(end_time - start_time))
start_time = time.time()
convolve1.naive_convolve(f, g)
end_time = time.time()
print('not parallel cython: {0:.4f} s'.format(end_time - start_time))
start_time = time.time()
convolve_py.naive_convolve(f, g)
end_time = time.time()
print('pure python: {0:.4f} s'.format(end_time - start_time))
