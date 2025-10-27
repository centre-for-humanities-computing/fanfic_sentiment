from numpy import *
import numpy as np
# from numba import jit

# @jit

# def detrending_coeff(win_len , order):

# #win_len = 51
# #order = 2
# 	n = (win_len-1)/2
# 	A = np.array([ones((win_len,order+1))])
# 	x = np.arange(-n , n+1)
# 	for j in range(0 , order + 1):
# 		A[:,j] = np.array([x ** j]).T

# 	coeff_output = (A.T * A).I * A.T
# 	return coeff_output , A

# coeff_output,A = detrending_coeff(5,2)
# print(coeff_output)
# print(A)

def detrending_coeff(win_len, order):
    n = (win_len - 1) / 2
    x = np.arange(-n, n+1)
    # create A as (win_len, order+1)
    A = np.ones((win_len, order+1))
    for j in range(order + 1):
        A[:, j] = x ** j
    # use @ for matrix multiplication
    coeff_output = np.linalg.inv(A.T @ A) @ A.T
    return coeff_output, A