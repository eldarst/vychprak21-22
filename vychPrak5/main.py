import numpy as np
import pandas as pd
from scipy.integrate import quadrature
from scipy import integrate

def phi(x):
    return np.sin(3 * np.pi * x)

def psi(x, p):
    return np.sqrt(2) * np.sin(p* np.pi * x)

def scalar(y, z):
    result, error = quadrature(lambda x: y(x) * z(x), 0, 1)
    return result

N = 5
M = 5
h = 1.0 / N
tau = 0.1 / M

x_arr = np.linspace(0, 1, N + 1)
t_arr = np.linspace(0, 0.1, M + 1)
P = 15

def Fourier_solve(coefs, P, x, t):
    res = 0
    for p in range(P - 1):
        res += coefs[p] * np.exp(-np.pi ** 2 * (p + 1.0) ** 2 * t) * psi(x, p + 1)
    return res

def dif_Fourier_solve(coefs, x, t):
    return Fourier_solve(coefs, N, x, t)

#%%

coeffs = []
for p in range(1, P):
    coeffs.append(integrate.quad(lambda x: phi(x) * psi(x, p), 0, 1)[0])
np.around(coeffs, 3)

cp = np.zeros(P)
for p in range(1, P):
    cp[p - 1] = integrate.quad(lambda x: phi(x) * psi(x, p), 0, 1)[0]
np.around(cp, 3)
uf = np.zeros((M + 1, N + 1))

for i in range(M + 1):
    for j in range(N + 1):
        uf[i][j] = Fourier_solve(cp, P, x_arr[j], t_arr[i])