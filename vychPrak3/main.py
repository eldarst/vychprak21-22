import numpy as np
from scipy.integrate import quadrature
from math import cos, factorial, fabs
a, b = 0, 1
N = 10

gauss_nodes_4n_xk = [-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]
gauss_nodes_4n_ck = [0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451]
gauss_nodes_5n_xk = [-0.9051798459, -0.5384693101, 0, 0.5384693101, 0.9051798459]
gauss_nodes_5n_ck = [0.2369268851, 0.4786286705, 0.5688888889, 0.4786286705, 0.2369268851]

for i in range(4):
    gauss_nodes_4n_xk[i] = (b-a)/2*gauss_nodes_4n_xk[i]+(b+a)/2
    gauss_nodes_4n_ck[i] = (b-a)/2*gauss_nodes_4n_ck[i]

for i in range(5):
    gauss_nodes_5n_xk[i] = (b-a)/2*gauss_nodes_5n_xk[i]+(b+a)/2
    gauss_nodes_5n_ck[i] = (b-a)/2*gauss_nodes_5n_ck[i]


def K(x, y):
    return 1.0 / (2 + x**2 + y**2)


def f(x):
    return 1 - x + x**2

def solve_MMK(n, gauss_nodes_xk, gauss_nodes_ck):
    D = np.zeros((n, n))
    g = np.zeros(n)

    for i in range(n):
        for j in range(n):
            D[i][j] = (i == j) + gauss_nodes_ck[j] * K(gauss_nodes_xk[i], gauss_nodes_xk[j])

    for i in range(n):
        g[i] = f(gauss_nodes_xk[i])

    z = np.linalg.solve(D, g)
    return z

def get_values(u):
    h = (b - a) / N
    values = np.zeros((N + 1, 2))
    for k in range(N + 1):
        values[k, 0] = a+k*h
        values[k, 1] = u(a+k*h)

    return values

def norm(res1, res2):
    return max([abs(res2[i][1] - res1[i][1]) for i in range(N + 1)])

def values(x_vals, u):
    return [u(x) for x in x_vals]

z_4 = solve_MMK(4, gauss_nodes_4n_xk, gauss_nodes_4n_ck)
u_new_4 = lambda x: f(x) - gauss_nodes_4n_ck[0] * K(x, gauss_nodes_4n_xk[0]) * z_4[0] - gauss_nodes_4n_ck[1] * K(x, gauss_nodes_4n_xk[1]) * z_4[1] - gauss_nodes_4n_ck[2] * K(x, gauss_nodes_4n_xk[2]) * z_4[2] - gauss_nodes_4n_ck[3] * K(x, gauss_nodes_4n_xk[3]) * z_4[3]

val_4 = get_values(u_new_4)



z_5 = solve_MMK(5, gauss_nodes_5n_xk, gauss_nodes_5n_ck)
u_new_5 = lambda x: f(x) - gauss_nodes_5n_ck[0] * K(x, gauss_nodes_5n_xk[0]) * z_5[0] - gauss_nodes_5n_ck[1] * K(x, gauss_nodes_5n_xk[1]) * z_5[1] - gauss_nodes_5n_ck[2] * K(x, gauss_nodes_5n_xk[2]) * z_5[2] - gauss_nodes_5n_ck[3] * K(x, gauss_nodes_5n_xk[3]) * z_5[3] - gauss_nodes_5n_ck[4] * K(x, gauss_nodes_5n_xk[4]) * z_5[4]
val_5 = get_values(u_new_5)

delta = norm(val_5, val_4)

x = np.arange(-100, 100, 1)


def alpha(k):
    return lambda x: (-1)**k*x**(2*k)/2

def beta(k):
    return lambda x: x**(2*k)/factorial(2*k)

def scalar(y, z):
    result, error = quadrature(lambda x: y(x) * z(x), a, b)
    return result

def solve_change(n):
    G = np.zeros((n, n))
    F = np.zeros((n, 1))

    for i in range(n):
        for j in range(n):
            G[j][i] = (i == j) + scalar(beta(j), alpha(i))

    for i in range(n):
        F[i] = scalar(beta(i), lambda x: f(x))

    A = np.linalg.solve(G, F)

    def U(x):
        return f(x) + sum([A[i] * alpha(i)(x) for i in range(n)])

    return U

xl = lambda x: x + 1
yl = lambda x: x + 3

result = scalar(alpha(1), beta(1))

u_3 = solve_change(3)
u_4 = solve_change(4)

val_change_3 = get_values(u_3)
val_change_4 = get_values(u_4)

delta_change = norm(val_5, val_4)