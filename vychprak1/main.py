import scipy.integrate
import numpy as np
import pandas as pd
import math
from scipy.special import jacobi
from scipy.integrate import solve_bvp

n = 10
p = lambda x: (x - 2.0) / (x + 2.0)
q = lambda x: x
r = lambda x: (1.0 - math.sin(x))
f = lambda x: x ** 2


def w(i, x):
    return (1.0 - x ** 2.0) * jacobi(n=i - 1, alpha=1, beta=1)(x)


def d_w(i, x):
    return (-2.0) * i * jacobi(n=i, alpha=0, beta=0)(x)


def dd_w(i, x):
    return -(i + 1.0) * i * jacobi(n=i - 1, alpha=1, beta=1)(x)


def psi_i(i, x):
    return jacobi(n=i - 1, alpha=0, beta=0)(x)


def psi(i):
    return lambda x: jacobi(n=i - 1, alpha=0, beta=0)(x)


def scalar(y, z):
    result, error = scipy.integrate.quad(lambda x: y(x) * z(x), -1, 1)
    return result


def integrate(i, j):
    def und_integral(x):
        return (p(x) * dd_w(i, x) + q(x) * d_w(i, x) + r(x) * w(i, x)) * psi_i(j, x)

    result, error = scipy.integrate.quad(und_integral, -1.0, 1.0)  # integrate(und_integral, -1, 1)
    return result


def moment_method(n):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i, j] = integrate((j + 1), (i + 1))
    b = np.zeros((n, 1))
    for i in range(n):
        b[i] = scalar(f, psi(i + 1))
    condition_number = np.linalg.norm(A) * np.linalg.norm(np.linalg.inv(A))
    c = np.linalg.solve(A, b)

    print([w(i + 1, -0.5) for i in range(n)])
    print(0)
    print([w(i + 1, 0) for i in range(n)])
    print(0.5)
    print([w(i + 1, 0.5) for i in range(n)])

    print(c)

    def y_n_c(x):
        return sum([float(c[i]) * w(i + 1, x) for i in range(n)])

    print(y_n_c(-0.5))
    print(y_n_c(0))
    print(y_n_c(0.5))

    return pd.DataFrame(A), pd.DataFrame(b), pd.DataFrame(c), condition_number, y_n_c(-0.5), y_n_c(0), y_n_c(0.5)


def eq(x, y):
    u, v = y
    return np.vstack((v, ((np.sin(x) - 1.0) * u - x * v + x ** 2) * (x + 2.0) / (x - 2.0)))


def bc(ya, yb):
    return np.array([ya[0], yb[0]])


x = np.linspace(-1, 1, 101)
y0 = np.zeros((2, x.size))

res = solve_bvp(eq, bc, x, y0)

t_v = [res.sol(-0.5)[0], res.sol(0)[0], res.sol(0.5)[0]]
columns = ['n', 'mu(A)', 'y_n(-0.5)', 'y_n(0)', 'y_n(0.5)', 'delta(-0.5)', 'delta(0)', 'delta(0.5)']
df_mm = pd.DataFrame(columns=columns)
for n in range(3, 11):
    t = moment_method(n)
    row = {'n': n, 'mu(A)': t[3], 'y_n(-0.5)': t[4], 'y_n(0)': t[5],
           'y_n(0.5)': t[6], 'delta(-0.5)': t[4] - t_v[0],
           'delta(0)': t[5] - t_v[1], 'delta(0.5)': t[6] - t_v[2]}
    df_mm = df_mm.append(row, ignore_index=True)

print(df_mm)
