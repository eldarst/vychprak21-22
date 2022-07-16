import numpy as np
from scipy.special import jacobi
from scipy.integrate import quadrature
import pandas as pd


a, b = -1, 1
n = 7
a1, a2, b1, b2 = -0.7, -1, 0.75, 1


def p(x):
    return 1 / (2 + x / 3)


def q(x):
    return np.exp(x / 5)

# (y, z)
def scalar(y, z):
    return quadrature(lambda x: y(x) * z(x), a, b)[0]


def d(f):
    eps = 1e-10
    return lambda x: (f(x + eps) - f(x - eps)) / (2 * eps)


def d2(f):
    eps = 1e-5
    return lambda x: (f(x - eps) - 2.0 * f(x) + f(x+eps)) / eps**2

# [y, z]
def integrate(y, z):
    def fun(x):
        return p(x) * d(y)(x) * d(z)(x) + q(x) * y(x) * z(x)

    Ql = a1 / a2 * p(a) * y(a) * z(a)
    Qr = b1 / b2 * p(b) * y(b) * z(b)

    return quadrature(fun, a, b)[0] + Ql + Qr

def w(x, k):
    return np.sqrt((2 * k - 1) / 2) * jacobi(n=k - 1, alpha=0, beta=0)(x)


def w_k(k):
    return lambda x: w(x, k)


G = np.zeros((n, n))
G_L = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        G[i, j] = scalar(w_k(i + 1), w_k(j + 1))
        G_L[i, j] = integrate(w_k(i + 1), w_k(j + 1))

lambdas, y_rit = np.linalg.eig(G_L)
lambdas, y_rit = zip(*sorted(zip(lambdas, y_rit)))

x = np.linspace(-1, 1, 1000)
p_max = max(map(p, x))
p_min = min(map(p, x))
q_max = max(map(q, x))
q_min = min(map(q, x))

nu1 = 0.760936
nu2 = 1.9300737

l_1_min = nu1**2 * p_min + q_min
l_1_max = nu1**2 * p_max + q_max
l_2_min = nu2**2 * p_min + q_min
l_2_max = nu2**2 * p_max + q_max


def y(nu):
    C = (b2 * nu * np.sin(nu) - b1 * np.cos(nu)) / (b1 * np.sin(nu) + b2 * nu * np.cos(nu))
    f = lambda x: np.cos(nu * x) + C * np.sin(nu * x)
    return lambda x: f(x) / np.sqrt(scalar(f, f))


norm_eig_f_1 = y(nu1)
norm_eig_f_2 = y(nu2)

data={' ':['min', 'max'],'оценка lambda_1;': [l_1_min, l_1_max],
'невязка lambda_1;': [l_1_min - lambdas[0], l_1_max - lambdas[0]],
'оценка lambda_2;': [l_2_min , l_2_max],
'невязка lambda_2;': [l_2_min - lambdas[1] , l_2_max - lambdas[1]]}

lambd_1 = integrate(norm_eig_f_1, norm_eig_f_1)
lambd_2 = integrate(norm_eig_f_2, norm_eig_f_2)

columns = ['n', 'lambda(n)', 'lambda_diff', 'L*y-lambda*y']
table = pd.DataFrame(columns=columns)
lamb1_ex = lambdas[0]

for k in range(3, n+1):
    G_L_k = np.linalg.inv(G_L[:k,:k])
    G_k = G[:k, :k]
    G_res = np.dot(G_L_k, G_k)
    z0 = np.ones((k, 1))
    accuracy = 10
    for i in range(accuracy):
        z1 = np.dot(G_res, z0)
        lambMax = z1[0] / z0[0]
        z0= z1
        # z1 = np.dot(G_res, z0)
        # lamb = np.linalg.norm(z1, 2) / np.linalg.norm(z0, 2)
        # z1 = z1 / np.linalg.norm(z1, 2)
        # z0= z1
    lambMin = 1.0/lambMax
    _f = lambda x: sum([z0[j] * w(x, j+1) for j in range(0, k)])
    y=_f
    right = lambda x: lambMin * y(x)
    left = lambda x: -(d(p)(x) * d(y)(x) + p(x) * d2(y)(x)) + q(x)*y(x)
    diff = lambda x: left(x) - right(x)

    row = {'n': k, 'lambda(n)': lambMin, 'lambda_diff': lambMin - lamb1_ex, 'L*y-lambda*y': right(0) - left(0)}
    table = table.append(row, ignore_index=True)