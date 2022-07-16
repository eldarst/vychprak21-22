import pandas as pd
import numpy as np

def a(x, t):
    return 1

def b(x, t):
    return 0

def c(x, t):
    return 0

def p(x):
    return 1

def alpha1(t):
    return 1

def alpha2(t):
    return 0

def beta1(t):
    return 1

def beta2(t):
    return 0

T = 0.1


def approximate_L(uk, i, k, N, M):
    h = 1 / N
    t = T / M
    first = a(h*i, t*k) * (uk[i + 1] - 2*uk[i] + uk[i - 1]) / h**2
    second = b(h*i, t*k) * (uk[i + 1] - uk[i-1]) / 2*h
    third = c(h*i, t*k) * uk[i]
    return first + second + third


def error(u, N, M):
    h = 1 / N
    t = T / M
    err = []
    for k in range (M + 1):
        for i in range (N + 1):
            err.append(abs(u[k][i] - U(i * h,k * t)))
    return max(err)


def U(x, t):
    return x ** 3 + t ** 3

def phi(x):
    return x ** 3

def alpha(t):
    return t ** 3

def beta(t):
    return 1 + t ** 3

def f(x, t):
    return 3 * t ** 2 - 6 * x


def u_0(N, M):
    h = 1 / N
    u = np.zeros((M + 1, N + 1))
    for i in range(N + 1):
        u[0][i] = phi(h * i)
    return u

def obvious_schema(N, M):
    h = 1 / N
    t = T / M
    u = u_0(N, M)
    for k in range(1, M+1):
        for i in range(1, N):
            u[k][i] = u[k-1][i] + (approximate_L(u[k-1], i, k - 1, N, M) + f(h * i, t * (k - 1))) * t
        u[k][0] = alpha(t * k) / alpha1(t * k)
        u[k][N] = beta(t * k) * beta1(t * k)
    return u

def progonka(N, A, B, C, G):
    solve = np.zeros((N + 1, 1))
    s = np.zeros((N + 1, 1))
    t = np.zeros((N + 1, 1))

    s[0] = C[0] / B[0]
    t[0] = -G[0] / B[0]

    for i in range(1, N + 1):
        s[i] = C[i] / (B[i] - A[i] * s[i - 1])
        t[i] = (A[i] * t[i - 1] - G[i]) / (B[i] - A[i] * s[i - 1])
    solve[N] = t[N]
    for i in range(N - 1, -1, -1):
        solve[i] = solve[i + 1] * s[i] + t[i]
    return solve.reshape(N + 1)

def weights_schema(sigma, N, M):
    if sigma == 0:
        return obvious_schema(N, M)
    h = 1 / N
    t = T / M
    u = u_0(N, M)
    r = t / h ** 2


    for k in range(1, M + 1):
        A = np.zeros((N + 1, 1))
        B = np.zeros((N + 1, 1))
        C = np.zeros((N + 1, 1))
        G = np.zeros((N + 1, 1))

        A[0] = 0
        B[0] = -1 * alpha1(t * k)
        C[0] = 0
        G[0] = alpha(t * k)

        A[N] = 0
        B[N] = -1 * beta1(t * k)
        C[N] = 0
        G[N] = beta(t * k)

        for i in range(1, N):
            G[i] = -u[k - 1][i] - t * ((1 - sigma) * approximate_L(u[k - 1], i, k - 1, N, M) + t * f(i * h, t * (k - 1 + sigma)))
            A[i] = t * sigma * (a(i * h, k * t) / (h ** 2) - b(i * h, k * t) / (2 * h))
            B[i] = t * -sigma * (-2 * a(i * h, k * t) / h ** 2 + c(i * h, k * t)) + 1
            C[i] = t * sigma * (a(i * h, k * t) / h ** 2 + b(i * h, k * t) / (2 * h))

        u[k] = progonka(N, A, B, C, G)
    return u

N = 20
M = 100
h = 1 / N
tau = T / M

weights_half = weights_schema(0.5, N, M)

