"""
Newton-Raphson法による非線形連立方程式の求解
"""
import numpy as np

def f(X):
    x, y = X
    
    f1 = (x - 3.)**2 + y**2 - 3.
    f2 = np.sin(x) + np.exp(y-1) - 1.

    return np.array([f1, f2])

def Jacob(X):
    x, y = X
    return np.array([
        [2.*(x - 3.), 2.*y],
        [np.cos(x), np.exp(y - 1.)]
    ])

X = np.array([5., 2.]) #初期解
epsilon = 1e-5
for _ in range(100):
    if np.linalg.norm(f(X)) < epsilon:
        break
    X += np.linalg.solve(Jacob(X), -f(X))
print(X)