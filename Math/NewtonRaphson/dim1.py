"""
1変数Newton-Raphson法による非線形方程式の求解
"""
import numpy as np

f = lambda x : x**2 - 2.
df = lambda x : 2.*x

epsilon = 1e-5
x = 5. #初期解
for i in range(100):
    if np.abs(f(x)) < epsilon:
        break
    x = x - f(x)/df(x)

print(x)