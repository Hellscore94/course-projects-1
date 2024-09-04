import numpy as np


def mesh_function(f, t):
    n = len(t)
    arr = np.zeros(n)
    
    for i in range(n):
        arr[i] = f(t[i]) 

    return arr

def func(t):
    if t > 3:
        return np.exp(-3*t)
    
    else:
        return np.exp(-t)

def test_mesh_function():
    t = np.array([1, 2, 3, 4])
    f = np.array([np.exp(-1), np.exp(-2), np.exp(-3), np.exp(-12)])
    fun = mesh_function(func, t)
    assert np.allclose(fun, f)
