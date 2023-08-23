import numpy as np
from myconst import c as light_c, h, k as kB
from math import exp

def Bnu(nu, T_K):
    return 2*h*nu**3/light_c**2/(exp(h*nu/kB/T_K)-1)

def integral_Bnu(T_K, nu_range):
    _from, _to = nu_range
    tmp = 0
    num = 100
    dx = (_to - _from) / (num-1)
    for nu in np.linspace(_from, _to, num):
        tmp = tmp + Bnu(nu, T_K)
    return (tmp - 0.5*Bnu(_from, T_K) - 0.5*Bnu(_to, T_K)) * dx

