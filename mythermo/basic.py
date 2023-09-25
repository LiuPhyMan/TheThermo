from math import exp, pi, sqrt, log

import numpy as np
from myconst import c as light_c, h, Hz2K, nm2Hz_f, nm2cm, h, m_e, kB


def Bnu(*, nuHz, T_K):
    try:
        tmp = 1/(exp(nuHz*Hz2K/T_K) - 1)
    except:
        tmp = 0
    return 2*h*nuHz**3/light_c**2*tmp


def avgBnu(*, nuHz_rng: tuple[float], T_K: float) -> float:
    return integral_Bnu(T_K=T_K, nuHz_rng=nuHz_rng)/(nuHz_rng[1] - nuHz_rng[0])


def dBnu_dT(*, nuHz, T_K):
    tmp = nuHz*Hz2K/T_K
    return tmp/T_K*Bnu(nuHz=nuHz, T_K=T_K)/(1 - exp(-tmp))


def integral_Bnu(*, T_K, nuHz_rng):
    _from, _to = nuHz_rng
    tmp = 0
    num = 100
    dx = (_to - _from)/(num - 1)
    for nu in np.linspace(_from, _to, num):
        tmp = tmp + Bnu(nuHz=nu, T_K=T_K)
    return (tmp - 0.5*Bnu(nuHz=_from, T_K=T_K) -
            0.5*Bnu(nuHz=_to, T_K=T_K))*dx


class WavelengthRange(object):
    __slots__ = ["wvlnm", "nuHz", "wvncm", "TempK",
                 "D_wvlnm", "D_nuHz", "D_wvncm", "D_TempK"]

    def __init__(self, wvlnm_start: float, wvlnm_end: float) -> None:
        self.wvlnm = (wvlnm_start, wvlnm_end)
        self.nuHz = (nm2Hz_f(wvlnm_end), nm2Hz_f(wvlnm_start))
        self.wvncm = (1/(wvlnm_end*nm2cm), 1/(wvlnm_start*nm2cm))
        self.TempK = (self.nuHz[0]*Hz2K, self.nuHz[1]*Hz2K)
        self.D_wvlnm = self.wvlnm[1] - self.wvlnm[0]
        self.D_nuHz = self.nuHz[1] - self.nuHz[0]
        self.D_wvncm = self.wvncm[1] - self.wvncm[0]
        self.D_TempK = self.TempK[1] - self.TempK[0]

def Le_th(T_K: float) -> float:
    return h/sqrt(2*pi*m_e*kB*T_K)

def CoulombLogarithm(*, ne: float, T_K: float, Zi:int=1) -> float:
    r"""Standard unit."""
    return 15.233732608189301 + 1.5*log(T_K) - 0.5*log(ne) - log(Zi)