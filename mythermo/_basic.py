import re
from functools import reduce
from math import exp, pi, sqrt, log
from pathlib import Path
from typing import NoReturn

import numpy as np
import yaml
from myconst import light_c, Hz2K, nm2Hz_f, nm2cm, h, m_e, kB, K2eV


def Bnu(*, nuHz, T_K: float):
  r""" Planck function in wavelength [Hz]
  unit: W m^-2 Hz^-1 sr^-1"""
  # try:
  #     tmp = 1/(exp(nuHz*Hz2K/T_K) - 1)
  # except:
  #     tmp = 0
  tmp = 1/(np.exp(nuHz*Hz2K/T_K) - 1)
  return 2*h*nuHz**3/light_c**2*tmp


# def B_nm(*, wvlnm, T_K):
#     return Bnu(nuHz=nm2Hz_f(wvlnm), T_K=T_K)*light_c/(wvlnm*1e-9)**2 *1e-9

def Bl_m(*, wvlnm, T_K: float):
  r""" Planck function in wavelength [m]
  unit: W m^-2 m^-1 sr^-1"""
  return Bnu(nuHz=nm2Hz_f(wvlnm), T_K=T_K)*light_c/(wvlnm*1e-9)**2


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

  def __init__(self, wvlnm_start: float, wvlnm_end: float) -> NoReturn:
    assert wvlnm_start < wvlnm_end
    self.wvlnm = (wvlnm_start, wvlnm_end)
    self.nuHz = (nm2Hz_f(wvlnm_end), nm2Hz_f(wvlnm_start))
    self.wvncm = (1/(wvlnm_end*nm2cm), 1/(wvlnm_start*nm2cm))
    self.TempK = (self.nuHz[0]*Hz2K, self.nuHz[1]*Hz2K)
    self.D_wvlnm = self.wvlnm[1] - self.wvlnm[0]
    self.D_nuHz = self.nuHz[1] - self.nuHz[0]
    self.D_wvncm = self.wvncm[1] - self.wvncm[0]
    self.D_TempK = self.TempK[1] - self.TempK[0]

  def isValueIn(self, *, wvlnm: float) -> bool:
    if (wvlnm >= self.wvlnm[0]) and (wvlnm <= self.wvlnm[1]):
      return True
    else:
      return False


def Le_th(T_K: float, mass: float = m_e) -> float:
  r""" Thermal de Broglie wavelength. """
  return h/sqrt(2*pi*mass*kB*T_K)


def CoulombLogarithm(*, ne: float, T_K: float, Zi: int = 1) -> float:
  r"""Standard unit. TODO"""
  # return 15.233732608189301 + 1.5*log(T_K) - 0.5*log(ne) - log(Zi)
  return 23 + 1.5*log(T_K*K2eV) - 0.5*log(ne*1e-6) - log(Zi)


def rdcdM(M1, M2):
  return M1*M2/(M1 + M2)


class ColliIntegralParas(object):
  def __init__(self, *, yaml_file: Path) -> NoReturn:
    self._file = yaml_file
    assert str(yaml_file).endswith('.yaml') or str(yaml_file).endswith('.yml')
    with yaml_file.open() as f:
      data = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert "Collision" in data
    assert "Reaction" in data
    self.collision = [re.match(r"([^,]+)\s*,\s*([^,]+)\s*,\s*(.*)", _).groups()
                      for _ in reduce(lambda x, y: x + y, data["Collision"])]
    self.reaction = data["Reaction"]
