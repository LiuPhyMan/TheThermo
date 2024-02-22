# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 23.May.2023
"""

from math import sqrt, log, exp, pi
from typing import List, NoReturn

import myconst as cns
import numpy as np
from myconst import (atm, Hz2K, kB, e, epsilon_0, Hz2nm_f, m2Hz_f,
                     light_c as C)
from myspecie import spc_dict

from . import gff, gffint
from .basic import Bnu


class _AbsComp(object):
  __slots__ = ("spcs_str", "spcs",
               "n_spcs", "n_elem",
               "Zc",
               "relM", "absM",
               "xj", "nj",
               "Aij", "elems")

  def __init__(self, *, spcs_str: List[str]) -> NoReturn:
    self.spcs_str = spcs_str
    self.n_spcs = len(spcs_str)
    self.spcs = [spc_dict[_] for _ in self.spcs_str]
    self._set_elements_by_species()
    self.Zc = np.array([_.Zc for _ in self.spcs])
    self.relM = [_.relM for _ in self.spcs]
    self.absM = [_.absM for _ in self.spcs]
    self.xj = np.zeros(self.n_spcs)
    self.nj = np.zeros(self.n_spcs)

  def _set_Aij(self) -> NoReturn:
    self.Aij = np.zeros((self.n_elem, self.n_spcs))
    for _j in range(self.n_spcs):
      for _i in range(self.n_elem):
        self.Aij[_i, _j] = self.spcs[_j].get_nElem(self.elems[_i])

  def _set_elements_by_species(self):
    # Aij = [ 1 0 1
    #         0 1 0]
    elems = []
    for _spcs in self.spcs:
      elems = elems + list(_spcs.elems.keys())
    self.elems = tuple(set(elems))
    self.n_elem = len(self.elems)
    self._set_Aij()
    # set Aij

  def set_elements(self, elems: tuple):
    self.elems = elems
    self.n_elem = len(self.elems)
    self._set_Aij()

  @property
  def nj_dict(self):
    _dct = dict()
    for i in range(self.n_spcs):
      _dct[self.spcs_str[i]] = self.nj[i]
    return _dct


class Composition(_AbsComp):
  __slots__ = ("p_atm", "T_K", "bf_cs",)

  def __init__(self, *, spcs_str: List[str]) -> NoReturn:
    super().__init__(spcs_str=spcs_str)
    self.T_K = None
    self.p_atm = None

  @property
  def ne(self):
    r"""Numerical density of electron."""
    return self.nj[self.spcs_str.index("e")]

  @property
  def nion(self):
    r"""Numerical density of ions."""
    return (self.nj[self.Zc > 0]).sum()

  @property
  def avgZc1(self):
    r""""""
    return ((self.Zc*self.nj)[self.Zc > 0]).sum()/self.nion

  @property
  def avgZc2(self):
    r""""""
    return ((self.Zc**2*self.nj)[self.Zc > 0]).sum()/self.nion

  @property
  def avgZp(self):
    r""""""
    return self.avgZc2/self.avgZc1

  @property
  def WS_radius(self):
    r""""""
    return (3/4/pi/self.nion)**(1/3)

  @property
  def eleFreq(self):
    return 8.97866758337293*sqrt(self.ne)

  def refracIndex(self, *, laser_wvl):
    laser_freq = m2Hz_f(laser_wvl)
    assert self.eleFreq < laser_freq
    return sqrt(1 - (self.eleFreq/laser_freq)**2)

  @property
  def norm_field(self):
    tmp = (self.Zc*self.nj**(2/3))[self.Zc > 0].sum()
    return 0.5*(4/15)**(2/3)*e/epsilon_0*tmp

  @property
  def ion_distn(self):
    return (self.nj[self.Zc > 0].sum()*4*pi/3)**(-1/3)

  # ------------------------------------------------------------------------- #
  # free-free transition
  # ------------------------------------------------------------------------- #
  def j_ff_Hz(self, *, nuHz: float, T_K: float) -> float:
    r""" Bremsstrahlung (free-free, electron-ion) emission per Hz
          in the unit of W m^-2 sr^-1 Hz^-1. """
    tmp = sum([self.Zc[_i]**2*self.nj[_i]*gff(wvlnm=Hz2nm_f(nuHz),
                                              T_K=T_K, Z=self.Zc[_i])
               for _i in range(self.n_spcs) if self.Zc[_i] > 0])
    return 5.4443668e-52/sqrt(T_K)*self.ne*exp(-nuHz*Hz2K/T_K)*tmp

  def j_ff_m(self, *, wvlnm: float, T_K: float) -> float:
    r""" Bremsstrahlung (free-free, electron-ion) emission per m
          in the unit of W m^-2 sr^-1 m^-1. """
    return self.j_ff_Hz(nuHz=cns.nm2Hz_f(wvlnm), T_K=T_K)*C/(wvlnm*cns.nm2m)**2

  def tot_j_ff(self, *, T_K: float, specSpc: str = None) -> float:
    r""" Bremsstrahlung (free-free, electron-ion) emission
          in the unit of W m^-2 sr^-1. """
    if specSpc is None:
      tmp = sum([self.Zc[_i]**2*gffint(T_K=T_K, Z=self.Zc[_i])*self.nj[_i]
                 for _i in range(self.n_spcs) if self.Zc[_i] > 0])
    else:
      assert specSpc in self.spcs_str
      assert self.Zc[self.spcs_str.index(specSpc)] > 0
      tmp = sum([self.Zc[_i]**2*gffint(T_K=T_K, Z=self.Zc[_i])*self.nj[_i]
                 for _i in range(self.n_spcs) if self.spcs_str[_i] == specSpc])
    return 1.1344220e-41*self.ne*sqrt(T_K)*tmp

  def kappa_ff(self, *, nuHz: float, T_K: float) -> float:
    return self.j_ff_Hz(nuHz=nuHz, T_K=T_K)/Bnu(nuHz=nuHz, T_K=T_K)

  # ------------------------------------------------------------------------- #
  # free-bound transition
  # ------------------------------------------------------------------------- #
  # def j_fb_Hz(self, *, nuHz: float, T_K: float) -> float:
  #   return self.kappa_bf(nuHz=nuHz, T_K=T_K)*Bnu(nuHz=nuHz, T_K=T_K)

  # def j_fb_m(self, *, wvlnm: float, T_K: float) -> float:
  #   return self.j_fb_Hz(nuHz=cns.nm2Hz_f(wvlnm), T_K=T_K)*cns.light_c/(
  #       wvlnm*cns.nm2m)**2

  # def avg_j_fb_Hz(self, *, wv_rng: WVRng, T_K: float,
  #                 num_point: int = 100) -> float:
  #   j_seq = [self.j_fb_Hz(nuHz=_nu, T_K=T_K)
  #            for _nu in
  #            np.linspace(wv_rng.nuHz[0], wv_rng.nuHz[1], num=num_point)]
  #   return np.mean(j_seq)

  # def kappa_bf(self, *, nuHz: float, T_K: float) -> float:
  #   kappa = 0
  #   for _i in range(self.n_spcs):
  #     kappa = kappa + self.bf_cs[_i].norm_cs(nuHz=nuHz, T_K=T_K)*self.nj[
  #     _i]/ \
  #             self.spcs[_i].qint(T_K=T_K)
  #   return kappa*(1 - exp(-nuHz*Hz2K/T_K))

  # ------------------------------------------------------------------------- #
  # bound-bound transition
  # ------------------------------------------------------------------------- #
  # def j_bb_m(self, *, wv):
  #   pass
  #
  # def avg_j_bb_Hz(self, *, wv_rng: WVRng, T_K: float) -> float:
  #   return np.dot(self.nj, [_spc.norm_j_bb(T_K=T_K, wvlnm_rng=wv_rng.wvlnm)
  #                           for _spc in self.spcs])/wv_rng.D_nuHz
  #
  # def avg_kappa_bb(self, *, wv_rng: WVRng, T_K: float):
  #   return self.avg_j_bb_Hz(T_K=T_K, wv_rng=wv_rng)*wv_rng.D_nuHz/ \
  #     integral_Bnu(T_K=T_K, nuHz_rng=wv_rng.nuHz)

  # ------------------------------------------------------------------------- #
  # def avg_j_Hz(self, *, wv_rng: WVRng, T_K: float) -> float:
  #   return self.avg_j_ff_Hz(wv_rng=wv_rng, T_K=T_K) + \
  #     self.avg_j_fb_Hz(wv_rng=wv_rng, T_K=T_K) + \
  #     self.avg_j_bb_Hz(wv_rng=wv_rng, T_K=T_K)
  #
  # def avg_kappa(self, *, wv_rng: WVRng, T_K: float) -> float:
  #   return self.avg_j_Hz(T_K=T_K, wv_rng=wv_rng)/avgBnu(nuHz_rng=wv_rng.nuHz,
  #                                                       T_K=T_K)

  # ------------------------------------------------------------------------- #
  def rdcd_mu0(self, *, T_K: float):
    return np.array([_.rdcd_mu0(T_K=T_K) for _ in self.spcs])

  def get_rho(self):
    r""" Mass density in the unit of kg m^-3."""
    return np.dot(self.nj, self.absM)

  def get_H(self, *, T_K: float):
    r""" Enthalpy in the unit of J kg^-1"""
    return np.dot(self.nj,
                  [_spc.get_h(T_K=T_K) for _spc in self.spcs])/self.get_rho()

  def set_lte_comp(self, *, p_atm, T_K, elem_comp) -> NoReturn:
    # ln N
    elem_bj = tuple(elem_comp[_spc] for _spc in self.elems)
    self.T_K = T_K
    self.p_atm = p_atm
    Nj = np.ones(self.n_spcs)/self.n_spcs
    N = 1
    lnN = N
    lnNj = np.log(Nj)
    # Solve Mkk \dot xk = bk    {k = self.n_elem + 1}
    Mkk = np.zeros((self.n_elem + 1, self.n_elem + 1))
    bk = np.zeros(self.n_elem + 1)
    # LJm-factor
    factor = np.ones(self.n_spcs)
    rdcd_mu0 = self.rdcd_mu0(T_K=T_K)
    for _i in range(500):
      # check whether log(Nj) is nan.
      with np.errstate(divide='ignore'):
        _lnNj_tmp = np.nan_to_num(np.log(Nj), neginf=-512)
      # ---
      rdcd_mu = rdcd_mu0 + log(p_atm) + _lnNj_tmp - log(Nj.sum())
      Mkk[:-1, :-1] = np.dot(self.Aij, (self.Aij*Nj).transpose())
      Mkk[:-1, -1] = np.dot(self.Aij, Nj)
      Mkk[-1, :-1] = np.dot(self.Aij, Nj)
      Mkk[-1, -1] = Nj.sum() - N
      bk[:-1] = elem_bj - np.dot(self.Aij, Nj) + \
                np.dot(self.Aij, Nj*rdcd_mu)
      bk[-1] = N - Nj.sum() + np.dot(Nj, rdcd_mu)
      sol = np.linalg.solve(Mkk, bk)
      dlnN = sol[-1]
      dlnNj = dlnN + np.dot(sol[:-1], self.Aij) - rdcd_mu
      for j in range(self.n_spcs):
        if (lnNj[j] - lnN) > -18.420680743952367:
          factor[j] = 2/abs(dlnNj[j]) if (abs(dlnNj[j]) >= 2) else 1
        elif dlnNj[j] >= 0:
          with np.errstate(divide='ignore'):
            factor[j] = abs(
              (lnNj[j] - lnN + 9.210340371976182)/(dlnNj[j] - dlnN))
        else:
          factor[j] = (2/(5*abs(dlnN))) if (abs(dlnN) > 2.5) else 1
      e_factor = min(1, np.min(factor))
      lnN = lnN + e_factor*dlnN
      lnNj = lnNj + e_factor*dlnNj
      N = exp(lnN)
      Nj = np.exp(lnNj)
      if np.all(Nj*np.abs(dlnNj)/Nj.sum() <= 1e-25):
        break
    self.xj = Nj/Nj.sum()
    self.nj = self.xj*p_atm*atm/(kB*T_K)

  def set_comp_by_dict(self, *, _dict: dict, default_value=0) -> NoReturn:
    self.nj = np.zeros(self.n_spcs)
    for _spc in _dict:
      assert _spc in self.spcs_str
      self.nj[self.spcs_str.index(_spc)] = _dict[_spc]
    self.xj = self.nj/self.nj.sum()

  def ionDebL(self) -> float:
    tmp = 4*pi*self.avgZp*self.ne*cns.e2_eV/(self.T_K*cns.K2eV)
    return sqrt(1/tmp)

  def eleDebL(self) -> float:
    r"""Classical electron debye screening length"""
    tmp = 4*pi*self.ne*cns.e2_eV/(self.T_K*cns.K2eV)
    return sqrt(1/tmp)

  def totDebL(self) -> float:
    tmp = 4*pi*(self.avgZp + 1)*self.ne*cns.e2_eV/(self.T_K*cns.K2eV)
    return sqrt(1/tmp)

  def spIPD_eV(self, *, Z: int) -> float:
    assert Z >= 1
    L = Z/self.avgZc1*(self.WS_radius/self.totDebL())**3
    return self.T_K*cns.K2eV/2/(self.avgZp + 1)*((1 + L)**(2/3) - 1)
