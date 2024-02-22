import re
from math import sqrt, pi
from pathlib import Path
from typing import NoReturn

import numpy as np
from myconst import kB, m_e
from mymath import M_coeff_1, M_coeff_2, N_coeff_1, N_coeff_2

from ._colli_intgl import (EmptyColli,
                           LJm6ColliIntegral,
                           CECI,
                           EleNeuColliIntegral,
                           ScreenCoulombColli)
from ._comp import Composition
from .basic import ColliIntegralParas


class PlasmaComp(object):
  __slots__ = ["comp",
               "p_atm", "T_K", "elem_comp",
               "rctn", "n_rctn", "rctn_Rik",
               "ci_paras_list", "ci_engine"]

  def __init__(self, *, spcs_str):
    r"""

    Parameters
    ----------
    spcs_str
    """
    assert spcs_str[0] == "e"
    self.comp = Composition(spcs_str=spcs_str)
    self.p_atm = None
    self.T_K = None
    self.elem_comp = None
    self.ci_paras_list = None

  # def set_pAtm(self, *, p_atm: float) -> None:
  #     assert p_atm > 0
  #     self.p_atm = p_atm

  # def set_T_K(self, *, T_K: float) -> None:
  #   assert T_K > 0
  #   self.T_K = T_K
  #   self.comp.T_K = T_K

  def set_elem_comp(self, **kwargs):
    if 'e' in kwargs:
      self.elem_comp = kwargs
    else:
      self.elem_comp = dict(kwargs, **dict(e=0))

  def set_lte_comp(self, *, p_atm, T_K):
    self.p_atm = p_atm
    self.T_K = T_K
    self.comp.set_lte_comp(p_atm=p_atm, T_K=T_K, elem_comp=self.elem_comp)

  def set_elements(self, elems: tuple):
    self.comp.set_elements(elems)

  def set_collisions(self, *, yaml_file: Path) -> NoReturn:
    colli_paras = ColliIntegralParas(yaml_file=yaml_file)
    self.set_reactions(rctn_list=colli_paras.reaction)
    self.set_ci_paras_list(colli_paras.collision)

  def set_reactions(self, *, rctn_list):
    self.rctn = []
    for rctn in rctn_list:
      if set(re.split(r"\s+(?:=>|\+)\s+", rctn)).issubset(self.comp.spcs_str):
        self.rctn.append(rctn)
      else:
        continue
    self.n_rctn = len(self.rctn)
    self.rctn_Rik = np.zeros((len(self.rctn), self.comp.n_spcs))
    for i, rctn_str in enumerate(self.rctn):
      rcnt_str, prdt_str = re.split(r"\s*=>\s*", rctn_str)
      rcnt_list = re.split(r"\s*\+\s*", rcnt_str.strip())
      prdt_list = re.split(r"\s*\+\s*", prdt_str.strip())
      for rcnt in rcnt_list:
        self.rctn_Rik[i, self.comp.spcs_str.index(rcnt)] = -1
      for prdt in prdt_list:
        self.rctn_Rik[i, self.comp.spcs_str.index(prdt)] = 1

  def set_ci_paras_list(self, ci_paras_list):
    self.ci_paras_list = []
    for ci_paras in ci_paras_list:
      if set(re.split(r"\s+", ci_paras[0])).issubset(self.comp.spcs_str):
        self.ci_paras_list.append(ci_paras)
    # self.ci_paras_list = ci_paras_list
    self._set_ci_engine()

  def get_ci_paras(self, spc1, spc2):
    for _para in self.ci_paras_list:
      if {spc1, spc2} == set(_para[0].split()):
        return _para
    return False

  def _set_ci_engine(self):
    CI_engine = []
    for i in range(self.comp.n_spcs):
      CI_engine.append([])
      for j in range(self.comp.n_spcs):
        ci_paras = self.get_ci_paras(self.comp.spcs_str[i],
                                     self.comp.spcs_str[j])
        if ci_paras is False:
          pairs_str = f"{self.comp.spcs_str[i]} {self.comp.spcs_str[j]}"
          CI_engine[i].append(EmptyColli(pairs_str=pairs_str, style="empty",
                                         coeff_str=""))
        else:
          match ci_paras[1]:
            case "SC":
              CI_engine[i].append(ScreenCoulombColli(pairs_str=ci_paras[0],
                                                     style="SC",
                                                     coeff_str=ci_paras[2]))
            case "CE":
              CI_engine[i].append(CECI(pairs_str=ci_paras[0],
                                       style="CE",
                                       coeff_str=ci_paras[2]))
            case "LJm6":
              CI_engine[i].append(LJm6ColliIntegral(pairs_str=ci_paras[0],
                                                    style="LJm6",
                                                    coeff_str=ci_paras[2]))
            case "EN":
              CI_engine[i].append(EleNeuColliIntegral(pairs_str=ci_paras[0],
                                                      style="EN",
                                                      coeff_str=ci_paras[2]))
            case _:
              raise Exception(ci_paras[1])
    self.ci_engine = CI_engine

  def mu_ij(self, iRow: int, jCol: int):
    return self.comp.absM[iRow]*self.comp.absM[jCol]/(
        self.comp.absM[iRow] + self.comp.absM[jCol])

  def Omega_ij(self, iRow: int, jCol: int, *, index: tuple) -> float:
    self.ci_engine[iRow][jCol].setReducedT(T_K=self.T_K)
    if isinstance(self.ci_engine[iRow][jCol], ScreenCoulombColli):
      # for screen coulomb interaction
      return self.ci_engine[iRow][jCol].getColliIntegral(index=index,
                                                         DebL=self.comp.totDebL())
    else:
      return self.ci_engine[iRow][jCol].getColliIntegral(index=index)

  def avg_CS_ij(self, iRow: int, jCol: int, *, index: tuple) -> float:
    self.ci_engine[iRow][jCol].setReducedT(T_K=self.T_K)
    if isinstance(self.ci_engine[iRow][jCol], ScreenCoulombColli):
      # for screen coulomb interaction
      return self.ci_engine[iRow][jCol].get_avg_CS(index=index,
                                                   DebL=self.comp.totDebL())
    else:
      return self.ci_engine[iRow][jCol].get_avg_CS(index=index)

  def Aij(self, i, j) -> float:
    r""" i: index of Row; j: index of Col """
    return 0.5*self.Omega_ij(i, j, index=(2, 2))/ \
      self.Omega_ij(i, j, index=(1, 1))

  def Bij(self, i, j) -> float:
    return 1/3*(5*self.Omega_ij(i, j, index=(1, 2)) - \
                self.Omega_ij(i, j, index=(1, 3)))/ \
      self.Omega_ij(i, j, index=(1, 1))

  def Cij(self, i, j):
    return 1/3*self.Omega_ij(i, j, index=(1, 2))/ \
      self.Omega_ij(i, j, index=(1, 1))

  # ------------------------------------------------------------------------- #
  # Cal viscosity
  # ------------------------------------------------------------------------- #
  def get_H_mtx(self, *, p, q, T_K):
    if p > q:
      return self.get_H_mtx(p=q, q=p, T_K=T_K).transpose()
    # Init
    A1 = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
    A2 = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
    H = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
    # ----------------------------------------------------------------------- #
    # Set A1, A2.
    for iRow in range(self.comp.n_spcs):
      for jCol in range(self.comp.n_spcs):
        M1 = self.comp.relM[iRow]/(self.comp.relM[iRow] + self.comp.relM[jCol])
        M2 = self.comp.relM[jCol]/(self.comp.relM[iRow] + self.comp.relM[jCol])
        self.ci_engine[iRow][jCol].setReducedT(T_K=T_K)
        tmp = 0
        for _c in M_coeff_1(p, q):
          omega = self.Omega_ij(iRow=iRow, jCol=jCol, index=_c[2])
          tmp += _c[0]*(M1**_c[1][0])*(M2**_c[1][1])*omega
        A1[iRow, jCol] = tmp*self.comp.xj[jCol]
        tmp = 0
        for _c in M_coeff_2(p, q):
          omega = self.Omega_ij(iRow=iRow, jCol=jCol, index=_c[2])
          tmp += _c[0]*(M1**_c[1][0])*(M2**_c[1][1])*omega
        A2[iRow, jCol] = tmp*self.comp.xj[jCol]
    # ----------------------------------------------------------------------- #
    # Set H matrix from A1, A2.
    for iRow in range(self.comp.n_spcs):
      for jCol in range(self.comp.n_spcs):
        H[iRow, jCol] = A2[iRow, jCol]
        if iRow == jCol:
          H[iRow, jCol] += np.sum(A1[iRow])
    H = H*2/5/kB/self.T_K
    return H

  def get_L_mtx(self, *, p, q, T_K):
    def get_c(_cc):
      if isinstance(_cc, tuple):
        return _cc[0]*_cc[1]
      else:
        return _cc

    if p > q:
      return self.get_L_mtx(p=q, q=p, T_K=T_K).transpose()
    # Init
    L = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
    N1 = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
    N2 = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
    # ----------------------------------------------------------------------- #
    # Set N1, N2
    for iRow in range(self.comp.n_spcs):
      for jCol in range(self.comp.n_spcs):
        M1 = self.comp.relM[iRow]/(self.comp.relM[iRow] + self.comp.relM[jCol])
        M2 = self.comp.relM[jCol]/(self.comp.relM[iRow] + self.comp.relM[jCol])
        self.ci_engine[iRow][jCol].setReducedT(T_K=T_K)
        tmp = 0
        for _c in N_coeff_1(p, q):
          omega = self.Omega_ij(iRow=iRow, jCol=jCol, index=_c[2])
          tmp += get_c(_c[0])*(M1**_c[1][0])*(M2**_c[1][1])*omega
        N1[iRow, jCol] = tmp*self.comp.xj[iRow]*self.comp.xj[jCol]
        # N1[iRow, jCol] = tmp*self.comp.xj[jCol]  # TODO
        tmp = 0
        for _c in N_coeff_2(p, q):
          omega = self.Omega_ij(iRow=iRow, jCol=jCol, index=_c[2])
          tmp += get_c(_c[0])*(M1**_c[1][0])*(M2**_c[1][1])*omega
        N2[iRow, jCol] = tmp*self.comp.xj[iRow]*self.comp.xj[jCol]
        # N2[iRow, jCol] = tmp*self.comp.xj[jCol]
    # Set L matrix from N1, N2
    for iRow in range(self.comp.n_spcs):
      for jCol in range(self.comp.n_spcs):
        L[iRow, jCol] = N2[iRow, jCol]
        if iRow == jCol:
          L[iRow, jCol] += np.sum(N1[iRow])
        L[iRow, jCol] = L[iRow, jCol]*sqrt(
          self.comp.absM[iRow]*self.comp.absM[jCol])
    L = L*8/75/kB**2/self.T_K
    return L

  def get_eta_order_2(self, *, T_K: float) -> float:
    H00 = self.get_H_mtx(p=0, q=0, T_K=T_K)
    H01 = self.get_H_mtx(p=0, q=1, T_K=T_K)
    H10 = self.get_H_mtx(p=1, q=0, T_K=T_K)
    H11 = self.get_H_mtx(p=1, q=1, T_K=T_K)
    H = np.vstack((np.hstack((H00, H01)),
                   np.hstack((H10, H11))))
    x = np.hstack((np.ones(self.comp.n_spcs), np.zeros(self.comp.n_spcs)))
    b = np.linalg.solve(H, x)
    return np.dot(self.comp.xj, b[:self.comp.n_spcs])

  def get_eta_order_1(self, *, T_K: float) -> float:
    # H x = b
    H00 = self.get_H_mtx(p=0, q=0, T_K=T_K)
    b10 = np.linalg.solve(H00, np.ones(self.comp.n_spcs))
    return np.dot(self.comp.xj, b10)

  # ------------------------------------------------------------------------- #
  def _mu_ij(self, i, j):
    return self.comp.absM[i]*self.comp.absM[j]/(
        self.comp.absM[i] + self.comp.absM[j])

  def _k_ij(self, i, j):
    return 75/64*kB**2*self.T_K/self._mu_ij(i, j)/self.Omega_ij(i, j,
                                                                index=(2, 2))

  # ------------------------------------------------------------------------- #
  #   Cal k_hp
  # ------------------------------------------------------------------------- #
  def _get_L_mtx(self):
    L = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
    # ----------------------------------------------------------------------- #
    # set Lii
    # ----------------------------------------------------------------------- #
    for i in range(self.comp.n_spcs):
      tmp = 0
      for k in range(self.comp.n_spcs):
        if k == i:
          continue
        mu_i = self.comp.absM[i]/(self.comp.absM[i] + self.comp.absM[k])
        mu_k = self.comp.absM[k]/(self.comp.absM[i] + self.comp.absM[k])
        tmp1 = np.dot(
          [15/2*mu_i**2 + 25/4*mu_k**2, -5*mu_k**2, mu_k**2, 2*mu_i*mu_k],
          [self.Omega_ij(i, k, index=(1, 1)),
           self.Omega_ij(i, k, index=(1, 2)),
           self.Omega_ij(i, k, index=(1, 3)),
           self.Omega_ij(i, k, index=(2, 2))])
        tmp = tmp + 256/75*self.comp.xj[i]*self.comp.xj[k]* \
              self._mu_ij(i, k)/kB**2/self.T_K*tmp1
      # L[i, i] = -256/75*self.comp.xj[i]**2*self._mu_ij(i,
      # i)/kB**2/self.T_K* \
      #           self.Omega_ij(i, i, index=(2, 2)) - tmp
      L[i, i] = -256/75*self.comp.xj[i]*self._mu_ij(i, i)/kB**2/self.T_K* \
                self.Omega_ij(i, i, index=(2, 2)) - tmp
    # ----------------------------------------------------------------------- #
    # set Lij
    # ----------------------------------------------------------------------- #
    for i in range(self.comp.n_spcs):
      for j in range(self.comp.n_spcs):
        mu_i = self.comp.absM[i]/(self.comp.absM[i] + self.comp.absM[j])
        mu_j = self.comp.absM[j]/(self.comp.absM[i] + self.comp.absM[j])
        tmp1 = np.dot([55/2, -10, 2, -4],
                      [self.Omega_ij(i, j, index=(1, 1)),
                       self.Omega_ij(i, j, index=(1, 2)),
                       self.Omega_ij(i, j, index=(1, 3)),
                       self.Omega_ij(i, j, index=(2, 2))])
        # L[i, j] = 128/75*self.comp.xj[i]*self.comp.xj[j]* \
        #           self._mu_ij(i, j)/kB**2/self.T_K*mu_i*mu_j*tmp1
        L[i, j] = 128/75*self.comp.xj[j]* \
                  self._mu_ij(i, j)/kB**2/self.T_K*mu_i*mu_j*tmp1
    return L

  # def get_k_hp_order_2_old(self):
  #   L = self._get_L_mtx()
  #   a = np.linalg.solve(L, np.ones_like(self.comp.xj))
  #   return -4*np.dot(self.comp.xj[1:], a[1:])

  def get_k_hp_order_2(self, *, T_K):
    L00 = self.get_L_mtx(p=0, q=0, T_K=T_K)
    L01 = self.get_L_mtx(p=0, q=1, T_K=T_K)
    L02 = self.get_L_mtx(p=0, q=2, T_K=T_K)
    L10 = self.get_L_mtx(p=1, q=0, T_K=T_K)
    L11 = self.get_L_mtx(p=1, q=1, T_K=T_K)
    L12 = self.get_L_mtx(p=1, q=2, T_K=T_K)
    L20 = self.get_L_mtx(p=2, q=0, T_K=T_K)
    L21 = self.get_L_mtx(p=2, q=1, T_K=T_K)
    L22 = self.get_L_mtx(p=2, q=2, T_K=T_K)
    L = np.vstack((np.hstack((L00, L01, L02)),
                   np.hstack((L10, L11, L12)),
                   np.hstack((L20, L21, L22))))
    # x = np.hstack(
    #   (np.zeros(self.comp.n_spcs), self.comp.xj, np.zeros(self.comp.n_spcs)))
    x = np.hstack((np.zeros(self.comp.n_spcs), self.comp.xj,
                   np.zeros(self.comp.n_spcs)))
    try:
      a = np.linalg.solve(L, x)
    except:
      diag = np.copy(L.diagonal())
      diag[diag == 0] = 1e-40*np.mean(diag[diag != 0])  # A trick.
      for _i, _vl in enumerate(diag):
        L[_i, _i] = _vl
      a = np.linalg.solve(L, x)
    a[a < 0] = 0  # A trick
    return np.dot(self.comp.xj[1:],
                  a[(self.comp.n_spcs + 1):(2*self.comp.n_spcs)])

  def get_k_hp_order_3(self, *, T_K):
    L00 = self.get_L_mtx(p=0, q=0, T_K=T_K)
    L01 = self.get_L_mtx(p=0, q=1, T_K=T_K)
    L02 = self.get_L_mtx(p=0, q=2, T_K=T_K)
    L03 = self.get_L_mtx(p=0, q=3, T_K=T_K)
    L10 = self.get_L_mtx(p=1, q=0, T_K=T_K)
    L11 = self.get_L_mtx(p=1, q=1, T_K=T_K)
    L12 = self.get_L_mtx(p=1, q=2, T_K=T_K)
    L13 = self.get_L_mtx(p=1, q=3, T_K=T_K)
    L20 = self.get_L_mtx(p=2, q=0, T_K=T_K)
    L21 = self.get_L_mtx(p=2, q=1, T_K=T_K)
    L22 = self.get_L_mtx(p=2, q=2, T_K=T_K)
    L23 = self.get_L_mtx(p=2, q=3, T_K=T_K)
    L30 = self.get_L_mtx(p=3, q=0, T_K=T_K)
    L31 = self.get_L_mtx(p=3, q=1, T_K=T_K)
    L32 = self.get_L_mtx(p=3, q=2, T_K=T_K)
    L33 = self.get_L_mtx(p=3, q=3, T_K=T_K)
    L = np.vstack((np.hstack((L00, L01, L02, L03)),
                   np.hstack((L10, L11, L12, L13)),
                   np.hstack((L20, L21, L22, L23)),
                   np.hstack((L30, L31, L32, L33))))
    x = np.hstack((np.zeros(self.comp.n_spcs), self.comp.xj,
                   np.zeros(self.comp.n_spcs), np.zeros(self.comp.n_spcs)))
    a = np.linalg.solve(L, x)
    return np.dot(self.comp.xj, a[self.comp.n_spcs:2*self.comp.n_spcs])

  # ------------------------------------------------------------------------- #
  #   Cal k_e
  # ------------------------------------------------------------------------- #
  def q_element(self):
    # q11
    q11 = 8*sqrt(2)*self.comp.nj[0]**2*self.avg_CS_ij(0, 0, index=(2, 2))
    for i in range(1, self.comp.n_spcs):
      avg_CS_seq = [self.avg_CS_ij(0, i, index=_idx)
                    for _idx in [(1, 1), (1, 2), (1, 3)]]
      tmp = np.dot([25/4, -15, 12], avg_CS_seq)
      q11 = q11 + 8*self.comp.nj[0]*self.comp.nj[i]*tmp
    # q12
    q12 = 8*sqrt(2)*self.comp.nj[0]**2*(
        7/4*self.avg_CS_ij(0, 0, index=(2, 2)) -
        2*self.avg_CS_ij(0, 0, index=(2, 3)))
    for i in range(1, self.comp.n_spcs):
      avg_CS_seq = [self.avg_CS_ij(0, i, index=_idx)
                    for _idx in [(1, 1), (1, 2), (1, 3), (1, 4)]]
      tmp = np.dot([175/16, -315/8, 57, -30], avg_CS_seq)
      q12 = q12 + 8*self.comp.nj[0]*self.comp.nj[i]*tmp
    # q22
    avg_CS_seq = [self.avg_CS_ij(0, 0, index=_idx)
                  for _idx in [(2, 2), (2, 3), (2, 4)]]
    q22 = 8*sqrt(2)*self.comp.nj[0]**2*np.dot([77/16, -7, 5], avg_CS_seq)
    for i in range(1, self.comp.n_spcs):
      avg_CS_seq = [self.avg_CS_ij(0, i, index=_idx)
                    for _idx in [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]]
      tmp = np.dot([1225/64, -735/8, 399/2, -210, 90], avg_CS_seq)
      q22 = q22 + 8*self.comp.nj[0]*self.comp.nj[i]*tmp
    return (q11, q12, q22)

  def get_k_e_order_3(self):
    q11, q12, q22 = self.q_element()
    return 75/8*self.comp.nj[0]**2*kB*sqrt(2*pi*kB*self.T_K/m_e)/(
        q11 - q12**2/q22)

  def get_k_e_order_4(self):
    pass

  def get_k_int_order(self):
    pass

  def get_k_rec_order(self):
    Aij = np.zeros((self.n_rctn, self.n_rctn))
    for i in range(self.n_rctn):
      for j in range(self.n_rctn):
        # set Aij
        tmp = 0
        for k in range(self.comp.n_spcs - 1):
          for l in range(k + 1, self.comp.n_spcs):
            with np.errstate(divide='ignore', invalid='ignore'):
              tmp1 = -self.rctn_Rik[i, k]*self.rctn_Rik[j, l] - \
                     self.rctn_Rik[i, l]*self.rctn_Rik[j, k] + \
                     self.rctn_Rik[i, k]*self.rctn_Rik[j, k]*self.comp.xj[l]/ \
                     self.comp.xj[k] + \
                     self.rctn_Rik[i, l]*self.rctn_Rik[j, l]*self.comp.xj[k]/ \
                     self.comp.xj[l]
              tmp = tmp + tmp1*self._mu_ij(k, l)*self.Omega_ij(k, l,
                                                               index=(1, 1))
        Aij[i, j] = tmp*16/(3*kB*self.T_K)
    dH = np.dot(self.rctn_Rik,
                [_spc.get_h(T_K=self.T_K) for _spc in self.comp.spcs])
    b = np.linalg.solve(Aij, dH)
    result = 1/kB/self.T_K**2*np.dot(dH, b)
    if np.isnan(result):
      return 0.0
    return result
