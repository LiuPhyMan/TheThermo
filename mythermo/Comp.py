# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 23.May.2023
"""

from math import sqrt, log, exp
import numpy as np
from typing import List
from myconst import atm, k
from myspecie import spc_dict


class AbsComp(object):

    def __init__(self, *, spcs_str: List[str]) -> None:
        self.spcs_str = spcs_str
        self.n_spcs = len(spcs_str)
        self.spcs = [spc_dict[_] for _ in self.spcs_str]
        self._set_elements_by_species()
        self.Zc = [_.Zc for _ in self.spcs]
        self.relM = [_.relM for _ in self.spcs]
        self.absM = [_.absM for _ in self.spcs]
        # self.Nj = np.zeros(self.n_spcs)
        # self.lnNj = np.zeros(self.n_spcs)
        self.xj = np.zeros(self.n_spcs)
        self.nj = np.zeros(self.n_spcs)

    def _set_Aij(self):
        self.Aij = np.zeros((self.n_elem, self.n_spcs))
        for _j in range(self.n_spcs):
            for _i in range(self.n_elem):
                self.Aij[_i, _j] = self.spcs[_j].get_nElem(self.elems[_i])

    def _set_elements_by_species(self):
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


class Composition(AbsComp):

    def __init__(self, *, spcs_str: List[str]) -> None:
        super().__init__(spcs_str=spcs_str)
        self.T_K = 0
        self.p_atm = 1

    def rdcd_mu0(self, *, T_K: float):
        return np.array([_.rdcd_mu0(T_K=T_K) for _ in self.spcs])

    # def set_Nj(self, *, Nj):
    #     self.Nj = Nj
    #     self.N = self.Nj.sum()

    def set_lte_comp(self, *, p_atm, T_K, elem_comp) -> None:
        elem_bj = tuple(elem_comp[_spc] for _spc in self.elems)
        self.T_K = T_K
        self.p_atm = p_atm
        Nj = np.ones(self.n_spcs) / self.n_spcs
        N = 1
        lnN = N
        lnNj = np.log(Nj)
        # Solve Mkk \dot xk = bk    {k = self.n_elem + 1}
        Mkk = np.zeros((self.n_elem + 1, self.n_elem + 1))
        bk = np.zeros(self.n_elem + 1)
        # factor
        factor = np.ones(self.n_spcs)
        rdcd_mu0 = self.rdcd_mu0(T_K=T_K)
        for _i in range(500):
            rdcd_mu = rdcd_mu0 + log(p_atm) + np.log(Nj) - log(Nj.sum())
            Mkk[:-1, :-1] = np.dot(self.Aij, (self.Aij * Nj).transpose())
            Mkk[:-1, -1] = np.dot(self.Aij, Nj)
            Mkk[-1, :-1] = np.dot(self.Aij, Nj)
            Mkk[-1, -1] = Nj.sum() - N
            bk[:-1] = elem_bj - np.dot(self.Aij, Nj) + np.dot(self.Aij, Nj * rdcd_mu)
            bk[-1] = N - Nj.sum() + np.dot(Nj, rdcd_mu)
            sol = np.linalg.solve(Mkk, bk)
            dlnN = sol[-1]
            dlnNj = dlnN + np.dot(sol[:-1], self.Aij) - rdcd_mu
            for j in range(self.n_spcs):
                if (lnNj[j] - lnN) > -18.420680743952367:
                    factor[j] = 2 / abs(dlnNj[j]) if (abs(dlnNj[j]) >= 2) else 1
                elif dlnNj[j] >= 0:
                    factor[j] = abs((lnNj[j] - lnN + 9.210340371976182) / (dlnNj[j] - dlnN))
                else:
                    factor[j] = (2 / (5 * abs(dlnN))) if (abs(dlnN) > 2.5) else 1
            e_factor = min(1, np.min(factor))
            lnN = lnN + e_factor * dlnN
            lnNj = lnNj + e_factor * dlnNj
            N = exp(lnN)
            Nj = np.exp(lnNj)
            if np.all(Nj * np.abs(dlnNj) / Nj.sum() <= 1e-25):
                break
        V = Nj.sum() * k * T_K / (p_atm * atm)
        self.xj = Nj / Nj.sum()
        self.nj = self.xj * p_atm * atm / (k * T_K)

    def DebL(self):
        temp = sum(_spc.Zc ** 2 * self.nj[_j] for _j, _spc in enumerate(self.spcs))
        return 1 / sqrt(0.00020998524287342308 / self.T_K * temp)
