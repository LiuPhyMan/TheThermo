# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 23.May.2023
"""

from math import sqrt, log, exp
from typing import List

import numpy as np
from myconst import (atm, light_c, K2J, K2Hz, Hz2K, h, nm2m, nm2Hz_f,
                     k as kB, nm2Hz_f)
from myspecie import spc_dict
from mythermo.basic import avgBnu, WavelengthRange as WVRng
from mythermo.rad import PhoiCS, EmptyCS, bfCS_file_dict

from .basic import Bnu, integral_Bnu


class AbsComp(object):
    __slots__ = ("spcs_str", "spcs",
                 "n_spcs", "n_elem",
                 "Zc",
                 "relM", "absM",
                 "xj", "nj",
                 "Aij", "elems")

    def __init__(self, *, spcs_str: List[str]) -> None:
        self.spcs_str = spcs_str
        self.n_spcs = len(spcs_str)
        self.spcs = [spc_dict[_] for _ in self.spcs_str]
        self._set_elements_by_species()
        self.Zc = np.array([_.Zc for _ in self.spcs])
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
    __slots__ = ("p_atm", "T_K", "bf_cs",)

    def __init__(self, *, spcs_str: List[str]) -> None:
        super().__init__(spcs_str=spcs_str)
        self.T_K = 0
        self.p_atm = 1
        self._set_PIcs()

    def _set_PIcs(self):
        self.bf_cs = []
        for _spc in self.spcs_str:
            if _spc in bfCS_file_dict.keys():
                self.bf_cs.append(PhoiCS(df_file=bfCS_file_dict[_spc]))
            else:
                self.bf_cs.append(EmptyCS())

    def rdcd_mu0(self, *, T_K: float):
        return np.array([_.rdcd_mu0(T_K=T_K) for _ in self.spcs])

    @property
    def ne(self):
        return self.nj[self.spcs_str.index("e")]

    @property
    def nion(self):
        return (self.nj[self.Zc > 0]).sum()

    @property
    def Zeff(self):
        return (self.Zc**2*self.nj)[self.Zc >0].sum() / (self.Zc*self.nj)[self.Zc >0].sum()


    # @property
    # def avgZc(self):
    #     return ((self.Zc*self.nj)[self.Zc > 0]).sum()/ self.nion
    #
    # @property
    # def avgZc2(self):
    #     return ((self.Zc**2*self.nj)[self.Zc > 0]).sum()/ self.nion

    # ------------------------------------------------------------------------------------------- #
    # free-free transition
    # ------------------------------------------------------------------------------------------- #
    def j_ff_Hz(self, *, nuHz: float, T_K: float) -> float:
        r""" """
        return 5.4443668e-52/sqrt(T_K)*(self.Zeff*self.ne**2)*exp(-nuHz*Hz2K/T_K)

    def avg_j_ff_Hz(self, *, wv_rng: WVRng, T_K: float) -> float:
        r""" """
        return 1.1344220e-41*sqrt(T_K)*(self.Zeff*self.ne**2) * exp(-wv_rng.TempK[0]/T_K)*\
            (1 - exp(-wv_rng.D_TempK/T_K)) / wv_rng.D_nuHz

    def kappa_ff(self, *, nuHz: float, T_K: float) -> float:
        return self.j_ff_Hz(nuHz=nuHz, T_K=T_K)/Bnu(nuHz=nuHz, T_K=T_K)

    def avg_kappa_ff(self, *, wv_rng: WVRng, T_K: float) -> float:
        return self.avg_j_ff_Hz(wv_rng=wv_rng, T_K=T_K) / avgBnu(nuHz_rng=wv_rng.nuHz, T_K=T_K)


    # ------------------------------------------------------------------------------------------- #
    # free-bound transition
    # ------------------------------------------------------------------------------------------- #
    def j_fb_Hz(self, *, nuHz: float, T_K: float) -> float:
        return self.kappa_bf(nuHz=nuHz, T_K=T_K)*Bnu(nuHz=nuHz, T_K=T_K)

    def avg_j_fb_Hz(self, *, wv_rng: WVRng, T_K: float) -> float:
        j_seq = [self.j_fb_Hz(nuHz=_nu, T_K=T_K)
                 for _nu in np.linspace(wv_rng.nuHz[0], wv_rng.nuHz[1], num=100)]
        return np.mean(j_seq)

    def kappa_bf(self, *, nuHz: float, T_K: float) -> float:
        kappa = 0
        for _i in range(self.n_spcs):
            kappa = kappa + self.bf_cs[_i].norm_cs(nuHz=nuHz, T_K=T_K)*self.nj[_i] / \
                self.spcs[_i].qint(T_K=T_K)
        return kappa*(1 - exp(-nuHz*Hz2K/T_K))

    # ------------------------------------------------------------------------------------------- #
    # bound-bound transition
    # ------------------------------------------------------------------------------------------- #
    def avg_j_bb_Hz(self, *, wv_rng: WVRng, T_K: float) -> float:
        return np.dot(self.nj, [_spc.norm_j_bb(T_K=T_K, wvlnm_rng=wv_rng.wvlnm)
                                for _spc in self.spcs])/wv_rng.D_nuHz

    def avg_kappa_bb(self, *, wv_rng: WVRng, T_K: float):
        return self.avg_j_bb_Hz(T_K=T_K, wv_rng=wv_rng)* wv_rng.D_nuHz / \
            integral_Bnu(T_K=T_K, nuHz_rng=wv_rng.nuHz)

    # ------------------------------------------------------------------------------------------- #
    def avg_j_Hz(self, *, wv_rng: WVRng, T_K: float) -> float:
        return self.avg_j_ff_Hz(wv_rng=wv_rng, T_K=T_K) + \
            self.avg_j_fb_Hz(wv_rng=wv_rng, T_K=T_K) + \
            self.avg_j_bb_Hz(wv_rng=wv_rng,T_K=T_K)

    def avg_kappa(self, *, wv_rng: WVRng, T_K: float) -> float:
        return self.avg_j_Hz(T_K=T_K, wv_rng=wv_rng) / avgBnu(nuHz_rng=wv_rng.nuHz, T_K=T_K)

    # ------------------------------------------------------------------------------------------- #
    def get_rho(self):
        return np.dot(self.nj, self.absM)

    def get_H(self, *, T_K: float):
        return np.dot(self.nj, [_spc.get_h(T_K=T_K) for _spc in self.spcs])/self.get_rho()

    def set_lte_comp(self, *, p_atm, T_K, elem_comp) -> None:
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
        # factor
        factor = np.ones(self.n_spcs)
        rdcd_mu0 = self.rdcd_mu0(T_K=T_K)
        for _i in range(500):
            rdcd_mu = rdcd_mu0 + log(p_atm) + np.log(Nj) - log(Nj.sum())
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
        V = Nj.sum()*kB*T_K/(p_atm*atm)
        self.xj = Nj/Nj.sum()
        self.nj = self.xj*p_atm*atm/(kB*T_K)

    def DebL(self):
        temp = sum(_spc.Zc**2*self.nj[_j] for _j, _spc in enumerate(self.spcs))
        return 1/sqrt(0.00020998524287342308/self.T_K*temp)
