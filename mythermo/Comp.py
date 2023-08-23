# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 23.May.2023
"""

from math import sqrt, log, exp
from typing import List

import numpy as np
from myconst import (atm, light_c, K2J, K2Hz, h,
                     k as kB)
from myspecie import spc_dict

from mythermo.rad import PhoiCS, EmptyCS, bfCS_file_dict


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
    def avgZc(self):
        return ((self.Zc*self.nj)[self.Zc > 0]).sum()/(self.nj[self.Zc > 0]).sum()

    @property
    def avgZc2(self):
        return ((self.Zc**2*self.nj)[self.Zc > 0]).sum()/(self.nj[self.Zc > 0]).sum()

    def j_ff(self, *, T_K, wv_range):
        r"""
        16*alpha**3*sqrt(2*pi)*Ry/3/sqrt(3)/a0**3*a0**6 =
        4.877325664563308e-30
        4*alpha*(2*pi)**2.5/3/sqrt(3)/a0*a0**6 =
        2.306740679536327e-52
        """
        # gaunt_factor = 1  # TODO
        # return 4.877325664563308e-30*self.ne**2*self.avgZc* \
        #     sqrt(2*Ry/kB/T_K)*gaunt_factor*kB*T_K/h* \
        #     (exp(-h*light_c/kB/T_K/wv_range[1]) - exp(-h*light_c/kB/T_K/wv_range[0]))
        #
        # alpha_f = lambda nu: 2.306740679536327e-52*self.ne**2*self.avgZc*sqrt(2*Ry/kB/T_K)* \
        #                (2*Ry/h/nu)**3*(1 - exp(-h*nu/kB/T_K))
        T_J = T_K*K2J
        T_Hz = T_K*K2Hz
        nu_range = (light_c/wv_range[1], light_c/wv_range[0])
        # alpha_f = lambda nu: 1.45 * self.avgZc**2*self.ne**2/(2*pi*nu)**3/sqrt(T_K)
        # nu_seq = np.linspace(light_c/wv_range[1], light_c/wv_range[0], num=30)
        # jff_seq = [alpha_f(_nu)  * Bnu(_nu, T_K) for _nu in nu_seq]

        # return 64*sqrt(2)/3*pi**1.5*fsconst**4*Ry*a0**3*self.avgZc2/self.avgZc**2* \
        #     light_c/sqrt(3*kB*T_K/m_e)*self.ne**2*kB*T_K/h* \
        #     (exp(-h*nu_range[0]/T_J) - exp(-h*nu_range[1]/T_J))
        return 3.0530435e-30*(self.avgZc2/self.avgZc**2*self.ne**2)*sqrt(T_J)* \
            exp(-nu_range[0]/T_Hz)*(1 - exp(-(nu_range[1] - nu_range[0])/T_Hz))

    def j_bb(self, *, T_K, wv_range):
        return np.dot(self.nj, [_spc.norm_j_bb(T_K=T_K, wv_range=wv_range)
                                for _spc in self.spcs])

    def j_fb(self, *, T_K, wv_range):
        nu_range = (light_c/wv_range[1], light_c/wv_range[0])

    def kappa_bf(self, *, T_K, nu):
        kappa = 0
        for _i in range(self.n_spcs):
            kappa = kappa + self.bf_cs[_i].norm_cs(nu=nu, T_K=T_K)*self.nj[_i]/ \
                    self.spcs[_i].qint(T_K=T_K)
        return kappa*(1 - exp(-h*nu/kB/T_K))

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
            bk[:-1] = elem_bj - np.dot(self.Aij, Nj) + np.dot(self.Aij, Nj*rdcd_mu)
            bk[-1] = N - Nj.sum() + np.dot(Nj, rdcd_mu)
            sol = np.linalg.solve(Mkk, bk)
            dlnN = sol[-1]
            dlnNj = dlnN + np.dot(sol[:-1], self.Aij) - rdcd_mu
            for j in range(self.n_spcs):
                if (lnNj[j] - lnN) > -18.420680743952367:
                    factor[j] = 2/abs(dlnNj[j]) if (abs(dlnNj[j]) >= 2) else 1
                elif dlnNj[j] >= 0:
                    factor[j] = abs((lnNj[j] - lnN + 9.210340371976182)/(dlnNj[j] - dlnN))
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
