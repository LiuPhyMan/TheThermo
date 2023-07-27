import numpy as np
from myconst import k as kB
from mymath import coeff1, coeff2
from .Comp import Composition
from .ColliIntegral import (EmptyColli,
                            LJm6ColliIntegral,
                            CECI,
                            EleNeuColliIntegral,
                            ScreenCoulombColli)


class LTEPlasma(object):

    def __init__(self, *, spcs_str):
        self.comp = Composition(spcs_str=spcs_str)
        self.p_atm = None
        self.T_K = None
        self.ci_paras_list = None

    def set_pAtm(self, *, p_atm: float) -> None:
        assert p_atm > 0
        self.p_atm = p_atm

    def set_T_K(self, *, T_K: float) -> None:
        assert T_K > 0
        self.T_K = T_K
        self.comp.T_K = T_K

    def set_ci_paras_list(self, ci_paras_list):
        self.ci_paras_list = ci_paras_list
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
                ci_paras = self.get_ci_paras(self.comp.spcs_str[i], self.comp.spcs_str[j])
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
            return self.ci_engine[iRow][jCol].getColliIntegral(index=index, DebL=self.comp.DebL())
        else:
            return self.ci_engine[iRow][jCol].getColliIntegral(index=index)

    def Aij(self, i, j) -> float:
        r""" i: index of Row; j: index of Col """
        return 0.5*self.Omega_ij(i, j, index=(2, 2))/self.Omega_ij(i, j, index=(1, 1))

    def Bij(self, i, j) -> float:
        return 1/3*(5*self.Omega_ij(i, j, index=(1, 2)) - self.Omega_ij(i, j, index=(1, 3)))/ \
            self.Omega_ij(i, j, index=(1, 1))

    def Cij(self, i, j):
        return 1/3*self.Omega_ij(i, j, index=(1, 2))/self.Omega_ij(i, j, index=(1, 1))

    # ------------------------------------------------------------------------------------------- #
    # Cal viscosity
    # ------------------------------------------------------------------------------------------- #
    def get_H_mtx(self, *, p, q):
        # Init
        A1 = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
        A2 = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
        H = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
        # --------------------------------------------------------------------------------------- #
        # Set A1, A2.
        for iRow in range(self.comp.n_spcs):
            for jCol in range(self.comp.n_spcs):
                M1 = self.comp.relM[iRow]/(self.comp.relM[iRow] + self.comp.relM[jCol])
                M2 = self.comp.relM[jCol]/(self.comp.relM[iRow] + self.comp.relM[jCol])
                self.ci_engine[iRow][jCol].setReducedT(T_K=self.T_K)
                tmp = 0
                for _c in coeff1(p, q):
                    omega = self.Omega_ij(iRow=iRow, jCol=jCol, index=_c[2])
                    tmp += _c[0]*(M1**_c[1][0])*(M2**_c[1][1])*omega
                A1[iRow, jCol] = tmp*self.comp.xj[iRow]*self.comp.xj[jCol]
                tmp = 0
                for _c in coeff2(p, q):
                    omega = self.Omega_ij(iRow=iRow, jCol=jCol, index=_c[2])
                    tmp += _c[0]*(M1**_c[1][0])*(M2**_c[1][1])*omega
                A2[iRow, jCol] = tmp*self.comp.xj[iRow]*self.comp.xj[jCol]
        # --------------------------------------------------------------------------------------- #
        # Set H matrix from A1, A2.
        for iRow in range(self.comp.n_spcs):
            for jCol in range(self.comp.n_spcs):
                H[iRow, jCol] = A2[iRow, jCol]
                if iRow == jCol:
                    H[iRow, jCol] += np.sum(A1[iRow])
        H = H*2/5/kB/self.T_K
        return H

    def get_eta_order_2(self, *, T_K: float) -> float:
        H00 = self.get_H_mtx(p=0, q=0)
        H01 = self.get_H_mtx(p=0, q=1)
        H11 = self.get_H_mtx(p=1, q=1)
        b20 = np.linalg.solve(H00 - np.dot(H01.dot(np.linalg.inv(H11)), H10), 2/kB/T*self.comp.xj)
        return 1/2*kB*T*np.dot(self.comp.xj, b20)

    def get_eta_order_1(self, *, T_K: float) -> float:
        H00 = self.get_H_mtx(p=0, q=0)
        b10 = np.linalg.solve(H00, 2/kB/T_K*self.comp.xj)
        return 1/2*kB*T_K*np.dot(self.comp.xj, b10)

    # ------------------------------------------------------------------------------------------- #
    def _mu_ij(self, i, j):
        return self.comp.absM[i]*self.comp.absM[j]/(self.comp.absM[i] + self.comp.absM[j])

    def _k_ij(self, i, j):
        return 75/64*kB**2*self.T_K/self._mu_ij(i, j)/self.Omega_ij(i, j, index=(2, 2))

    def get_L_mtx(self):
        L = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
        for i in range(self.comp.n_spcs):
            tmp = 0
            for k in range(self.comp.n_spcs):
                if k == iRow:
                    continue
                mu_i = self.comp.absM[i]/(self.comp.absM[i] + self.comp.absM[k])
                mu_k = self.comp.absM[k]/(self.comp.absM[i] + self.comp.absM[k])
                tmp1 = np.dot([15/2, 25/4, -3, 4],
                              [mu_i**2, mu_k**2,
                               mu_k**2*self.Bij(i, k), mu_i*mu_k*self.Aij(i, k)])
                tmp = tmp + 2*self.comp.xj[k]/self._k_ij(i, k)*tmp1/self.Aij(i, k)
            L[i, i] = -4*self.comp.xj[i]/self._k_ij(i, i) - tmp

    def get_k_hp(self):
        pass
