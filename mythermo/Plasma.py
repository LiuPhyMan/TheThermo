import numpy as np
from myconst import k as kB
from .Comp import Composition
from .ColliIntegral import EmptyColli, ScreenCoulombColli, LJColli


class LTEPlasma(object):

    def __init__(self, *, spcs_str):
        self.comp = Composition(spcs_str=spcs_str)
        self.p_atm = None
        self.T_K = None

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
                    if ci_paras[1] == "SC":
                        CI_engine[i].append(ScreenCoulombColli(pairs_str=ci_paras[0],
                                                               style="SC",
                                                               coeff_str=ci_paras[2]))
                    elif ci_paras[1] == "LJ":
                        CI_engine[i].append(LJColli(pairs_str=ci_paras[0],
                                                    style="LJ",
                                                    coeff_str=ci_paras[2]))
                    else:
                        raise Exception("")
        self.ci_engine = CI_engine

    def get_Omega_ij(self, *, iRow:int, jCol:int, index:tuple) -> float:
        self.ci_engine[iRow][jCol].setReducedT(T_K=self.T_K)
        if isinstance(self.ci_engine[iRow][jCol], ScreenCoulombColli):
            return self.ci_engine[iRow][jCol].getColliIntegral(index=index, DebL=self.comp.DebL())
        else:
            return self.ci_engine[iRow][jCol].getColliIntegral(index=index)

    def get_H_mtx(self, *, p, q):
        # Init
        A1 = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
        A2 = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
        H = np.zeros((self.comp.n_spcs, self.comp.n_spcs))
        # --------------------------------------------------------------------------------------- #
        # Set A1, A2.
        for iRow in range(self.comp.n_spcs):
            for jCol in range(self.comp.n_spcs):
                M1 = self.comp.relM[iRow] / (self.comp.relM[iRow] +self.comp.relM[jCol])
                M2 = self.comp.relM[jCol] / (self.comp.relM[iRow] +self.comp.relM[jCol])
                self.ci_engine[iRow][jCol].setReducedT(T_K=self.T_K)
                tmp = 0
                for _c in coeff1(p, q):
                    omega = self.get_Omega_ij(iRow=iRow, jCol=jCol, index=_c[2])
                    tmp += _c[0] * (M1**_c[1][0]) * (M2**_c[1][1]) * omega
                A1[iRow, jCol] = tmp * self.comp.xj[iRow] * self.comp.xj[jCol]
                tmp = 0
                for _c in coeff2(p, q):
                    omega = self.get_Omega_ij(iRow=iRow, jCol=jCol, index=_c[2])
                    tmp += _c[0] * (M1**_c[1][0]) * (M2**_c[1][1]) * omega
                A2[iRow, jCol] = tmp * self.comp.xj[iRow] * self.comp.xj[jCol]
        # --------------------------------------------------------------------------------------- #
        # Set H matrix from A1, A2.
        for iRow in range(self.comp.n_spcs):
            for jCol in range(self.comp.n_spcs):
                H[iRow, jCol] = A2[iRow, jCol]
                if iRow == jCol:
                    H[iRow, jCol] += np.sum(A1[iRow])
        H = H * 2 / 5 / kB / self.T_K
        return H

    def get_eta(self, *, T_K:float) -> float:
        H00 = self.get_H_mtx(p=0, q=0)
        H01 = self.get_H_mtx(p=0, q=1)
        H11 = self.get_H_mtx(p=1, q=1)
        b20 = np.linalg.solve(H00 - np.dot(H01.dot(np.linalg.inv(H11)), H10), 2/kB/T*self.comp.xj)
        return 1 / 2 * kB * T * np.dot(self.comp.xj, b20)

def coeff1(p, q):
    if (p, q) == (0, 0):
        return [[80 / 3, (1, 1), (1, 1)],
                [8, (0, 2), (2, 2)]]
    elif (p, q) == (0, 1):
        return [[280 / 3, (1, 2), (1, 1)],
                [-112 / 3, (1, 2), (1, 2)],
                [28, (0, 3), (2, 2)],
                [-8, (0, 3), (2, 3)]]
    elif (p, q) == (1, 1):
        return [[560 / 3, (3, 1), (1, 1)],
                [980 / 3, (1, 3), (1, 1)],
                [-784 / 3, (1, 3), (1, 2)],
                [128 / 3, (1, 3), (1, 3)],
                [308 / 3, (2, 2), (2, 2)],
                [294 / 3, (0, 4), (2, 2)],
                [-56, (0, 4), (2, 3)],
                [8, (0, 4), (2, 4)],
                [16, (1, 3), (3, 3)]]
    else:
        raise Exception("")

def coeff2(p, q):
    if (p, q) == (0, 0):
        return [[-80 / 3, (1, 1), (1, 1)],
                [8, (1, 1), (2, 2)]]
    elif (p, q) == (0, 1):
        return [[-280 / 3, (2, 1), (1, 1)],
                [112 / 3, (2, 1), (1, 2)],
                [28, (2, 1), (2, 2)],
                [-8, (2, 1), (2, 3)]]
    elif (p, q) == (1, 1):
        return [[-1540 / 3, (2, 2), (1, 1)],
                [784 / 3, (2, 2), (1, 2)],
                [-128 / 3, (2, 2), (1, 3)],
                [602 / 3, (2, 2), (2, 2)],
                [-56, (2, 2), (2, 3)],
                [8, (2, 2), (2, 4)],
                [-16, (2, 2), (3, 3)]]
    else:
        raise Exception("")































