from math import pi, log, sqrt, log10, gamma
from math import log as ln
import numpy as np
import myconst as const
from myconst import eV2J, J2eV, K2J, J2K, eV2K, K2eV, relM2absM, e2, epsilon_0
from myconst import k as kB
from myspecie import spec_df
from scipy import constants as const
from scipy.constants import k as _kB
from scipy.constants import m_u as _m_u
from scipy.interpolate import interp1d

__all__ = ["ExpColli", "LJColli", "MorseColli", "ScreenCoulombColli"]

_e2 = 2.30708e-28


class Spec(object):

    def __init__(self, _str):
        self.relM = spec_df.loc[_str, "relM"]
        self.absM = self.relM * relM2absM


# =============================================================================================== #
class AbsColliIngrl(object):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        spc_strs = pairs_str.split()
        self.relM = [spec_df.loc[_str, "relM"] for _str in spc_strs]
        self.absM = [_ * relM2absM for _ in self.relM]
        self.sumM = sum(self.absM)
        self.rdcdM = self.absM[0] * self.absM[1] / \
                     (self.absM[0] + self.absM[1])
        self.coeff = [float(_) for _ in coeff_str.split()]
        self.style = style

    def setReducedT(self, *, T: float):
        self.rdcdT = T

    def getColliIntegral(self, *, index: tuple):
        pass


# =============================================================================================== #
class HSColli(AbsColliIngrl):
    r"""
    colli_coeff "sigma"
    """

    def __init__(self, *, pairs_str, style, coeff_str):
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)

    def getColliIntegral(self, *, index):
        result = pi * self.coeff[0] ** 2 * \
                 sqrt(kB * self.rdcdT / 2 / pi / self.rdcdM)
        if index == (1, 1):
            return result
        elif index == (1, 2):
            return 3 * result
        elif index == (2, 2):
            return 2 * result
        else:
            raise Exception("Index {} is error".format(index))


# =============================================================================================== #
class LJColli(AbsColliIngrl):

    def __init__(self, *, pairs_str, style, coeff_str):
        r"""
        colli_coeff "sigma[A]" "epsilon[K]"
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        self._setParas()

    def _setParas(self):
        self.tempValue = pi * self.coeff[0] ** 2 * \
                         sqrt(kB / 2 / pi / self.rdcdM) * 1e-20
        data = np.loadtxt("factor/LJ(12-6).txt", delimiter=",")
        # 11, 12, 13, 22, 23, 24
        self.interp_f11 = interp1d(data[:, 0], data[:, 1])
        self.interp_f12 = interp1d(data[:, 0], data[:, 2])
        self.interp_f22 = interp1d(data[:, 0], data[:, 4])

    def getColliIntegral(self, *, index):
        if index == (1, 1):
            interp_value = self.interp_f11(self.rdcdT / self.coeff[1])
            return 1 * self.tempValue * sqrt(self.rdcdT) * interp_value
        elif index == (1, 2):
            interp_value = self.interp_f12(self.rdcdT / self.coeff[1])
            return 3 * self.tempValue * sqrt(self.rdcdT) * interp_value
        elif index == (2, 2):
            interp_value = self.interp_f22(self.rdcdT / self.coeff[1])
            return 2 * self.tempValue * sqrt(self.rdcdT) * interp_value
        else:
            raise Exception("The index '{}' is error.".format(index))
        # print(f"{self.rdcdT/self.coeff[1]}")
        # print(f"{self.interp_f11(self.rdcdT / self.coeff[1])}")


# =============================================================================================== #


class ExpColli(AbsColliIngrl):

    def __init__(self, *, pairs_str, style, coeff_str):
        r"""
        \phi = A * \exp( - r / \rho)
        return ColliIntegral[m^3 s^-1]: 
        colli_coeff "A[eV]" "rho[A]"
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        self._setParas()

    def _setParas(self):
        data = np.loadtxt("factor/Exp-factor.txt", delimiter=",")
        self.interp_f11 = interp1d(data[:, 0], data[:, 1])
        self.interp_f12 = interp1d(data[:, 0], data[:, 2])
        self.interp_f22 = interp1d(data[:, 0], data[:, 3])

    def getColliIntegral(self, *, index):
        alpha = log(self.coeff[0] * eV2K / self.rdcdT)
        assert 3.5 <= alpha <= 28.5, "{}".format(alpha)
        self.tempValue = 1e-20 * 4 * sqrt(pi * kB * self.rdcdT / 2 / self.rdcdM) * \
                         alpha ** 2 * self.coeff[1] ** 2
        if index == (1, 1):
            return self.tempValue * self.interp_f11(alpha)
        elif index == (1, 2):
            return self.tempValue * self.interp_f12(alpha)
        elif index == (2, 2):
            return self.tempValue * self.interp_f22(alpha)
        else:
            raise Exception("The index '{}' is error.".format(index))


class ChargeExchange(AbsColliIngrl):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        r"""
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)

    def getColliIntegral(self, *, index: tuple):
        A, B = self.coeff
        l, s = index
        xi = sum(1 / n for n in range(1, s + 2)) - 0.5772
        alpha = pi ** 2 / 6 - sum(1 / n ** 2 for n in range(1, s + 2))
        temp = B ** 2 / 4 * ((ln(2 * kB * self.rdcdT * 1e4 / self.rdcdM) - 2 * A / B + xi)
                             ** 2 + alpha) * sqrt(kB * self.rdcdT / 2 / pi / self.rdcdM) * 1e-20
        if index == (1, 1):
            return temp * 1
        elif index == (1, 2):
            return temp * 3
        elif index == (2, 2):
            return temp * 2
        else:
            raise Exception("The index '{}' is error.".format(index))


class MorseColli(AbsColliIngrl):

    def __init__(self, *, data_dict):
        r"""
        \phi = De * {
        \exp(  - 2*(C / \sigma) * (r-r_e)  )
        -2 * \exp(  - (C / \sigma) * (r-r_e)  )
        }

        where:
        r_e / \sigma = 1 + log(2) / C

        unit:
        De:     eV
        r_e:    A
        C:      1

        """
        super().__init__(data_dict=data_dict)
        self._setParas()

    def _setParas(self):
        assert self.Poten == "Morse"
        if 4 < self.Paras["C"] <= 20:
            log_C = np.log10([4, 6, 8, 20])
            log_T = np.log10(
                [0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200])
            _pos_11 = (3, 15)
            _pos_22 = (43, 15)
        elif 2 <= self.Paras["C"] <= 4:
            log_C = np.log10([2, 4])
            log_T = np.log10([0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4])
            _pos_11 = (20, 10)
            _pos_22 = (60, 10)
        elif 1 <= self.Paras["C"] < 2:
            log_C = np.log10([1, 2])
            log_T = np.log10([0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4])
            _pos_11 = (32, 7)
            _pos_22 = (72, 7)
        else:
            raise Exception("The C {} is error".format(self.Paras["C"]))
        data_11 = np.loadtxt("factor/Morse-factor.txt", delimiter=",", skiprows=_pos_11[0],
                             max_rows=_pos_11[1])
        coef_11 = interp1d(log_C, data_11)(np.log10(self.Paras["C"]))
        self.f_11 = interp1d(log_T, coef_11)
        data_22 = np.loadtxt("factor/Morse-factor.txt", delimiter=",", skiprows=_pos_22[0],
                             max_rows=_pos_22[1])
        coef_22 = interp1d(log_C, data_22)(np.log10(self.Paras["C"]))
        self.f_22 = interp1d(log_T, coef_22)
        sigma = self.Paras["C"] * self.Paras["re"] / (self.Paras["C"] + log(2))
        print(sigma)
        self.tempValue = sqrt(_kB / 2 / pi / self.rdcdM) * \
                         pi * sigma ** 2 * 1e-20

    def getColliIntegral(self, *, index):
        De_T = self.Paras["De"] * _eV2K
        if index == (1, 1):
            return self.f_11(
                np.log10(self.rdcdT / De_T)) * self.tempValue * sqrt(self.rdcdT) * 1
        if index == (2, 2):
            return self.f_22(
                np.log10(self.rdcdT / De_T)) * self.tempValue * sqrt(self.rdcdT) * 2
        else:
            raise Exception("The index '{}' is error".format(index))


# =============================================================================================== #
class ScreenCoulombColli(AbsColliIngrl):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        r"""
        Ref
        ---
        PHYSICS OF PLASMAS 20, 093504(2013)
        Collision integrals for charged-charged interaction in two-temperature non-equilibrium plasma.
        http://dx.doi.org/10.1063/1.4821605
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)

    def getColliIntegral(self, *, index: tuple, DebL: float):
        l, s = index
        beta = 1.03
        Z1, Z2 = self.coeff
        b0 = 1 / 8 / pi / epsilon_0 * self.coeff[0] * Z1 * Z2 * const.e ** 2 / kB / self.rdcdT
        L = 2 * DebL / beta / b0
        Psi = 0 if (s == 1) else (sum(1 / n for n in range(1, s)))
        return l * sqrt(2 * pi * kB * self.rdcdT / self.rdcdM) * beta ** 2 * b0 ** 2 * gamma(s) * \
            (ln(L) - l / 2 - 2 * 0.57722 + Psi)
