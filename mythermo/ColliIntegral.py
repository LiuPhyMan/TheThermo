from math import pi, log, sqrt, log10, gamma, exp
from math import log as ln
from math import factorial, isclose
import numpy as np
from mymath import integrate_by_DCT, ls_index
import myconst as const
from myconst import eV2J, J2eV, K2J, J2K, eV2K, K2eV, relM2absM, e2, epsilon_0
from myconst import k as kB
from myspecie import spec_df
from scipy import constants as const
from scipy.constants import k as _kB
from scipy.constants import m_u as _m_u
from scipy.interpolate import interp1d

__all__ = ["ExpColli", "LJColli", "ScreenCoulombColli"]

_e2 = 2.30708e-28

# import os
# print(os.path.abspath(__path__))
from mythermo import path


class Spec(object):

    def __init__(self, _str):
        self.relM = spec_df.loc[_str, "relM"]
        self.absM = self.relM*relM2absM


# =============================================================================================== #
class AbsColliIngrl(object):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        self.spcs = pairs_str.split()
        self.relM = [spec_df.loc[_str, "relM"] for _str in self.spcs]
        self.absM = [_*relM2absM for _ in self.relM]
        self.sumM = sum(self.absM)
        self.rdcdM = self.absM[0]*self.absM[1]/ \
                     (self.absM[0] + self.absM[1])
        # self.coeff = [float(_) for _ in coeff_str.split()]
        self.coeff_str = coeff_str
        self.style = style

    def setReducedT(self, *, T_K: float):
        self.rdcdT = T_K

    def get_avg_CS(self, *, index: tuple):
        return self.getColliIntegral(index=index)/sqrt(kB*self.rdcdT/2/pi/self.rdcdM)/ \
            ls_index(index[0], index[1])

    def getColliIntegral(self, *, index: tuple):
        pass


class EmptyColli(AbsColliIngrl):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)

    def getColliIntegral(self, *, index: tuple):
        return 0


# =============================================================================================== #
class HSColli(AbsColliIngrl):
    r"""
    colli_coeff "sigma"
    """

    def __init__(self, *, pairs_str, style, coeff_str):
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        self.coeff = [float(_) for _ in coeff_str.split()]

    def getColliIntegral(self, *, index):
        result = pi*self.coeff[0]**2* \
                 sqrt(kB*self.rdcdT/2/pi/self.rdcdM)
        if index == (1, 1):
            return result
        elif index == (1, 2):
            return 3*result
        elif index == (2, 2):
            return 2*result
        else:
            raise Exception("Index {} is error".format(index))


# =============================================================================================== #
class LJColli(AbsColliIngrl):

    def __init__(self, *, pairs_str, style, coeff_str):
        r"""
        colli_coeff "sigma[A]" "epsilon[K]"
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        self.coeff = [float(_) for _ in coeff_str.split()]
        self._setParas()

    def _setParas(self):
        self.tempValue = pi*self.coeff[0]**2* \
                         sqrt(kB/2/pi/self.rdcdM)*1e-20
        data = np.loadtxt("factor/LJ(12-6).txt", delimiter=",")
        # 11, 12, 13, 22, 23, 24
        self.interp_f11 = interp1d(data[:, 0], data[:, 1])
        self.interp_f12 = interp1d(data[:, 0], data[:, 2])
        self.interp_f13 = interp1d(data[:, 0], data[:, 3])
        self.interp_f22 = interp1d(data[:, 0], data[:, 4])
        self.interp_f23 = interp1d(data[:, 0], data[:, 5])
        self.interp_f24 = interp1d(data[:, 0], data[:, 6])
        self.interp_f25 = interp1d(data[:, 0], data[:, 7])

    def getColliIntegral(self, *, index):
        coeff = 0
        if index == (1, 1):
            interp_value = self.interp_f11(self.rdcdT/self.coeff[1])
            coeff = 1
        elif index == (1, 2):
            interp_value = self.interp_f12(self.rdcdT/self.coeff[1])
            coeff = 3
        elif index == (1, 3):
            interp_value = self.interp_f13(self.rdcdT/self.coeff[1])
            coeff = 12
        elif index == (2, 2):
            interp_value = self.interp_f22(self.rdcdT/self.coeff[1])
            coeff = 2
        elif index == (2, 3):
            interp_value = self.interp_f23(self.rdcdT/self.coeff[1])
            coeff = 8
        elif index == (2, 4):
            interp_value = self.interp_f24(self.rdcdT/self.coeff[1])
            coeff = 40
        else:
            raise Exception("The index '{}' is error.".format(index))
        return coeff*self.tempValue*sqrt(self.rdcdT)*interp_value


class AbsImpovedLJColli(AbsColliIngrl):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        r"""
        Parameters
        ----------
        pairs_str: str

        style: str

        coeff_str: str
            beta[1], epsilon[meV], r_e[A]
        Note
        ----
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        self.coeff = [float(_) for _ in coeff_str.split()]
        self._path = None
        self._coef = [None, None]
        # self._set_paras()

    def _set_paras(self):
        self.c_mtx_dict = dict()
        for _index in ("1-1", "1-2", "1-3", "1-4", "1-5", "2-2", "2-3", "2-4", "3-3", "4-4"):
            self.c_mtx_dict[_index] = np.zeros((7, 3))
            self.c_mtx_dict[_index][:4, :2] = np.loadtxt(self._path + f"/{_index}.txt",
                                                         delimiter=",", skiprows=0, max_rows=4)
            self.c_mtx_dict[_index][4:, :3] = np.loadtxt(self._path + f"/{_index}.txt",
                                                         delimiter=",", skiprows=4, max_rows=3)

    def get_rdcd_ci(self, *, index: tuple):
        c_mtx = self.c_mtx_dict[f"{index[0]}-{index[1]}"]
        a = np.dot(c_mtx, self.coeff[0]**np.arange(3))
        x = ln(self.rdcdT*K2eV*1e3/self.coeff[1])
        tmp1 = exp((x - a[2])/a[3])
        tmp2 = exp((x - a[5])/a[6])
        ln_rdcd_CI = (a[0] + a[1]*x)*tmp1**2/(tmp1**2 + 1) + \
                     a[4]*tmp2**2/(tmp2**2 + 1)
        return exp(ln_rdcd_CI)

    def getColliIntegral(self, *, index: tuple):
        r"""
        Return collision integral []
        """
        l, s = index[0], index[1]
        factor = factorial(s + 1)*(2*l + 1 - (-1)**l)/4/(l + 1)
        # sigma = 0.8002 * self.coeff[0] ** 0.049256 * self.coeff[2]
        sigma = self._coef[0]*self.coeff[0]**self._coef[1]*self.coeff[2]
        convert_factor = factor*pi*sigma**2*sqrt(kB*self.rdcdT/2/pi/self.rdcdM)
        return self.get_rdcd_ci(index=index)*convert_factor*1e-20


class LJm6ColliIntegral(AbsImpovedLJColli):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        r"""
        Parameters
        ----------
        pairs_str: str

        style: str

        coeff_str: str
            beta[1], epsilon[meV], r_e[A]
        Note
        ----
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        self._path = path + f"/factor/LJm6"
        self._coef = [0.8002, 0.049256]
        super()._set_paras()


class LJm4ColliIntegral(AbsImpovedLJColli):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        self._path = path + f"/factor/LJm4"
        self._coef = [0.7564, 0.064605]
        super()._set_paras()


# =============================================================================================== #


class ExpColli(AbsColliIngrl):

    def __init__(self, *, pairs_str, style, coeff_str):
        r"""
        \phi = A * \exp( - r / \rho)
        return ColliIntegral[m^3 s^-1]: 
        colli_coeff "A[eV]" "rho[A]"
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        self.coeff = [float(_) for _ in coeff_str.split()]
        self._setParas()

    def _setParas(self):
        data = np.loadtxt("factor/Exp-factor.txt", delimiter=",")
        self.interp_f11 = interp1d(data[:, 0], data[:, 1])
        self.interp_f12 = interp1d(data[:, 0], data[:, 2])
        self.interp_f22 = interp1d(data[:, 0], data[:, 3])

    def getColliIntegral(self, *, index):
        alpha = log(self.coeff[0]*eV2K/self.rdcdT)
        assert 3.5 <= alpha <= 28.5, "{}".format(alpha)
        self.tempValue = 1e-20*4*sqrt(pi*kB*self.rdcdT/2/self.rdcdM)* \
                         alpha**2*self.coeff[1]**2
        if index == (1, 1):
            return self.tempValue*self.interp_f11(alpha)
        elif index == (1, 2):
            return self.tempValue*self.interp_f12(alpha)
        elif index == (2, 2):
            return self.tempValue*self.interp_f22(alpha)
        else:
            raise Exception("The index '{}' is error.".format(index))


class CEinCI(AbsColliIngrl):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        r"""
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        self.coeff = [float(_) for _ in coeff_str.split()]

    def getColliIntegral(self, *, index: tuple):
        A, B = self.coeff
        l, s = index
        xi = sum(1/n for n in range(1, s + 2)) - 0.5772
        alpha = pi**2/6 - sum(1/n**2 for n in range(1, s + 2))
        temp = B**2/4*((ln(2*kB*self.rdcdT*1e4/self.rdcdM) - 2*A/B + xi)
                       **2 + alpha)*sqrt(kB*self.rdcdT/2/pi/self.rdcdM)*1e-20
        if index == (1, 1):
            return temp*1
        elif index == (1, 2):
            return temp*3
        elif index == (2, 2):
            return 0
        else:
            return 0
            # raise Exception("The index '{}' is error.".format(index))


# ----------------------------------------------------------------------------------------------- #
class CECI(AbsColliIngrl):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str) -> None:
        r"""

        Parameters
        ----------
        pairs_str
        style
        coeff_str
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        coeff_el, coeff_in = coeff_str.split(",")
        self.CEel = LJm4ColliIntegral(pairs_str=pairs_str, style="",
                                      coeff_str=coeff_el.strip())
        self.CEin = CEinCI(pairs_str=pairs_str, style="",
                           coeff_str=coeff_in.strip())

    def setReducedT(self, *, T_K: float):
        super().setReducedT(T_K=T_K)
        self.CEel.setReducedT(T_K=T_K)
        self.CEin.setReducedT(T_K=T_K)

    def getColliIntegral(self, *, index: tuple):
        if isclose(index[0], 1):
            CI0 = self.CEel.getColliIntegral(index=index)
            CI1 = self.CEin.getColliIntegral(index=index)
            return sqrt(CI0**2 + CI1**2)
        else:
            return self.CEel.getColliIntegral(index=index)


# class MorseColli(AbsColliIngrl):

#     def __init__(self, *, data_dict):
#         r"""
#         \phi = De * {
#         \exp(  - 2*(C / \sigma) * (r-r_e)  )
#         -2 * \exp(  - (C / \sigma) * (r-r_e)  )
#         }

#         where:
#         r_e / \sigma = 1 + log(2) / C

#         unit:
#         De:     eV
#         r_e:    A
#         C:      1

#         """
#         super().__init__(data_dict=data_dict)
#         self._setParas()

#     def _setParas(self):
#         assert self.Poten == "Morse"
#         if 4 < self.Paras["C"] <= 20:
#             log_C = np.log10([4, 6, 8, 20])
#             log_T = np.log10(
#                 [0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4, 10, 20, 40, 100, 200])
#             _pos_11 = (3, 15)
#             _pos_22 = (43, 15)
#         elif 2 <= self.Paras["C"] <= 4:
#             log_C = np.log10([2, 4])
#             log_T = np.log10([0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4, 1, 2, 4])
#             _pos_11 = (20, 10)
#             _pos_22 = (60, 10)
#         elif 1 <= self.Paras["C"] < 2:
#             log_C = np.log10([1, 2])
#             log_T = np.log10([0.004, 0.01, 0.02, 0.04, 0.1, 0.2, 0.4])
#             _pos_11 = (32, 7)
#             _pos_22 = (72, 7)
#         else:
#             raise Exception("The C {} is error".format(self.Paras["C"]))
#         data_11 = np.loadtxt("factor/Morse-factor.txt", delimiter=",", skiprows=_pos_11[0],
#                              max_rows=_pos_11[1])
#         coef_11 = interp1d(log_C, data_11)(np.log10(self.Paras["C"]))
#         self.f_11 = interp1d(log_T, coef_11)
#         data_22 = np.loadtxt("factor/Morse-factor.txt", delimiter=",", skiprows=_pos_22[0],
#                              max_rows=_pos_22[1])
#         coef_22 = interp1d(log_C, data_22)(np.log10(self.Paras["C"]))
#         self.f_22 = interp1d(log_T, coef_22)
#         sigma = self.Paras["C"] * self.Paras["re"] / (self.Paras["C"] + log(2))
#         print(sigma)
#         self.tempValue = sqrt(_kB / 2 / pi / self.rdcdM) * \
#                          pi * sigma ** 2 * 1e-20

#     def getColliIntegral(self, *, index):
#         De_T = self.Paras["De"] * _eV2K
#         if index == (1, 1):
#             return self.f_11(
#                 np.log10(self.rdcdT / De_T)) * self.tempValue * sqrt(self.rdcdT) * 1
#         if index == (2, 2):
#             return self.f_22(
#                 np.log10(self.rdcdT / De_T)) * self.tempValue * sqrt(self.rdcdT) * 2
#         else:
#             raise Exception("The index '{}' is error".format(index))


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
        self.coeff = [float(_) for _ in coeff_str.split()]

    def get_avg_CS(self, *, index: tuple, DebL: float):
        return self.getColliIntegral(index=index, DebL=DebL)/ \
            sqrt(kB*self.rdcdT/2/pi/self.rdcdM)/ls_index(index[0], index[1])

    def getColliIntegral(self, *, index: tuple, DebL: float):
        l, s = index
        beta = 1.03
        Z1, Z2 = self.coeff
        b0 = 1/8/pi/epsilon_0*abs(Z1*Z2)*const.e**2/kB/self.rdcdT
        L = 2*DebL/beta/b0
        ##
        # print(f"L = {L}")
        # print(f"temp = {2 * pi * kB * self.rdcdT / self.rdcdM}")
        # print(f"gamma={gamma(s)}")
        ##
        Psi = 0 if (s == 1) else (sum(1/n for n in range(1, s)))
        tmp1 = l*sqrt(2*pi*kB*self.rdcdT/self.rdcdM)*beta**2*b0**2
        tmp2 = gamma(s)
        tmp3 = (ln(L) - l/2 - 2*0.57722 + Psi)
        return tmp1*tmp2*tmp3
        # return l * sqrt(2 * pi * kB * self.rdcdT / self.rdcdM) * beta ** 2 * b0 ** 2 * gamma(s) * \
        #     (ln(L) - l / 2 - 2 * 0.57722 + Psi)


# ----------------------------------------------------------------------------------------------- #
class EleNeuColliIntegral(AbsColliIngrl):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        r"""

        Parameters
        ----------
        pairs_str
        style
        coeff_str:
            relative path of momentum transfer cross section file

        Notes
        -----
        """
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)
        self._set_paras()

    def _set_paras(self):
        data = np.loadtxt(self.coeff_str, comments="#", delimiter="\t")
        self.Q_f = interp1d(data[:, 0], data[:, 1], fill_value=0.0,
                            bounds_error=False)

    def get_rdcd_ci(self):
        pass

    def getColliIntegral(self, *, index: tuple):
        def f1(y):
            s = index[1]
            x1 = (s + 1)/2*(1 + y)
            return (s + 1)/2*exp(-x1)*x1**(s + 1)*self.Q_f(x1*self.rdcdT*K2eV)

        def f2(y):
            s = index[1]
            if abs(y) < 1e-15:
                return 0
            x2 = (s + 1)/abs(y)
            return (s + 1)/abs(y)**2*exp(-x2)*x2**(s + 1)*self.Q_f(x2*self.rdcdT*K2eV)

        N = 50
        temp = integrate_by_DCT(f1, N) + 0.5*integrate_by_DCT(f2, N)
        return 0.5*sqrt(kB*self.rdcdT/2/pi/self.rdcdM)*temp


# ----------------------------------------------------------------------------------------------- #
class TableColliIntegral(AbsColliIngrl):

    def __init__(self, *, pairs_str: str, style: str, coeff_str: str):
        super().__init__(pairs_str=pairs_str, style=style, coeff_str=coeff_str)

    def getColliIntegral(self, *, index: tuple):
        pass
        # return super().getColliIntegral(index=index)
