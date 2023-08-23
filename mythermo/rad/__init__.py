# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 17.Jul.2023
"""
from math import exp

from myconst import Hz2eV, K2eV
from pandas import read_csv

from mythermo import path


class EmptyCS(object):

    def __init__(self):
        pass

    def norm_cs(self, *, nu: float, T_K: float) -> float:
        return 0


class PhoiCS(object):
    __slots__ = ["_df"]

    def __init__(self, *, df_file):
        self._df = read_csv(df_file, index_col=0)

    def norm_cs(self, *, nu: float, T_K: float) -> float:
        tmp = 0
        Pn = 1
        for _id in self._df.index:
            if nu*Hz2eV <= self._df.loc[_id, "Eth_eV"]:
                continue
            else:
                cs = 2.815404e+25*nu**(-3)*self._df.loc[_id, "Zeff"]**4/self._df.loc[_id, "n"]**5
                Qj = self._df.loc[_id, "g"]*exp(-self._df.loc[_id, "E"]/(T_K*K2eV))
                tmp = tmp + cs*Qj
        return tmp

    # def kappa(self, *, nu: float, T_K:float):
    #     return self.value(nu=nu) * self._df.loc[_id, "g"] * \
    #         exp(-self._df.loc[_])


bfCS_file_dict = {"Ar": f"{path}/rad/bf_cs/Ar I.csv",
                  "Ar_1p": f"{path}/rad/bf_cs/Ar II.csv",
                  "Ar_2p": f"{path}/rad/bf_cs/Ar III.csv",
                  "Xe": f"{path}/rad/bf_cs/Xe I.csv",
                  "Xe_1p": f"{path}/rad/bf_cs/Xe II.csv"}
