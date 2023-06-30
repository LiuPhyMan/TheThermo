# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 23.May.2023
"""
import numpy as np
from typing import List
from myspecie import spc_dict

class AbsComp(object):

    def __init__(self, *, spcs_str: List[str]):
        self.spcs_str = spcs_str
        self.n_spcs = len(spcs_str)
        self.spcs = [spc_dict[_] for _ in self.spcs_str]
        self._set_elements_by_species()
        self.Zc = [_.Zc for _ in self.spcs]
        self.Nj = np.zeros(self.n_spcs)
        self.lnNj = np.zeros(self.n_spcs)

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

    def __init__(self, *, spcs_str: List[str]):
        super().__init__(spcs_str=spcs_str)

    def rdcd_mu0(self, *, T_K: float):
        return np.array([_.rdcd_mu0(T_K=T_K) for _ in self.spcs])

    def set_Nj(self, *, Nj):
        self.Nj = Nj

    def LD(self, *, T_K: float):
        

