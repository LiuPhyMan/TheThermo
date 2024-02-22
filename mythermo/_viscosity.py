# -*- coding: utf-8 -*-
"""
@author: Liu.Jinbao
@contact: liu.jinbao@outlook.com
@time: 30.June.2023
"""

import numpy as np


class viscosity(object):

  def __init__(self) -> None:
    pass


# One prime
p, q = 1, 2
# 0, 0
if (p, q) == (0, 0):
  coeffs = [[80/3, (1, 1), (1, 1)],
            [8, (0, 2), (2, 2)]]
elif (p, q) == (0, 1):
  coeffs = [[280/3, (1, 2), (1, 1)],
            [-112/3, (1, 2), (1, 2)],
            [28, (0, 3), (2, 2)],
            [-8, (0, 3), (2, 3)]]
elif (p, q) == (1, 1):
  coeffs = [[560/3, (3, 1), (1, 1)],
            [980/3, (1, 3), (1, 1)],
            [-784/3, (1, 3), (1, 2)],
            [128/3, (1, 3), (1, 3)],
            [308/3, (2, 2), (2, 2)],
            [294/3, (0, 4), (2, 2)],
            [-56, (0, 4), (2, 3)],
            [8, (0, 4), (2, 4)],
            [16, (1, 3), (3, 3)]]
else:
  raise Exception("")


# Two prime
def coeffs(p, q):
  if (p, q) == (0, 0):
    coeffs = [[-80/3, (1, 1), (1, 1)],
              [8, (1, 1), (2, 2)]]
  elif (p, q) == (0, 1):
    coeffs = [[-280/3, (2, 1), (1, 1)],
              [112/3, (2, 1), (1, 2)],
              [28, (2, 1), (2, 2)],
              [-8, (2, 1), (2, 3)]]
  elif (p, q) == (1, 1):
    coeffs = [[-1540/3, (2, 2), (1, 1)],
              [784/3, (2, 2), (1, 2)],
              [-128/3, (2, 2), (1, 3)],
              [602/3, (2, 2), (2, 2)],
              [-56, (2, 2), (2, 3)],
              [8, (2, 2), (2, 4)],
              [-16, (2, 2), (3, 3)]]
  else:
    raise Exception("")
  return coeffs


#
# 0, 1
# 1, 1
def H(*, i, j, p, q, CI):
  coeffs
  A1 = sum(
    [_c[0]*mu1**(_c[1][0])*mu2**(_c[1][1])*CI.getcolliIntgrl(index=_c[2]) for
     _c
     in coeffs(p, q)])
  A2 = sum(
    [_c[0]*mu1**(_c[1][0])*mu2**(_c[1][1])*CI.getcolliIntgrl(index=_c[2]) for
     _c
     in coeffs(p, q)])

  for iRow in range(K):
    for jCol in range(K):
      M1 = m[iRow]/(m[iRow] + m[jCol])
      M2 = m[jCol]/(m[iRow] + m[jCol])

      tmp = 0
      for _c in coeffs1(p, q):
        tmp = tmp + _c[0]*M1**_c[1][0]*M2**_c[1][1]*CI.getColliIntegral(
          index=(_c[2]))
      A1[iRow, jCol] = tmp*xj[iRow]*xj[jCol]
      tmp = 0
      for _c in coeffs2(p, 1):
        tmp = tmp + _c[0]*M1**_c[1][0]*M2**_c[1][1]*CI.getColliIntegral(
          index=(_c[2]))
      A2[iRow, jCol] = tmp*xj[iRow]*xj[jCol]
  H[iRow, jCol] = A2[iRow, jCol]
  if i == j:
    H[iRow, jCol] = H[iRow, jCol] + np.sum(A1[iRow])
  H = H*2/5/kB/T
  return H


b10 = np.linalg.solve(H, 2/kB/T*xj)
eta = 1/2*kB*T*np.dot(xj, bj)

# two order
b20 = np.linalg.solve(H00 - np.dot(H01.dot(np.linalg.inv(H11)), H10),
                      2/kB/T*xj)
eta = 1/2*kB*T*np.dot(xj, b20)
