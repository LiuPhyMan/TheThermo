from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator as interp2D
from myconst import Ry_eV, eV2K, nm2Hz_f, Hz2K

data = np.loadtxt(str(Path(__file__).parent/Path('data/gauntff.dat')),
                  comments='#', encoding='utf')
interp = interp2D((np.arange(-16, 13 + 0.2, 0.2), np.arange(-6, 10 + 0.2, 0.2)), data)


def gff(*, wvlnm, T_K, Z):
    nuHz = nm2Hz_f(wvlnm)
    log_u = np.log10(nuHz*Hz2K/T_K)
    log_gamma2 = np.log10(Z**2*Ry_eV*eV2K/T_K)
    return float(interp((log_u, log_gamma2)))


gffint_data = np.loadtxt(str(Path(__file__).parent/Path('data/gauntff_freqint.dat')), skiprows=37)
x = gffint_data[:, 0]
y = gffint_data[:, 1]


def gffint(*, T_K, Z):
    log_gam2 = np.log10(Z**2*Ry_eV*eV2K/T_K)
    return np.interp(log_gam2, x, y)
