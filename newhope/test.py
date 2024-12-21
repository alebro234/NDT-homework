import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import functions as fn

# number of axial nodes
nz = 100
# number of radial nodes in the fuel
nr = 100
Ha = 0.85

t_gap = 0.16e-3 * np.ones(nz)
Rf = 5.42e-3/2 * np.ones(nz)
Rci = Rf + t_gap
Rco = 6.55e-3/2 * np.ones(nz)

OM = 2 - 1.945
y0 = 0.29
p = 1 - 0.945
t_irr = 365
# axial linear power evaluation
q_0 = 38.7e3  # W m^-1
c = np.array([0.572, 0.737, 0.868, 0.958, 1, 0.983, 0.912, 0.802, 0.658, 0.49])
# # max power @ z = 0
# q0 = q_avg*10 / c.sum()
z_fit = np.array([42.5, 127.5, 212.5, 297.5, 382.5, 467.5,552.5, 637.5, 722.5, 807.5])*0.001 - Ha/2 # m
q_fit = q_0 * c
polyn = np.poly1d(np.polyfit(z_fit, q_fit, q_fit.size-1))
z = np.linspace(-Ha/2, +Ha/2, nz)
def q_lin(z): return polyn(z)  # q = q(z) [W m^-1]
# total rod power [W]
Q = np.trapz(q_lin(z), z)
# burnup [GWd tHM^-1]
BU = 45

Rf_avg = np.mean(Rf)
r0 = 0.5e-3
chi, I = fn.radial_redstr(Rf_avg, r0)
kf = lambda T: fn.tc_fuel(T, OM, y0, p, BU)
Omega = np.pi * (Rf_avg**2 - r0**2)
q0 = lambda z : q_lin(z) / Omega
def dTdr(r, T):
    return - q0(0) / (kf(T) * r) * (I(r) - I(r0))


