import numpy as np
from Plenum_calc import FGR

rho_TD = 11.31 * 1e3 #[kg/m^3] fuel theoretical density 
P = (1 - 0.945) #[-] fuel porosity
rho_f = rho_TD * (1 - P) #[kg/m^3] fuel real density

Flux = 6.1 * 1e15
R_fo = (5.42 * 1e-3) / 2
t_irr = 365 * 24 * 3600 #[s]

[y, A, p, n1, n2] = FGR(1, rho_f, Flux, 2700, 1200, t_irr, 0, R_fo, 160e-6, 0.85, 5)
print(f"Plenum pressure: {p:.5f} MPa")
print(f"Fission gas moles: {n1:.5f}")
print(f"Helium moles: {n2:.5f}")
print(f"fission gas molar fraction: {y:.9f}")

