import matplotlib.pyplot as plt
import numpy as np
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

# axial linear power evaluation
q_avg = 38.7e3  # W m^-1
c = np.array([0.572, 0.737, 0.868, 0.958, 1, 0.983, 0.912, 0.802, 0.658, 0.49])
# max power @ z = 0
q0 = q_avg*10 / c.sum()
z_fit = np.array([42.5, 127.5, 212.5, 297.5, 382.5, 467.5,552.5, 637.5, 722.5, 807.5])*0.001 - Ha/2
q_fit = q_avg * c
polyn = np.poly1d(np.polyfit(z_fit, q_fit, q_fit.size-1))
z = np.linspace(-Ha/2, +Ha/2, nz)
def q_lin(z): return polyn(z)  # q = q(z) [W m^-1]

# Pu redistribution
y0 = 0.29
y_Pu = fn.Pu_redistribution(Rf, 0)

q_vol = lambda r, z: q_lin(z) * y_Pu(r)
y_Pu = lambda r: q0 * y_Pu(r)

T_cool, Tco, Tci, Tgap, Tfuel = fn.thermal_effects(q_lin, Rco, Rci, Rf)

print(f"T max fuel: {Tfuel.max()}")

plt.plot(T_cool, z, label='T_cool')
plt.plot(Tco, z, label='Tco')
plt.plot(Tci,z, label='Tci')
plt.plot(Tgap,z, label='Tgap')
plt.plot(Tfuel[:,-1], z, label='Tfuel outer')
plt.plot(Tfuel[:,0], z, label='Tfuel inner')
plt.legend()
plt.show()

