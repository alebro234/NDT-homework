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
R0 = np.zeros(nz)

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
# plt.plot(z, q_lin(z))
# plt.xlabel('z [m]')
# plt.ylabel('q [W m^-1]')
# plt.show()
Q = np.trapz(q_lin(z), z)
print(f"ROD POWER: {Q} W")

t_irr = 365 # d
temps = fn.thermal_effects(q_lin, t_irr, Rco, Rci, Rf)
T_cool = temps["T_cool"]
Tco = temps["T_co"]
Tci = temps["T_ci"]
Tgap = temps["T_gap"]
Tfuel = temps["T_f"]

print(f"T max fuel: {Tfuel.max()}")
print(f"T max gap: {Tgap.max()}")
print(f"T max clad: {Tci.max()}")
print(f"T max coolant: {T_cool.max()}")
print(f"R0: {0} mm, Rf: {Rf[0]*1e3} mm, Rci: {Rci[0]*1e3} mm, Rco: {Rco[0]*1e3:.2f} mm")

plt.plot(T_cool, z, label='T_cool')
plt.plot(Tco, z, label='Tco')
plt.plot(Tci,z, label='Tci')
plt.plot(Tgap,z, label='Tgap')
plt.plot(Tfuel[:,-1], z, label='Tfuel outer')
plt.plot(Tfuel[:,0], z, label='Tfuel inner')
plt.legend()
plt.show()

fn.expand(q_lin, t_irr, Rco, Rci, Rf, R0)

