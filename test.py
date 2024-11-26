import numpy as np
import ChannelAnalysis as ca
import matplotlib.pyplot as plt

# channel geometry
pitch = 8.275e-3  # m
Ha = 0.85  # m

# irradiation time [d]
t_irr = 7

# num of axial nodes
nz = 100
# num of radial nodes in clad and fuel
nr = 100

# axial linear power evaluation
q_avg = 38.7e3  # W m^-1
c = np.array([0.572, 0.737, 0.868, 0.958, 1, 0.983, 0.912, 0.802, 0.658, 0.49])
# max power @ z = 0
q0 = q_avg*10 / c.sum()
z_fit = np.array([42.5, 127.5, 212.5, 297.5, 382.5, 467.5,
                 552.5, 637.5, 722.5, 807.5])*0.001 - Ha/2
q_fit = q_avg * c
polyn = np.poly1d(np.polyfit(z_fit, q_fit, q_fit.size-1))
z = np.linspace(-Ha/2, +Ha/2, nz)
def q_lin(z): return polyn(z)  # q = q(z) [W m^-1]


fuel = ca.MOXFuel(R=5.42e-3/2, nz=nz, nr=nr, Ha=Ha)
gap = ca.HeGap(t=0.16e-3, nz=nz, Ha=Ha)
clad = ca.Cladding(Ri=fuel.R + gap.t, Ro=6.55e-3/2, nz=nz, nr=nr, Ha=Ha)
cool = ca.NaCoolant(nz=nz, Ha=Ha)


fuel.evaluate_burnup(q_avg, t_irr)

fuel, gap, clad, cool = ca.thermal_analysis(
    q_lin, 0.5, pitch, fuel, gap, clad, cool)

fuel.plot_T_colormap()
plt.show()
clad.plot_T_colormap()
# fuel.restructure(q_avg)

print(f"Tmax fuel = {fuel.T.max()}")
print(f"Fuel melting T = {fuel.T_melt()}")


plt.show()
