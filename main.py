import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import functions as fn
import Correlations as cor

                                             ### DATA ###

    ## Neutronic specifications
# neutron flux @ peak power node
Phi = 6.1 # 10^15 n cm^-2 s^-1

    ## Thermo-hydraulic specifications
pitch = 8.275e-3 # m
Tc_in = 395 # °C
Pc_in = 0.1e6 # Pa
G = 0.049 # kg s^-1

    ## Fuel pin specifications
Ha = 0.850 # m
Df = 5.42e-3 # m
Rfo = Df/2
Hf = 0.007 # m
TD = 11310 # kg m^-3
rho_f = 0.945 * TD
p = 1 - 0.945
D_grain = 10e-6 # m
O_M_ratio = 1.957
x = (2 - O_M_ratio) / 100
Rco = 0.5*6.55e-3 # m
P_He = 0.1e6 # Pa
T_He = 20 # °C

    ## Fuel composition
wt_Pu = 0.29
wt_235_U = ( 1 - wt_Pu ) * 0.00711
wt_238_U = 1 - wt_235_U - wt_Pu

    ## Cladding composition
cl_comp = [15.0, 15.0, 1.5, 1.5, 0.9, 0.4, 0.09] # wt%
B_conc = 60 # ppm







# Linear power, from polynomial fitting
q_avg = 38.7e3 # W m^-1
c =  np.array([0.572, 0.737, 0.868, 0.958, 1, 0.983, 0.912, 0.802, 0.658, 0.49])
# max power
q0 = q_avg*10 / c.sum()
z_fit = np.array([42.5 ,127.5, 212.5, 297.5, 382.5, 467.5, 552.5 ,637.5, 722.5, 807.5])*0.001 - Ha/2
q_fit = 0.5 * q0 * c # 0.5 triangular channel factor
polyn = np.poly1d( np.polyfit(z_fit, q_fit, q_fit.size-1) )

z = np.linspace(-Ha/2, +Ha/2)
q_lin = lambda z: polyn(z) # axial linear power [W m^-1]

# MOX fuel after 1 week irradiation time
BU = cor.burnup(q_avg, Ha, Rfo, rho_f, 7)
print(f"BU = {BU}")
MOX = fn.Fuel(x, p, wt_Pu, 56)



# Random values

t_gap = 0.16e-3
Rci = Rfo + t_gap

z_vec = np.linspace(-Ha/2, Ha/2, 100)
r_fuel_vec = np.linspace(0, Rfo)

print(f"Cold geometry: {Rfo*1000:.2f}, {Rci*1000:.2f}, {Rco*1000:.2f} mm")
print(f"t_gap = {t_gap*1000} mm")
z, r_fuel, T_fuel, r_clad, T_clad, Tgap  = fn.thermal_analysis(q_lin, Rco, Rci, Rfo, MOX)
print(f"Max fuel T: {T_fuel.max()} °C")
print(f"Max clad T: {T_clad.max()} °C")

# Tfo = T_fuel[:,-1].max()
# fn.restructuring(q_avg, TD, 0.97*TD, 0.95*TD, rho_f, Rfo_hot, x, wt_Pu, BU, Tfo)


# n_z = len(z)
# Tfo = T_fuel[int(n_z/2), -1]
# r0, r1, r2, T0 = fn.restructuring(q0, TD, 0.98*TD, 0.96*TD, rho_f, Rfo_hot, x, wt_Pu, BU, Tfo)
# print(f"\n\nRestructuring radii: {r0*1000}, {r1*1000}, {r2*1000}, {Rfo_hot*1000} mm")
# print(f"At temperatures: {T0}, 1800, 1600, {Tfo} °C")




# # T(r,z) plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# surf = ax.plot_surface(z*1000, r_fuel*1000, T_fuel, cmap="viridis")
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
# plt.title("Fuel temperature")
# plt.xlabel("z [mm]")
# plt.ylabel("r [mm]")
# plt.clabel("T [°C]")

# plt.show()









