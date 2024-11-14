"""
The script to test the thermal analisys for the homework:
"""
import numpy as np
import matplotlib.pyplot as plt
import Functions as fun
import Correlations as co

# Data 
    # Irradiation time and power data:
q1_avg = 38.7 * 1e3 #[W/m] linear power (in the central node)
node_coef = np.array([0.527, 0.737, 0.868, 0.958, 1, 0.983, 0.912, 0.802, 0.658, 0.498])
t_irr = 360 #[days] irradiation period
    # Thermo - hydraulic specifications for the coolant (Sodium: Na):
pin_p = 8.275 * 1e-3 #[m] the pin pitch
Tc_in = 395 #[°C] coolant inlet temperature
pc_in = 0.1 #[MPa] coolant inlet pressure
mdot = 0.049 #[kg/s] coolant mass flow rate 
    # Fuel - Pin specification for MOX fuel:
        # Geometrical properties:
H_a = 850 * 1e-3 #[m] fuel column active height
H_p = 7 * 1e-3 #[m] fuel pellet height
R_fo = (5.42 / 2) * 1e-3 #[m] fuel pellet outer radius
R_co = (6.55 / 2) * 1e-3 #[m] cladding outer radius
D_grain = 10 * 1e-6 #[m] fuel grain diamter 
        # Material properties:
rho_TD = 11.31 * 1e3 #[kg/m^3] fuel theoretical density 
P = (1 - 0.945) #[-] fuel porosity
rho_MOX = rho_TD * (1 - P) #[kg/m^3] fuel real density
O_M = 1.957 #[-] oxigen to metal rateo
Tg_in = 20 #[°C] initial tempearture of filling gas
pg_in = 0.1 #[MPa] initial pressure of filling gas
        # Fuel chemical/material properties:
Pu = 0.29 #[wt%] The concentration of plution in weight percentage
# Bu = 0
Bu = co.burnup(q1_avg, H_a, R_fo, rho_MOX, t_irr) #[GWd/ton(HM)]
Tmelt_fu = co.t_melt_fuel(Pu, Bu)
Tmelt_cl = 1673 - 273.15 #[°C]

# here we build the z-axis vector containing the positions of the power nodes:
H_n1 = H_a / (np.size(node_coef) * 2) # the posizion of the first node
z_node = np.linspace(H_n1, H_a - H_n1, np.size(node_coef))
H_node = H_a / np.size(node_coef)
# And here we do the reverse of the avaradge to find the maximum value of the linear heat rate,
# and hen we multipy it to find the array where we report the linear heat rate for each node:
q1_max = (q1_avg * np.size(node_coef)) / (np.sum(node_coef))
q1_vec = q1_max * node_coef

# GUESS for the gap thickness:
t_gap = 0.00016 #[m]
# t_gap = np.mean((R_fo, R_co)) - R_fo

# And now we use the previouslt built function to calculate the axial temperature profile
# of the coolant discretized according to the nodal division of the linear heat rate
n = 100
f = 1 / 2 # The form factor accounting for the triangular pitch and our positioning in the equivalent channel
R_co_vec = np.ones(np.size(node_coef)) * R_co # The vecotr of the cladding outer radius along z in COLD GEOMETRY
R_ci_vec = np.ones(np.size(node_coef)) * (R_fo + t_gap) # The vecotr of the cladding inner radius along z in COLD GEOMETRY
R_fo_vec = np.ones(np.size(node_coef)) * R_fo # The vecotr of the fuel outer radius along z in COLD GEOMETRY
R_fi_vec = np.zeros(np.size(node_coef)) # The vecotr of the fuel inner radius along z in COLD GEOMETRY

[T_z_cool, T_map, T_ax_vec, R_tot, u_Na, h_Na] = fun.Th_Analysis(q1_vec, mdot, Tc_in, pin_p, R_co_vec, R_ci_vec, R_fo_vec, R_fi_vec, H_node, P, Pu, Bu, f, n)

# Now we procede to accounto for the thermal expansion effect by finding the HOT GEOMETRY
alpha_l = 1.2 * 1e-5 #[°C^-1]
T_ref = 25 #[°C] reference ambient temperature
# Here we must iterate the calculations for the radii and temperatures so we define the starting values as '_old'
R_co_cold = R_co_vec
R_ci_cold = R_ci_vec
R_fo_cold = R_fo_vec
R_fi_cold = R_fi_vec

[T_z_cold, T_cold_m, T_cold_v, R_cold, u_cold, h_cold] = fun.Th_Analysis(q1_vec, mdot, Tc_in, pin_p, R_co_cold, R_ci_cold, R_fo_cold, R_fi_cold, H_node, P, Pu, Bu, f, n)
[R_co_hot, R_ci_hot, R_fo_hot, R_fi_hot] = fun.Th_Expansion(R_co_cold, T_cold_v[3], R_ci_cold, T_cold_v[2], R_fo_cold, T_cold_v[1], R_fi_cold, T_cold_v[0], alpha_l, T_ref)
[T_z_hot, T_hot_m, T_hot_v, R_hot, u_hot, h_hot] = fun.Th_Analysis(q1_vec, mdot, Tc_in, pin_p, R_co_hot, R_ci_hot, R_fo_hot, R_fi_hot, H_node, P, Pu, Bu, f, n)

# # And here we create the while cycle to interate untill convergence
# err = 100
# while err >= 1e-5:
#     [T_z_cold, T_cold_m, T_cold_v, R_cold, u_cold, h_cold] = fun.Th_Analysis(q1_vec, mdot, Tc_in, pin_p, R_co_cold, R_ci_cold, R_fo_cold, R_fi_cold, H_node, P, Pu, Bu, f, n)
#     [R_co_hot, R_ci_hot, R_fo_hot, R_fi_hot] = fun.Th_Expansion(R_co_cold, T_cold_v[3], R_ci_cold, T_cold_v[2], R_fo_cold, T_cold_v[1], R_fi_cold, T_cold_v[0], alpha_l, T_ref)
#     [T_z_hot, T_hot_m, T_hot_v, R_hot, u_hot, h_hot] = fun.Th_Analysis(q1_vec, mdot, Tc_in, pin_p, R_co_hot, R_ci_hot, R_fo_hot, R_fi_hot, H_node, P, Pu, Bu, f, n)
#     # We define the error as:
#     err = np.sum(np.abs((T_cold_v[3] - T_hot_v[3])) + np.abs((T_cold_v[2] - T_hot_v[2])) + np.abs((T_cold_v[1] - T_hot_v[1])) + np.abs((T_cold_v[0] - T_hot_v[0])))
#     # And we change the values for the cold case to reiterate the loop till convergence:
#     R_co_cold = R_co_hot
#     R_ci_cold = R_ci_hot
#     R_fo_cold = R_fo_hot
#     R_fi_cold = R_fi_hot

# Let's  finally plot our so bastard to get results! 
z_cool = np.arange(0, H_node * np.size(node_coef), H_node)

plt.figure(1)
plt.plot(T_z_cool, z_cool, label = '$T_{cool}$')
plt.plot(T_ax_vec[0], z_cool, label = '$T_{co}$')
plt.plot(T_ax_vec[1], z_cool, label = '$T_{ci}$')
plt.plot(T_ax_vec[2], z_cool, label = '$T_{fo}$')
plt.plot(T_ax_vec[3], z_cool, label = '$T_{fi}$')
plt.axvline(x = Tmelt_cl, color = 'orange')
plt.axvline(x = Tmelt_fu, color = 'red')
plt.xlabel('Temperature along z-axis $[°C]$')
plt.ylabel('z-axis of fuel column height $[m]$')
plt.legend()

plt.figure(2)
plt.plot(z_cool, u_Na, label = 'COLD', color = 'blue')
plt.plot(z_cool, u_hot, label = 'HOT', color = 'red')
plt.xlabel('z-axis of fuel column height $[m]$')
plt.ylabel('Sodium velocity along z-axis $[m/s]$')
plt.legend()

plt.figure(3)
plt.plot(z_cool, h_Na, label = 'COLD', color = 'blue')
plt.plot(z_cool, h_hot, label = 'HOT', color = 'red')
plt.xlabel('z-axis of fuel column height $[m]$')
plt.ylabel('Heat transfer coefficint along z-axis $[W/m^{2}*K]$')
plt.legend()

plt.figure(4)
plt.plot(R_tot, T_map[9, :], label = 'T(r, z = 9°nodo) COLD GEOMETRY', color = 'blue')
plt.plot(R_hot, T_hot_m[9, :], label = 'T(r, z = 9°nodo) HOT GEOMETRY', color = 'red')
plt.xlabel('r-axis of the fuel column $[m]$')
plt.ylabel('Tempertaure profile [°C]')
plt.axvline(x = R_co_vec[9], color = 'blue')
plt.axvline(x = R_co_hot[9], color = 'red')
plt.axvline(x = R_ci_vec[9], color = 'blue')
plt.axvline(x = R_ci_hot[9], color = 'red')
plt.axvline(x = R_fo_vec[9], color = 'blue')
plt.axvline(x = R_fo_hot[9], color = 'red')
plt.axvline(x = R_fi_vec[9], color = 'blue')
plt.axvline(x = R_fi_hot[9], color = 'red')
plt.legend()

plt.show()

# print(f'The cold temperature profiles at the interfaces are ', T_cold_v)
# print(f'The hot temperature profiles at the interfaces are', T_hot_v)
# print(R_hot)
# print(R_co_cold - R_co_hot)
# T_cl_avg = np.mean(np.array([T_ax_vec[3], T_ax_vec[2]]), axis = 0)
# T_fu_avg = np.mean(np.array([T_ax_vec[1], T_ax_vec[0]]), axis = 0)
# R_ci_hot2 = R_ci_vec * (1 + co.epsilon_cl(T_cl_avg))
# R_fo_hot2 = R_fo_vec * (1 + alpha_l * (T_fu_avg - T_ref))
# print(t_gap - (R_ci_hot2 - R_fo_hot2))
