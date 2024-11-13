"""
Created on Fri Oct 18 15:52:25 2024
@author: rumbl
"""
import numpy as np
import sympy as sp
import scipy as sc
import matplotlib.pyplot as plt
import scipy.constants
from scipy.optimize import fsolve

# Data

# Fuel Assembly (FA) & Equivalent Channel (EC) characteristics in avaradge conditions: 
Q_FA = 6.641 * 1e6 #[W]
N_FP = 284 # Number of Fuel Pins in a Fuel Assembly
p_FP = 13.6 * 1e-3 #[m] Pitch between two pins
q_prime0 = 26.03 * 1e3 #[W/m]
Tc_in = 400 #[°C]
Tc_out = 480 #[°C]

# Coolant properties averaged and uniform along the Z axis - Liquid Lead (Pb):
C_Pb = 146 #[J/°C*Kg] Heat capacity 
rho_Pb = 10520 #[kg/m^3] Density
mu_Pb = 0.002 #[kg/m*s] or [Pa*s] Dynemic Viscosity
k_Pb = 15.5 #[W/m*°C] Thermal Conductivity

# Fuel Pins geomtrical dimensions:
D_cl = 8.5 * 1e-3 #[m] Pin diameter = 2*R_co
t_c = 0.6 * 1e-3 #[m] Cladding Thickness
H_a = 1.1 #[m] Active Length
H_tot = 2.1 #[m] Total Length

# Solid Fuel Pellet (MOX) properties:
rho_MOX = 0.88 #[%TD] with respect to a Totaly Dense (no porosity) Pellets
T_m_fu = 2730 #[*C] Melting Temperature of Fresh Fuel (FF)
D_fu = 6.98 * 1e-3 #[m] Pellet diamter = 2*R_fo
alpha_th_fu = 12 * 1e-6 #[°C^-1] Fuel's Thermal Expantion linear coefficient
T_ref = 25 #[°C] Reference Temperature for the Delta_T in Th. Expantion Eq.

# Cladding - Ferritic-Martensitic Steel T91 properties:
T_m_cl = 1500 #[°C] Melting Temperature of Cladding
k_cl = 28.7 #[W/m*°C] 
alpha_th_cl = 12 * 1e-6 #[°C^-1] 
'''
Cold Geometry Analysis
I) Fuel assembly in avaradge core conditions:
'''
# I a) Coolant mass flowrate (m_dot), avardge velocity (v_avg), pressure drops (deltaP) ?

# First we find the usefull geometrical parameters:
R_co = D_cl/2 #[m] The Pin radius and so the Cladding outer radius
Omega = p_FP**2 - np.pi*(R_co)**2 #[m^2] Area of passage of the equivalent channel
p_w = 2*np.pi*R_co #[m] Wetted perimeter
D_h = 4*Omega/p_w #[m] Hydraulic diamter

# Than we find the mass flowrate form an energy balance on the single pin:
q_FP = Q_FA/N_FP #[W] The heat flowrate of the single Pin
m_dot = q_FP/(C_Pb*(Tc_out - Tc_in)) #[kg/s] 
print('The mass flowrate of Lead across the channel is', m_dot, '[kg/s]')

# Than we use the mass flowrate definition to find the velocity:
v_avg = m_dot/(rho_Pb*Omega) #[m/s]
print('The velocity of Lead across the channel is', v_avg, '[m/s]')

# Finally we evaluate the different pressure drops with a correlation for the Darcy friction factor:
Re_Pb = (rho_Pb*v_avg*D_h)/mu_Pb #[-]
# Defining the correlation with is applicability intervals.
def f_darcy(Re):
    if Re > 3*1e4 and Re < 1e6:
        f_darcy = 0.184*Re**(-0.2)
    else:
        raise ValueError('The Rainolds Number is outside the applicability interval of the correlation')
    return f_darcy
f_Pb = f_darcy(Re_Pb) #[-]
deltaP_f = (f_Pb*H_tot*rho_Pb*(v_avg**2))/(2*D_h) #[kg/m*s^2] or [Pa] frictional pressure drop
deltaP_g = rho_Pb*9.81*H_tot #[Pa] gravitational pressure drops
deltaP_tot = deltaP_g + deltaP_f #[Pa]
deltaP_rel = deltaP_g/deltaP_f #[-] relative percentege of gravitational to frictional pressure drops

if deltaP_rel > 1e3:
    print('The Frictional pressure drops are negligible and the totale pressure drop across the channel is', deltaP_g, '[Pa]')
elif deltaP_rel < 1e-3:
    print('The Gravitational pressure drops are negligible and the total pressure drop across the channel is', deltaP_f, '[Pa]')
else :
    print('Neither Gravitational or Frictional pressure drops are negligible and the total is', deltaP_tot, '[Pa]')
    
# I b) Avaradge Heat Transfer Coefficient (HTC) between the coolant and the Fuel Pin 

# First we implement a function to calculate HTC choosing based on the applicability interval:
def HTC(Re, D, D_h, p, mu, cp, k):
    Pr = (cp*mu)/k
    Pe = Re*Pr
    x = p/D #[-] Pin-Pitch to Pin outer diamter rateo (packing rateo)
    if 80 < Pe < 4000 and 1.1 < x < 1.5:
        print('Evaluating the Peclet Number and the packing rateo we find we CAN use the fisrt correlation for Nu')
        Nu = 0.58*((D_h/D)**(0.55))*(Pe**(0.45))
        h = (Nu*k)/D_h
    else:
        print('Evaluating the Peclet Number and the packing rateo we find we must use the general, less conservative, second correlation for Nu')
        Nu = 4.5 + 0.014*(Pe**(0.8))
        h = (Nu*k)/D_h
    return h, Nu, Pr, x

# And now we can find the HTC form Nusselt's number definition:
h_Pb, Nu_Pb, Pr_Pb, x = HTC(Re_Pb, D_cl, D_h, p_FP, mu_Pb, C_Pb, k_Pb) #[W/m^2*K]
print('The HTC calculated with the most appropiate correlation is', h_Pb, '[W/m^2*K]')

'''
II) Fuel Pin in avardge core conditions with axial linear heat rate (q' = q/L) in the active region given by:
    q'(z) = q'(0)*cos(pi*z/H_e)
    where z is the axial coordinate starting in the midplane and H_e is the extrapolated length of the active region 
'''
# II a) Evluate the average linear heat rate (q_prime_avg) and the extrapolated length (H_e)

# First we find q_prime_avg as the heat flowrate of the Fuel Pin divided by the active length of the rod:
q_prime_avg = q_FP/H_a; #[W/m]
print('The avaradge linear heat rate over the active length of the rod  is', q_prime_avg, '[W/m]')

# Then we find the extrapolated length by defining q_prime_avg as the integral mean of the linear heat rate over the active length:
# We define the independent variables and the constants as symbols using the sympy library:
z_sym = sp.symbols('z', real = True)
H_e_sym = sp.symbols('H_e', real = True, nonnegative = True)
H_a_sym = sp.symbols('H_a', real = True, nonnegative = True)
q_prime0_sym = sp.symbols('q^I_0', real = True, nonnegative = True)

# We build the integrand function symbolicaly and we perform the integration on z€[-H_a/2,H_a/2]
q_prime_z_sym = q_prime0_sym*sp.cos((sp.pi*z_sym)/H_e_sym)
q_prime_avg_sym = sp.integrate(q_prime_z_sym/H_a_sym, 
                       (z_sym, -H_a_sym/2, H_a_sym/2))

# We substitute the numerical values of the known quantities in the integrated function where H_e is the unknown 
q_prime_avg_sym = q_prime_avg_sym.subs({
    q_prime0_sym:q_prime0,
    H_a_sym:H_a,
    })

# We buld the final non-linear equation in the unknown H_e and we solve numericaly with the appropiate solver: 
Eq_non_lin = sp.Eq(q_prime_avg, q_prime_avg_sym)
H_e_sym = sp.nsolve(Eq_non_lin, H_e_sym, H_a)
H_e = float(H_e_sym)
print('The value for the extrapultaed heigth of the Fuel Rod is', H_e, '[m]')

# II b) Draw the radial Temperature profile at the midplane (z = 0) of the active heigth,
#       evaluating in particular the Tempertures at the interfaces:

# We must subdvide the internal composition of the pellet along tha radius in different subsections:
#   - from 0 < r < R_fo     we have the internal MOX pellet 
#   - from R_fo < r < R_ci  we have the gap to accomodate fissio products and much more
#   - from R_ci < r < R_co  we have the external Steel cladding
#   - from R_co < r < +oo   we have the Lead colant 

# Being a Power Imposed system we start solving the equivalent thermical resistance network from the coolant all the 
# way up to the internal maximum temperture of the pellet and at the end we shoud result with the radial Tempertaure 
# profile at a given elevation of the Fuel Rod: T(r, z=0)

# First we must characterize the thermal and geometrical properites in the midplane:
T_cool0 = np.mean([Tc_in, Tc_out]) #[°C] the collant temperatue at the midplane z = 0
R_ci = R_co - t_c; #[m] the inner radius of the Cladding
R_fo = D_fu/2 #[m] the outer Fuel Pellet dimeter (in this case the pellet is not hollow)
t_g = R_ci - R_fo #[m] the thickness of the Gap between the Fuel Pellet and the Cladding

# And now we solve from r = R_co up to r = 0, defining a function for each resistance wich has the output of the correct
# Tempertaure interval and will be later used for plotting:
def T_cool(q_prime, HTC, r, T_cool_z):
    T_cool_r = T_cool_z + q_prime/(2*np.pi*HTC*r)
    return T_cool_r
r_cool = np.linspace(R_co, R_co + 0.005, 500)
T_cool_r = T_cool(q_prime0, h_Pb, r_cool, T_cool0)
T_co = T_cool_r[0]
print('The outer Cladding temperature is', T_co, '[°C]')

def T_clad(q_prime, k, r, R, T_co_z):
    T_clad_r = T_co_z + (q_prime*np.log(R/r))/(2*np.pi*k)
    return T_clad_r
r_clad = np.linspace(R_ci, R_co, 500)
T_clad_r = T_clad(q_prime0, k_cl, r_clad, R_co, T_co)
T_ci = T_clad_r[0]
print('The inner Cladding temperature is', T_ci, '[°C]')

def T_gap(q_prime, R1, R2, T_gas, T_ci_z):
    t = R2 - R1 # calculation the tickness as the difference between the two radii
    k_gap = 15.8 * 1e-4 * ((T_gas + 273.15)**(0.79)) #[W/m*°C]
    T_gap_r = T_ci_z + (q_prime*t)/(2*np.pi*R2*k_gap)
    return T_gap_r, k_gap
toll = 100
T_try = T_ci # 0-th initial guess for the tempertaure to be used in finding the initial value for the gas thermal cinductivity
while toll >= 1e-5:
    # Claculating the n-th value for the Temperure of the outer fuel with the (n-th - 1) guess for T_gas
    T_gas_old = T_ci + (T_try - T_ci)/2
    T_fo_old, k_gap_old = T_gap(q_prime0, R_fo, R_ci, T_gas_old, T_ci)
    # With the n-th guess we find the (n-th + 1) value for the outer fuel Temperture 
    T_gas_new = T_ci + (T_fo_old - T_ci)/2
    T_fo, k_gap = T_gap(q_prime0, R_fo, R_ci, T_gas_new, T_ci)
    # We define the error as the difference of those two and we update the initial guess of T_gas
    toll = np.abs(T_fo_old - T_fo)
    T_try = T_fo
T_gas = T_ci + (T_fo - T_ci)/2 #[°C] the temprature of the gas as the mean of the temperatures at the two extremes
r_gas = np.linspace(R_fo, R_ci, 500)
T_gas_r, k_gas = T_gap(q_prime0, r_gas, R_ci, T_gas, T_ci)
print('The outer Fuel temperature is', T_fo, '[°C], while the tempertaure of the gas contained in the gap is', T_gas, 
      '[°C], and wtih it we calculate the conductivity of the gas wich turns out to be', k_gas, '[W/m*°C]')

def T_pellet(q_prime, r, R, T_fo_z):
    # Here we caluclate using he coonductivity integral the maximum temperature of the fuel
    T_fi_z = T_fo_z + q_prime/(4*np.pi*(0.02*1e2))
    T_pellet_r = T_fi_z - (q_prime*(r**2))/(4*np.pi*0.02*1e2*(R**2))
    return T_pellet_r
r_pin = np.linspace(0, R_fo, 500)
T_pin_r = T_pellet(q_prime0, r_pin, R_fo, T_fo)
T_fi = T_pin_r[0]
print('The inner maximum Fuel temperature is', T_fi, '[°C]')

r = np.concatenate((r_pin[:-1], r_gas[:-1], r_clad[:-1], r_cool))
T_r = np.concatenate((T_pin_r[:-1], T_gas_r[:-1], T_clad_r[:-1], T_cool_r))
plt.figure(1)
plt.plot(r, T_r, 'k')
# Plotting the vertical division lines:
plt.axvline(x = R_fo, color = 'red')
plt.axvline(x = R_ci, color = 'orange')
plt.axvline(x = R_co, color = 'yellow')
# Filling the vertical division segments with colours:
plt.axvspan(0, R_fo, color = 'red', alpha = 0.3, label = 'MOX Fuel Pellet')
plt.axvspan(R_fo, R_ci, color = 'orange', alpha = 0.3, label = 'Fuel-Cladding Gap filled with Gas')
plt.axvspan(R_ci, R_co, color = 'yellow', alpha = 0.3, label = 'Stainless Steel Cladding')
plt.axvspan(R_co, r[-1], color = 'green', alpha = 0.3, label = 'Lead(Pb) Coolant')
# Naming the axis:
plt.xlabel('r := The radius from the center of the pellet [m]')
plt.ylabel('T(r)')
plt.grid()

# II b) Draw the axial profile of the Temperature form the coolant (wich will be found using the shape of the linear heat rate))
#       and from there calculate back the axial profile of the Cladding and the Fuel:

# We solve energy balances again starting from the imposed condition wich is on the coolant inlet Temperature (Tc_in)
# by having the axial heat profile (cosine) we are able to define a function wich is going to be the T_cool(z), with the
# equivalent resistance network we will be able to climb back to the fuel temperature along the Z-axis:

def T_z(q_prime, G, Cp, HTC, k_cl, k_gas, H_e, H_a, R_ci, R_co, R_fo, T_in):
    # First we define the usefull beaviours of the linear heat rate and the z-axis:
    z = np.linspace(-H_a/2, H_a/2, 500)
    q_prime_z = q_prime*np.cos((z*np.pi)/H_e)
    # calculate the Temperature of the coolant as function of z:
    T_cool_z = T_in + ((q_prime*H_e)/(G*Cp*np.pi))*(np.sin((np.pi*z)/(H_e)) + np.sin((np.pi*H_a)/(2*H_e)))
    # From here we procede to calculate the other Temperature profiles: T_co(z), T_ci(z), T_fo(z), T_fi(z)
    T_co_z = T_cool_z + (q_prime_z)/(2*np.pi*R_co*HTC)
    T_ci_z = T_co_z + (q_prime_z*np.log(R_co/R_ci))/(2*np.pi*k_cl)
    T_fo_z = T_ci_z + (q_prime_z*(R_ci - R_fo))/(2*np.pi*R_ci*k_gas)
    T_fi_z = T_fo_z + (q_prime_z)/(8*np.pi)
    return z, T_cool_z, T_co_z, T_ci_z, T_fo_z, T_fi_z

# We use the defined function to find the various temperature profiles aas the radius changes:
[z, T_cool_z, T_co_z, T_ci_z, T_fo_z, T_fi_z] = T_z(q_prime0, m_dot, C_Pb, h_Pb, k_cl, k_gas, H_e, H_a, R_ci, R_co, R_fo, Tc_in)
plt.figure(2)
plt.plot(T_cool_z, z, 'blue')
plt.plot(T_co_z, z, 'green')
plt.plot(T_ci_z, z, 'yellow')
plt.plot(T_fo_z, z, 'orange')
plt.plot(T_fi_z, z, 'red')

# II c) Determine the position, starting from the midplane, where the cladding outer Temperature is maximum, and it's value:
T_co_max = np.max(T_co_z) #[°C]
H_co_max = z[np.argmax(T_co_z)] #[m]
T_ci_max = np.max(T_ci_z) #[°C]
H_ci_max = z[np.argmax(T_ci_z)] #[m]
T_fo_max = np.max(T_fo_z) #[°C]
H_fo_max = z[np.argmax(T_fo_z)] #[m]
T_fi_max = np.max(T_fi_z) #[°C]
H_fi_max = z[np.argmax(T_fi_z)] #[m]
print('The maximum temperature of the outer surface of the cladding is', T_co_max, '[°C], and we havita at', H_co_max, '[m] from the midplane')

'''
III) We now consider the subchannel rappresentative of the most demanding operating conditions in the core (THE HOT CHANNEL),
     we keep the same extrapulated length of the cold geometry and the same mass flowrate and we consider a total peak factor,
     wich is the rateo q'(0)_max/q'_avg, equal to 1.5
'''
# III a) Determine the hot channel outer fuel Temperature, and the axial coordinate at wich we have tghe maximum outer cladding Temperature:
q_prime0_hot = 1.5*q_prime0 #[W/m]
[z, T_cool_z_hot, T_co_z_hot, T_ci_z_hot, T_fo_z_hot, T_fi_z_hot] = T_z(q_prime0_hot, m_dot, C_Pb, h_Pb, k_cl, k_gas, H_e, H_a, R_ci, R_co, R_fo, Tc_in)
plt.figure(3)
plt.plot(T_cool_z_hot, z, 'blue')
plt.plot(T_co_z_hot, z, 'green')
plt.plot(T_ci_z_hot, z, 'yellow')
plt.plot(T_fo_z_hot, z, 'orange')
plt.plot(T_fi_z_hot, z, 'red')
Tc_out_hot = T_cool_z_hot[-1]
T_co_max_hot = np.max(T_co_z_hot)
H_co_max_hot = z[np.argmax(T_co_z_hot)]
print('The outlet coolant Temperature in the HOT CHANNEL is', Tc_out_hot, '[°C] while the maximum Temperature of the Cladding is', 
      T_co_max_hot, '[°C] wich is reached at an heigth of', H_co_max_hot, '[m] from the midpalne')    

# Nad now we calculate in the hot channel the radial behaviour with also the consideration of  
# the formation of an oxide layer on the outside, meaning we add a resistance:

# Coolant
T_cool0_hot = np.mean([Tc_in, Tc_out_hot])
T_cool_r_hot = T_cool(q_prime0_hot, h_Pb, r_cool, T_cool0_hot)
T_co_hot = T_cool_r_hot[0]
print('HC: The outer Cladding temperature is', T_co_hot, '[°C]')
# Oxide layer
def T_ox(q_prime, R_co, T_co):
    if T_co <= 380:
        t_max = 1e-6 #[m] the thickness of the oxide layer
        t = np.linspace(t_max, 0, 100)
    elif T_co > 380:
        t_max = (0.26*(T_co - 380) + 1)*1e-6 #[m]
        t = np.linspace(t_max, 0, 100)
    T_ox_r = T_co + (q_prime*t)/(2*np.pi*R_co*1)
    return t, t_max, T_ox_r
[t, t_ox, T_ox_r] = T_ox(q_prime0_hot, R_co, T_co_hot)
T_ox_max = T_ox_r[0]
print('HC: The Oxide maximum temperature is', T_ox_max, '[°C]')
# Cladding
r_clad_hot = np.linspace(R_ci, R_co - t_ox, 500)
T_clad_r_hot = T_clad(q_prime0_hot, k_cl, r_clad_hot, R_co, T_ox_max)
T_ci_hot = T_clad_r_hot[0]
print('HC: The inner Cladding temperature is', T_ci_max, '[°C]')
# Gap
Non_lin_correlation = lambda var: [ - var[1] + 15.8 * 1e-4 * (((T_ci_hot + (var[0] - T_ci_hot)/2) + 273.15)**(0.79)), 
                           - var[0] + T_ci_hot + (q_prime0_hot*t_g)/(2*np.pi*R_ci*var[1])]
[T_fo_hot, k_gap_hot] = fsolve(Non_lin_correlation, [T_ci_hot, k_gap])
T_gas_hot = T_ci_hot + (T_fo_hot - T_ci_hot)/2
T_gap_r_hot, k_gap_hot = T_gap(q_prime0_hot, r_gas, R_ci, T_gas_hot, T_ci_hot)
print('HC: The outer Fuel temperature is', T_fo_hot, '[°C], while the tempertaure of the gas contained in the gap is', T_gas_hot, 
      '[°C], and wtih it we calculate the conductivity of the gas wich turns out to be', k_gap_hot, '[W/m*°C]')
# Pellet
T_pin_r_hot = T_pellet(q_prime0_hot, r_pin, R_fo, T_fo_hot)
T_fi_hot = T_pin_r_hot[0]
print('HC: The inner maximum Fuel temperature is', T_fi_hot, '[°C]')

r_hot = np.concatenate((r_pin[:-1], r_gas[:-1], r_clad_hot[:-1], np.linspace(R_co - t_ox, R_co, 100)[:-1], r_cool))
T_r_hot = np.concatenate((T_pin_r_hot[:-1], T_gap_r_hot[:-1], T_clad_r_hot[:-1], T_ox_r[:-1],T_cool_r_hot))
plt.figure(4)
plt.plot(r_hot, T_r_hot, 'k')
# Plotting the vertical division lines:
plt.axvline(x = R_fo, color = 'red')
plt.axvline(x = R_ci, color = 'orange')
plt.axvline(x = R_co - t_ox, color = 'yellow')
plt.axvline(x = R_co, color = 'green')
# Filling the vertical division segments with colours:
plt.axvspan(0, R_fo, color = 'red', alpha = 0.3, label = 'MOX Fuel Pellet')
plt.axvspan(R_fo, R_ci, color = 'orange', alpha = 0.3, label = 'Fuel-Cladding Gap filled with Gas')
plt.axvspan(R_ci, R_co - t_ox, color = 'yellow', alpha = 0.3, label = 'Stainless Steel Cladding')
plt.axvspan(R_co - t_ox, R_co, color = 'green', alpha = 0.3, label = 'Oxidised Cladding Layer')
plt.axvspan(R_co, r_hot[-1], color = 'blue', alpha = 0.3, label = 'Lead(Pb) Coolant')
# Naming the axis:
plt.xlabel('r_hot := The radius from the center of the pellet in the HOT CHANNEL [m]')
plt.ylabel('T(r)_hot')
plt.grid()

'''
Function for homework and Fuel Restructionig:
'''
def T_radial(q1, h_cool, k_cl, k_f, p, R_co, R_ci, R_fo, R_fi, T_cool):
    
    # Radial tempertaure of the coolant:
    T_co = T_cool + q1 / (2 * np.pi * h_cool * R_co)
    T_cool_r = lambda r: T_cool + (T_co - T_cool) * ((np.log(2 * r / p))/(np.log(2 * R_co / p)))
    r_cool = np.linspace(R_co, p / 2, 500)
    T_cool_vec = T_cool_r(r_cool)
    
    # Radial temperature of the cladding:
    T_ci = T_co + ((q1 * np.log(R_co / R_ci))/(2 * np.pi * k_cl))
    T_clad_r = lambda r: T_co + (T_ci - T_co) * ((np.log(r / R_co))/(np.log(R_ci / R_co)))
    r_clad = np.linspace(R_ci, R_co, 500)
    T_clad_vec = T_clad_r(r_clad)
    
    # Radial tempearture in the gap:
    t_g = R_ci - R_fo
    Eq_Tfo = lambda T_fo: - T_fo + T_ci + (q1 * t_g) / (2 * np.pi * R_fo * (15.8 * 1e-4 * ((T_fo + 273.15) ** (0.79))))
    T_fo = float(fsolve(Eq_Tfo, T_ci))
    T_gap_r = lambda r: T_ci + (T_fo - T_ci) * ((np.log(r / R_ci))/(np.log(R_fo / R_ci)))
    r_gap = np.linspace(R_fo, R_ci, 500)
    T_gap_vec = T_gap_r(r_gap)
    
    # Radial temperature in the fuel pellet:
    T_fi = T_fo + q1 / (4 * np.pi * k_f)
    T_pellet_r = lambda r: T_fo + (q1 / (4 * np.pi * k_f)) * (1 - ((r / R_fo) ** (2)))
    r_pellet = np.linspace(R_fi, R_fo, 500)
    T_pellet_vec = T_pellet_r(r_pellet)
    
    # We build the arrays for the plotting
    r = np.concatenate((r_pellet[:-1], r_gap[:-1], r_clad[:-1], r_cool))
    T_r = np.concatenate((T_pellet_vec[:-1], T_gap_vec[:-1], T_clad_vec[:-1], T_cool_vec))
    T_int = [T_fi, T_fo, T_ci, T_co]
    return [T_int, r, T_r, T_cool_r, T_clad_r, T_gap_r, T_pellet_r]
    
[T_vec, r, T_r, T1_r, T2_r, T3_r, T4_r] = T_radial(q_prime0, h_Pb, k_cl, 2, p_FP, R_co, R_ci, R_fo, 0, T_cool0)
plt.figure(5)
plt.plot(r, T_r)














