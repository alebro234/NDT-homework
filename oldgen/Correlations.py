"""

Correlations for the NTD project Homework:
follows a list of correlations to be employed in the design of the reactor component, subdivided into
the three main physical subsystems (cool, clad, gas, fuel) and the kind of property correlated:

"""
import numpy as np
# Coolant:
    # Thermal properties:
    
# Specific heat capacity    
def cp_cool(T_C):
    T_K = T_C + 273.15 # conversion to Kelvin 
    cp_c = 1608 - 0.7481 * T_K + 3.929 * 1e-4 * (T_K**2) #[J/kg*K]
    return cp_c

# Density
def rho_cool(T_C):
    T_F = (T_C * (9/5)) + 32 # conversion to Fahrenheit
    rho_c = 954.1579 + T_F * (T_F * (T_F * 0.9667 * 1e-9 - 0.46 * 1e-5) - 0.1273534) #[kg/m^3]
    return rho_c

# Dynamic viscosity
def mu_cool(T_C):
    T_K = T_C + 273.15 # conversion to Kelvin
    mu_c = (np.exp((813.9/T_K) - 2.530))*1e-3 #[Pa*s]
    return mu_c

# Thermal conductivity
def k_cool(T_C):
    T_K = T_C + 273.15 # conversion to Kelvin
    k_c = 110 - 0.0648 * T_K + 1.16 * 1e-5 * (T_K**2) #[W/m*K]
    return k_c

# Nusselt number
def Nu_cool(Pe):
    Nu_c = 7 + 0.025 * (Pe**(0.8))
    return Nu_c

# Péclet number
def Pe_cool(T_C, u, D_h):
    Re_c = (rho_cool(T_C) * u * D_h) / mu_cool(T_C)
    Pr_c = (cp_cool(T_C) * mu_cool(T_C)) / k_cool(T_C)
    Pe_c = Re_c* Pr_c
    return Pe_c


# Cladding:
    # Termal and mechanical properties: 

# Linear thermal expansion
def epsilon_cl(T_C):
    epsilon_th = -3.101e-4 + 1.545e-5 * T_C + 2.75e-9 * T_C**2
    return epsilon_th

# Density
def rho_cl(epsilon_th):
    rho_cl = 7900 * (1 + epsilon_th)**-3 # kg/m^3
    return rho_cl

# Thermal conductivity 
def k_cl(T_C):
    k_cl = 13.95 + 0.01163 * T_C # W/m*K
    return k_cl

# Young's modulus
def young_mod_cl(T_C):
    y_module = 202.7 - 0.08167 * T_C # GPa
    return y_module

# Poisson's ratio
def poisson_ratio_cl(T_C):
    poiss_ratio = 0.277 + 6e-5 * T_C
    return poiss_ratio

# Yield stress at 0.2% strain
def sigma_y_02_cl(T_C):
    if T_C <= 600:
        sigma_y_02 = 555.5 - 0.25 * T_C # MPa
        return sigma_y_02
    elif 600 < T_C < 1000:
        sigma_y_02 = 405.5 - 0.775 * (T_C - 600) # MPa
        return sigma_y_02
    else:
        sigma_y_02 = 345.5 - 0.25 * T_C # MPa
        return sigma_y_02

# Ultimate tensile strength (UTS)
def uts_cl(T_C):
    if T_C <= 600:
        uts = 700 - 0.3125 * T_C # MPa
        return uts
    elif 600 < T_C < 1000:
        uts = 512.5 - 0.969 * (T_C - 600) # MPa
        return uts
    else:
        uts = 437.5 - 0.3125 * T_C # MPa
        return uts

# Rupture strain
def eps_rupt(T_C):
    eps_rupt = 8 + 4.74e-3 * (T_C-500) + 6.2e-5 * (T_C-500)**2 # %  # %
    return eps_rupt

# Void swelling
def void_swelling(T_C, phi):
    delta_v_over_v = 1.5e-3 * np.exp(-2.5 * ((T_C - 450) / 100)) * ((phi / 1e22)**2.75) # %
    return delta_v_over_v


# Gap:
    # Thermal properties:

# Thermal conductivity
def k_gap(T_C):
    T_k = T_C + 273.15 
    k_gap = 15.8e-4 * T_k**0.79 # W/m*K
    return k_gap


# Fuel:
    # Thermal and mechanical properties:

# Melting temperature
def t_melt_fuel(Pu, Bu):
    x =  2 - 1.957
    T_melt_f = 2964.92 + ((3147 - 364.85 * Pu - 1014.15 * x) - 2964.92) * np.exp(- Bu / 41.01) # K
    return T_melt_f 

# Thermal conductivity 
def k_fuel(Bu, p, Pu, T_C):
    x =  2 - 1.957
    A = 0.01926 + (1.06 * 1e-6 * x) + (2.63 * 1e-8 * Pu)    
    B = (2.39 * 1e-4) + (1.37 * 1e-13 * Pu)
    D = 5.27 * 1e9
    E = 17109.5
    T_K = T_C + 273.15
    k_0 = ((1 / (A + (B * T_K))) + ((D / T_K**2) * np.exp(-(E / T_K)))) * ((1 - p)**2.5) # W/m*K
    k_fuel = 1.755 + ((k_0 - 1.755) * np.exp(-(Bu / 128.75))) # W/m*K
    return k_fuel

# Young's modulus
def young_mod_f(T_C, p):
    y_module_fuel = (22.43e4 - 31.19 * T_C) * (1 - 2.6 * p) # MPa
    return y_module_fuel

# Swelling strain
def swelling_strain(Bu):
    eps_swe_tot_fuel = 0.07 * Bu
    return eps_swe_tot_fuel 

# Burnup
def burnup(q_prime, L_act, R, rho, t_irr):
    # togliere ossigeno alla massa del fuel (?)
    Bu = (q_prime * L_act * 1e-9 * t_irr) / (L_act * np.pi * R**2 * rho * 1e-3) # GWd/tHM
    return Bu

    

    