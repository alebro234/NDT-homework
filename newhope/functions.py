import numpy as np
from scipy.optimize import fsolve

def tc(T, x, y_Pu, p, Bu):
    # fuel thermal conductivity [W m^-1 K^-1]
    # T: temperature [°C]
    # x: O/M ratio [-]
    # y_Pu: Pu weight percent [-]
    # p: porosity [-]
    # Bu: burnup [GWd/tHM]

    T = T + 273.15  # K
    A = 0.01926 + 1.06e-6 * x + 2.63e-8 * y_Pu
    B = 2.39e-4 + 1.37e-13 * y_Pu
    D = 5.27e9
    E = 17109.5

    k0 = (1/(A + B*T) + D * np.exp(-E/T) / T**2)*(1 - p)**2.5

    k = 1.755 + (k0 - 1.755) * np.exp(-Bu/128.75)
    return k


def tc_integral(Ti, To, x, y_Pu, p, Bu):
    # fuel thermal conductivity integral [W m^-1]
    # Ti, To: boundary temperatures [°C]
    # x: O/M ratio [-]
    # y_Pu: Pu weight percent [-]
    # p: porosity [-]
    # Bu: burnup [GWd/tHM]
    Ti, To = max(Ti, To), min(Ti, To)
    Ti = Ti + 273.15  # K
    To = To + 273.15  # K
    A = 0.01926 + 1.06e-6 * x + 2.63e-8 * y_Pu
    B = 2.39e-4 + 1.37e-13 * y_Pu
    D = 5.27e9
    E = 17109.5

    ko_int = (np.log((A + B*Ti)/(A + B*To))/B + D *
                (np.exp(-E/Ti) - np.exp(-E/To))/E)*(1-p)**2.5

    k_int = 1.755*(Ti-To) + (ko_int - 1.755*(Ti-To)) * np.exp(-Bu/128.75)
    return k_int

def tc_avg(Ti, To, x, y_Pu, p, Bu):
    # fuel thermal conductivity average [W m^-1 K^-1]
    # Ti, To: boundary temperatures [°C]
    # x: O/M ratio [-]
    # y_Pu: Pu weight percent [-]
    # p: porosity [-]
    # Bu: burnup [GWd/tHM]
    Ti, To = max(Ti, To), min(Ti, To)
    k_int = tc_integral(Ti, To, x, y_Pu, p, Bu)
    k_avg = k_int/(Ti - To)
    return k_avg

def radial_y_Pu(y0, R, R0, r_star = None):
    # total Pu weight percent [-]
    # R: fuel outer radius [m]
    # R0: inner void radius [m]
    # r_star: parameter from mass balance [m]
    #
    
    D = 0.01
    alpha = 10
    Ha = 0.85
    r = np.linspace(R0, R, 100)

    if r_star is None:
        I1 = 1/(alpha) * ( (R0/R)*np.exp(-alpha*R0/R) - np.exp(-alpha) ) + 1/(alpha**2) * ( np.exp(-alpha*R0/R) - np.exp(-alpha) )
        I2 = 1/(2*alpha) * ( (R0/R)*np.exp(-2*alpha*R0/R) - np.exp(-2*alpha) ) + 1/(4*alpha**2) * ( np.exp(-2*alpha*R0/R) - np.exp(-2*alpha) )

        r_star = R * np.log(2*I1/I2)/alpha
        y_Pu = y0 + y0*D*( np.exp(-2*alpha*(r-r_star)/R) - 2*np.exp(-alpha*(r-r_star)/R) )

        # y_Pu = lambda r_star: y0 + y0*D*( np.exp(-2*alpha*(r-r_star)/R) - 2*np.exp(-alpha*(r-r_star)/R) )
        # def fun(r_star):
        #     return 2*np.pi*Ha*np.trapz(r*y_Pu(r_star), r) - wt_Pu
        # r_star = fsolve(fun, R/2)[0]
        # y_Pu = y_Pu(r_star)
    else:
        y_Pu = y0 + y0*D*( np.exp(-2*alpha*(r-r_star)/R) - 2*np.exp(-alpha*(r-r_star)/R) )
    return r, y_Pu, r_star

def y_Pu_avg(y_Pu, r1, r2):
    r = np.linspace(r1, r2, 100)
    return np.trapz(y_Pu, r)/(r2 - r1)
    


