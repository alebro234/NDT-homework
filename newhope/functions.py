import numpy as np
from scipy.integrate._ivp.radau import T
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

def tc_fuel(T, x, y_Pu, p, Bu):
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


def tc_fuel_integral(Ti, To, x, y_Pu, p, Bu):
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


def K_fuel_Trz(y_Pu, x, p, BU):
    # builds lambda function k = k(T,r,z)
    # y_Pu(r): radial Pu fraction
    # x: deviation from O/M stoichiometry
    # p: porosity of region
    # BU(r,z): local burnup [GWd tHM^-1]

    K = lambda T,r,z: tc_fuel(T, x, y_Pu(r), p, BU(r,z))
    return K
    

def Pu_redistribution(R, R0, r_star = None):
    # total Pu weight percent [-]
    # R: fuel outer radius [m]
    # R0: inner void radius [m]
    # r_star: parameter from mass balance [m]
    #
    
    D = 0.01
    alpha = 10
    if r_star is None:
        I1 = 1/(alpha) * ( (R0/R)*np.exp(-alpha*R0/R) - np.exp(-alpha) ) + 1/(alpha**2) * ( np.exp(-alpha*R0/R) - np.exp(-alpha) )
        I2 = 1/(2*alpha) * ( (R0/R)*np.exp(-2*alpha*R0/R) - np.exp(-2*alpha) ) + 1/(4*alpha**2) * ( np.exp(-2*alpha*R0/R) - np.exp(-2*alpha) )

        r_star = R * np.log(2*I1/I2)/alpha
        y_Pu = lambda r: 1 + D*( np.exp(-2*alpha*(r-r_star)/R) - 2*np.exp(-alpha*(r-r_star)/R) )

        # y_Pu = lambda r_star: y0 + y0*D*( np.exp(-2*alpha*(r-r_star)/R) - 2*np.exp(-alpha*(r-r_star)/R) )
        # def fun(r_star):
        #     return 2*np.pi*Ha*np.trapz(r*y_Pu(r_star), r) - wt_Pu
        # r_star = fsolve(fun, R/2)[0]
        # y_Pu = y_Pu(r_star)
    else:
        y_Pu = lambda r: 1 + D*( np.exp(-2*alpha*(r-r_star)/R) - 2*np.exp(-alpha*(r-r_star)/R) )
    return y_Pu

def burnup_density(q_vol, t_irr, Rf, R0):
    # q_vol(r,z): volumetric heat rate [W m^-3]
    # t_irr: irradiation time [d]
    rho_f = 11310 * 0.945  # fuel density [kg m^-3]
    V_f = np.pi*(Rf**2 - R0**2) * 0.850
    bu = lambda r,z: 1e-6 * q_vol(r,z) * t_irr / (rho_f * V_f) # GWd tHm^-3 m^-3
    return bu

def cp_cool(T):
    # coolant isobaric specific heat [J kg^-1 K^-1]
    T = T + 273.15  # K
    return 1608 - 0.7481 * T + 3.929e-4 * T**2

def mu_cool(T):
    # coolant dynamic viscosity [Pa s]
    T = T + 273.15  # K
    return np.exp(813.9/T - 2.530)*1e-3

def tc_cool(T):
    # coolant thermal conductivity [W m^-1 K^-1]
    T = T + 273.15  # K
    return 110 - 0.0648 * T + 1.16e-5 * T**2

def rho_cool(T):
    # coolant density [kg m^-3]
    T = (T * (9/5)) + 32  # °F
    return 954.1579 + T * (T * (T * 0.9667 * 1e-9 - 0.46 * 1e-5) - 0.1273534)


def tc_clad(T):
    # cladding thermal conductivity [W m^-1 K^-1]
    return 13.95 + 0.01163 * T


def tc_gap(T):
    # gap thermal conductivity [W m^-1 K^-1]
    T = T + 273.15  # K
    return 15.8e-4 * T**0.79

def thermal_effects(q_lin, Rco, Rci, Rf, R0 = 0, nr = 100):
    # q_lin(z): linear heat rate [W m^-1]
    # Rco: cladding outer radius [m]
    # Rci: cladding inner radius [m]
    # Rfo: fuel outer radius [m]
    # R0: inner void radius [m]
    # nr: number of radial points for fuel

    # channel geometry
    f = 0.5
    pitch = 8.275e-3  # m
    Ap = 0.25*np.sqrt(3)*pitch**2 - 0.5*np.pi*Rco**2
    Dh = 4*Ap / Rco
    Ha = 0.85
    # number of axial points
    nz = len(Rco)
    z = np.linspace(-Ha/2, Ha/2, nz)

    # coolant
    G = 0.049
    def dTcool_dz(z, T):
        return f * q_lin(z) / (G * cp_cool(T))

    solution = solve_ivp(dTcool_dz, (-Ha/2, Ha/2), y0=395*np.ones(nz), t_eval=z)
    T_cool = solution.y[0]

    # convective HTC
    u = G / (Ap * rho_cool(T_cool))
    # u_max = np.max(u)
    Re = (rho_cool(T_cool) * u * Dh) / mu_cool(T_cool)
    Pr = (cp_cool(T_cool) * mu_cool(T_cool)) / tc_cool(T_cool)
    Pe = Re * Pr  # Pèclet number
    Nu = 7 + 0.025 * (Pe**(0.8))
    h = Nu * tc_cool(T_cool) / Dh

    # cladding outer
    T_co =  T_cool + f * q_lin(z) / (2 * np.pi * Rco * h)

    # cladding inner
    def fun1(T_ci):
        return T_ci - T_co - q_lin(z) * np.log(Rco/Rci) / (2 * np.pi * tc_clad(0.5*(T_ci + T_co)))

    T_ci = fsolve(fun1, T_co)

    # gap
    t_gap = Rci - Rf
    def fun2(T_fo):
        return T_fo - T_ci - q_lin(z) * t_gap / (2 * np.pi *Rf * tc_gap(0.5*(T_fo + T_ci)))

    T_fo = fsolve(fun2, T_ci)
    T_gap = 0.5 * (T_fo + T_ci)

    # fuel

    Rf_avg = np.mean(Rf)
    R0_avg = np.mean(R0)
    # Pu redistribution
    y0 = 0.29
    q_r = Pu_redistribution(Rf_avg, R0_avg)
    # volumetric heat power gen [W m^-3]
    q_vol = lambda r, z: q_lin(z) * q_r(r) / (np.pi*(Rf_avg**2 - R0_avg**2))
    # radial plutonium fraction
    y_Pu = lambda r: y0 * q_r(r)

    bu = burnup_density(q_vol, 7, Rf_avg, R0_avg) # GWd tHM^-3 m^-3
    dr = (Rf_avg - R0_avg) / nr
    dz = z[1] - z[0]
    # local burnup in cell (r, z)
    BU = lambda r,z: 2*np.pi*r*bu(r,z)*dr*dz # Gwd tHM^-3

    if R0 == 0:
        # as fabricated fuel
        T_f = np.zeros((nz, nr))
        kf = K_fuel_Trz(y_Pu, 2 - 1.975, 1 - 0.945, BU)
        for i in range(nz):
            r_fuel = np.linspace(Rf[i], 0, nr)

            def dTf_dr(r, T):
                return -q_vol(r, z[i]) * r / (2 * kf(T, r, z[i]))

            solution = solve_ivp(
                dTf_dr, 
                (Rf[i], 0), 
                y0=[T_fo[i]], 
                t_eval=r_fuel
            )
            T_f[i, :] = np.flip(solution.y[0])
        else:
            # restructured fuel

            # restructuring radii
            R1 = np.zeros(nz)
            R2 = np.zeros(nz)

            # densities and porosities 
            TD = 11310  # fuel theoretical density [kg m^-3]
            rho1 = 0.98 * TD
            rho2 = 0.96 * TD
            rho3 = 0.945 * TD # = rho_f

            p1 = 1 - rho1 / TD
            p2 = 1 - rho2 / TD
            p3 = 1 - rho3 / TD

            # integrate in region 3 up to T2 = 1600°C and find R2(z)
            T2 = 1600
            kf = K_fuel_Trz(y_Pu, 2 - 1.975, p3, BU)


            # integrate in region 2 up to T1 = 1800°C and find R1(z)
            T2 = 1600
            kf = K_fuel_Trz(y_Pu, 2 - 1.975, p3, BU)

            # R0 from mass balance

            # integrate in region 1 up to R0 and find T0(z)





    return T_cool, T_co, T_ci, T_gap, T_f


def plot_axial_T(T, r, r0, R, n_r, Ha):
    if r > R or r < r0:
        print("r not in range in plot_axial_T")
        return

    # Find the radial index corresponding to the given r
    radial_positions = np.linspace(r0, R, n_r)
    radial_index = np.argmin(abs(r - radial_positions))

    # Define the axial positions
    z = np.linspace(-Ha / 2, Ha / 2, T.shape[0])
    
    # Plot the temperature profile along the axial direction
    plt.figure()
    plt.plot(T[:, radial_index], z * 1000)
    plt.title(f"Fuel axial T [°C] @ r = {r * 100} mm")
    plt.xlabel("T [°C]")
    plt.ylabel("z [mm]")
    plt.grid()

def plot_radial_T(T, z, r0, R, n_r, Ha):
    # Find the axial index corresponding to the given z
    axial_positions = np.linspace(-Ha / 2, Ha / 2, T.shape[0])
    axial_index = np.argmin(abs(z - axial_positions))

    # Define the radial positions
    r = np.linspace(r0, R, n_r)

    # Plot the temperature profile along the radial direction
    plt.figure()
    plt.plot(r * 1000, T[axial_index, :])
    plt.title(f"Fuel radial T [°C] @ z = {z * 1000} mm")
    plt.xlabel("r [mm]")
    plt.ylabel("T [°C]")
    plt.grid()

def plot_T_colormap(T, r0, R, n_r, Ha):
    # Define the axial and radial positions
    z = np.linspace(-Ha / 2, Ha / 2, T.shape[0])
    r = np.linspace(r0, R, n_r)
    r, z = np.meshgrid(r, z, indexing="xy")

    # Plot the temperature colormap
    plt.figure()
    plt.title("Fuel temperature")
    plt.pcolormesh(r * 1000, z * 1000, T, shading='auto', cmap='hot')
    plt.colorbar(label='T [°C]')
    plt.xlabel('r [mm]')
    plt.ylabel('z [mm]')
    plt.grid()

