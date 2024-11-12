import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import fsolve
import Correlations as cr

class Fuel:
    # fuel properties class
    # T: temperature [°C]
    # x: deviation from O/M stoichiometry
    # p: porosity [%]
    # wt_Pu: Pu concentration in weigth [%]
    # BU: burnup [GWd/tHM]
    def __init__(self, x, p, wt_Pu, BU):
        self.x = x
        self.p = p
        self.wt_Pu = wt_Pu
        self.BU = BU

    def thermal_cond(self, T):
        # fuel thermal conductivity [W m^-1 K^-1]
        T = T + 273.15 # K
        A = 0.01926 + 1.06e-6 * self.x + 2.63e-8 * self.wt_Pu
        B = 2.39e-4 + 1.37e-13 * self.wt_Pu
        D = 5.27e9
        E = 17109.5

        k0 = ( 1/(A + B*T) + D/T**2 * np.exp(-E/T)  )*(1 - self.p)**2.5

        k = 1.755 + (k0 - 1.755) * np.exp(-self.BU/128.75)
        return k

    def T_melt(self):
        # fuel melting temperature [°C]
        return 2964.92 + (3147 - 364.85 * self.wt_Pu - 1014.15 * self.x - 2964.92 ) * np.exp( -self.BU/41.01 ) - 273.15
    

def k_fuel(T, x, p, wt_Pu, BU):
    # fuel thermal conductivity [W m^-1 K^-1]
    T = T + 273.15 # K
    A = 0.01926 + 1.06e-6 * x + 2.63e-8 * wt_Pu
    B = 2.39e-4 + 1.37e-13 * wt_Pu
    D = 5.27e9
    E = 17109.5

    k0 = ( 1/(A + B*T) + D/T**2 * np.exp(-E/T)  )*(1 - p)**2.5

    k = 1.755 + (k0 - 1.755) * np.exp(-BU/128.75)
    return k



def T_cyl(Ti, To, Ri, Ro, r):
    # Temperature profile in hollow cylinder, no power generation

    return To + (Ti - To) * np.log(r/Ro) / np.log(Ri/Ro)



def thermal_analysis(q_lin, Rco, Rci, Rfo, fuel, nz = 100, nr = 50):
    # thermal analysis of fuel element at fixed geometry (no restructuring)
    # q_lin: (vector) linear thermal power along z [W m^-1] alredy accounts for
    #        channel shape
    # Rco: cladding outer radius [m]
    # Rci: cladding inner radius [m]
    # Rfo: fuel outer radius [m]
    # fuel: fuel properties class
    # Outputs
    # z, r_fuel, r_clad: (2d vector) coordinates meshes
    # T_fuel: (2d vector) fuel temperature evaluated in z, r_fuel
    # T_clad: (2d vector) cladding temperature evaluated in z, r_clad
    # Tgap: (vector) gap temperature along z
    
    # geometric data
    pitch = 8.275e-3 # m
    Ha = 0.85 # m
    Ap = 0.25*np.sqrt(3)*pitch**2 - 0.5*np.pi*Rco**2
    Dh = 4*Ap / (np.pi*Rco)

    # coolant data
    G = 0.049 # kg s^-1
    Tc_in = 395 # °C
    cp_c = lambda T: 1608 - 0.7481 * (T+273.15) + 3.929e-4 * (T+273.15) # J kg^-1 K^-1
    def rho_c(T):
        T = T * 9/5 + 32 # °F
        return 954.1579 + T * ( T * (T*0.9667e-9 - 0.46e-5) - 0.1273534 )   # kg m^-3
    mu_c = lambda T: 0.001 * np.exp( 813.9/T - 2.530 ) # Pa s
    k_c = lambda T: 110 - 0.0648 * (T+273.15) + 1.16e-5 * (T+273.15)**2     #W m^-1 K^-1

    # cladding data
    k_cl = lambda T: 13.95 + 0.01163 * T
    
    # gap data
    k_He = lambda T: 15.8e-4 * (T+273.15)**0.79
    t_gap = Rci - Rfo



                                    ## coolant axial temperature ##
    def dTdz(z, T):
        return q_lin(z) / ( G * cp_c(T) )

    z = np.linspace(-Ha/2, Ha/2, nz)
    solution = solve_ivp(dTdz, (-Ha/2, Ha/2), [Tc_in], t_eval = z)
    Tcool_z = solution.y[0]

    rho_c_z = rho_c(Tcool_z)
    k_c_z = k_c(Tcool_z)
    cp_c_z = cp_c(Tcool_z)
                                ## convective HTC
    v_z = G / (rho_c_z * Ap)
    Pe_z = rho_c_z * v_z * Dh * cp_c_z / k_c_z
    Nu_z = 7 + 0.025 * Pe_z**0.8
    h_z = Nu_z * k_c_z / Dh

                                ## Cladding external 
    Tcl_o_z = Tcool_z + q_lin(z) / (2*np.pi*Rco*h_z)
    
                                ## Cladding internal
    def f(Tcl_i_z):
        return Tcl_i_z - Tcl_o_z - q_lin(z) * np.log(Rco/Rci) / ( 2*np.pi*k_cl( 0.5*(Tcl_i_z+Tcl_o_z) ) )

    Tcl_i_z = fsolve(f, Tcl_o_z)

                                ## Gap
    def f(Tfo_z):
        return Tfo_z - Tcl_i_z - q_lin(z) * t_gap / ( 2*np.pi*Rfo*k_He(0.5*(Tfo_z + Tcl_i_z)) )
        
    Tfo_z = fsolve(f, Tcl_i_z)

    Tgap = 0.5*(Tfo_z + Tcl_i_z)


    

                                ## fuel
    def dTfdr(r, T):
        return  -q_lin(z) * r / ( 2*np.pi*fuel.thermal_cond(T)*Rfo**2 )

    r_fuel = np.linspace(Rfo, 0, nr)
    sol = solve_ivp(dTfdr, (Rfo, 0), Tfo_z, t_eval = r_fuel)
    T_fuel = sol.y


    # Temperature mesh building
    T_clad = np.zeros((nz, nr))
    r_clad = np.linspace(Rci, Rco, nr)
    
    for i in range(nz):
        T_clad[i,:] = T_cyl( Tcl_i_z[i], Tcl_o_z[i], Rci, Rco, r_clad )

    # orient from 0 to Rfo 
    r_fuel = np.flip(r_fuel)
    T_fuel = np.fliplr(T_fuel)

    z, r_fuel = np.meshgrid(np.linspace(-Ha/2, Ha/2, nz), r_fuel, indexing="ij")
    z, r_clad = np.meshgrid(np.linspace(-Ha/2, Ha/2, nz), r_clad, indexing="ij")

    return z, r_fuel, T_fuel, r_clad, T_clad, Tgap


def fuel_thermal_expansion(T, Tref, Rfo):

    alpha = 1.2e-5 # °C^-1

    return  Rfo * ( 1 + alpha*(T - Tref) )


def clad_thermal_expansion(Tci, Tco, Rci, Rco):

    eps_th = lambda T: -3.101e-4 + 1.545e-5*T + 2.75e-9*T**2

    Rci_new = Rci * ( 1 + eps_th(Tci) )
    Rco_new = Rco * ( 1 + eps_th(Tco) )
    
    return Rci_new, Rco_new

    
def hot_geometry(q_lin, Rco_cold, Rci_cold, Rfo_cold, fuel):

    Tref = 25
    
    z, rf, Tf, rcl, Tcl, Tgap = thermal_analysis(q_lin, Rco_cold, Rci_cold, Rfo_cold, fuel)

    Tfo = np.max(Tf[:,-1])
    Tci = np.max(Tcl[:,0])
    Tco = np.max(Tcl[:,-1])

    Rfo = Rfo_cold
    Rci = Rci_cold
    Rco = Rco_cold
  
    toll = 1e-9
    err = toll+1

    while err >= toll:
        Rfo_new = fuel_thermal_expansion(Tfo, Tref, Rfo_cold)
        Rci_new, Rco_new = clad_thermal_expansion(Tci, Tco, Rci_cold, Rco_cold)

        z, rf, Tf, rcl, Tcl, Tgap = thermal_analysis(q_lin, Rco_new, Rci_new, Rfo_new, fuel)

        Tfo_new = np.max(Tf[:,-1])
        Tci_new = np.max(Tcl[:,0])
        Tco_new = np.max(Tcl[:,-1])

        err = np.sqrt( (Rfo - Rfo_new)**2 + (Rco - Rco_new)**2 + (Rci - Rci_new)**2 )

        Tfo = Tfo_new
        Tco = Tco_new
        Tci = Tci_new

        Rfo = Rfo_new
        Rci = Rci_new
        Rco = Rco_new

    return Rco, Rci, Rfo


def restructuring(q3, TD, rho1, rho2, rho3, R, x, wt_Pu, BU, Tfo, T1=1800, T2=1600):
    # HP: uniform x, wt_Pu, BU in pellet
    # -> power change due to densification only
    # -> k_i change due to porosity only
    # -> explicit problem for r1, r2; implicit for r0, T0

    q1 = q3 * rho1 / rho3
    q2 = q3 * rho2 / rho3

    # porosities
    p1 = 1 - rho1/TD
    p2 = 1 - rho2/TD
    p3 = 1 - rho3/TD

    # conductivity integrals
    I1 = lambda T0: quad(lambda T: k_fuel(T,x,p1,wt_Pu,BU), T1,T0)[0]
    I2 = quad(lambda T: k_fuel(T,x,p2,wt_Pu,BU),T2,T1)[0]
    I3 = quad(lambda T: k_fuel(T,x,p3,wt_Pu,BU),Tfo,T1)[0]

    print(I3)

    r2 = np.sqrt( R**2 * ( 1 - 4*np.pi*I3/q3 ) )

    def f(r1):
        return q2*(r2**2 - r1**2)/(4*np.pi*R**2) - q3*r2**2*(rho2-rho3)*np.log(r2/r1) / (2*np.pi*R**2*rho3) - I2

    r1 = fsolve(f, r2)
    
    r0 = np.sqrt( (rho1-rho2)*r1**2/rho1 + (rho2-rho3)*r2**2/rho1 )

    def f(T0):
        return q1*(r1**2 - r0**2)/(4*np.pi*R**2) - q3*r0**2*np.log(r1/r0) / (2*np.pi*R**2) - I1(T0)
        

    T0 = fsolve(f, T1)

    return r0[0], r1, r2, T0[0]
