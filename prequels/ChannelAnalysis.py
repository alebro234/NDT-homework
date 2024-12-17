import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
# from mpl_toolkits.mplot3d import Axes3D


class MOXFuel():
    def __init__(self, R, Ha, nz, nr):
        # external radius
        self.R = R  # m
        self.Ha = Ha
        self.nz = nz
        self.nr = nr
        self.r0 = 0  # will change with restructuring
        self.TD = 11310  # kg m^-3
        self.rho = 0.945*self.TD
        self.p = 1 - self.rho / self.TD
        self.wt_Pu = 0.29  # %
        self.x = 2 - 1.957
        self.BU = 0  # GWd/tHM
        self.T = np.zeros((nz, nr))
        self.mass = self.rho * np.pi * self.R**2 * Ha
        self.restructured = False

    def display_geom(self):
        print("\nFuel geometry")
        print(f"r in [{self.r0*1000}, {self.R*1000} mm]")
        print(f"Ha = {self.Ha*1000} mm")

    def tc(self, T):
        # fuel thermal conductivity [W m^-1 K^-1]
        T = T + 273.15  # K
        A = 0.01926 + 1.06e-6 * self.x + 2.63e-8 * self.wt_Pu
        B = 2.39e-4 + 1.37e-13 * self.wt_Pu
        D = 5.27e9
        E = 17109.5

        k0 = (1/(A + B*T) + D * np.exp(-E/T) / T**2)*(1 - self.p)**2.5

        k = 1.755 + (k0 - 1.755) * np.exp(-self.BU/128.75)
        return k

    def tc_integral(self, Ti, To, p):
        # fuel thermal conductivity integral [W m^-1]
        Ti = Ti + 273.15  # K
        To = To + 273.15  # K
        A = 0.01926 + 1.06e-6 * self.x + 2.63e-8 * self.wt_Pu
        B = 2.39e-4 + 1.37e-13 * self.wt_Pu
        D = 5.27e9
        E = 17109.5

        ko_int = (np.log((A + B*Ti)/(A + B*To))/B + D *
                  (np.exp(-E/Ti) - np.exp(-E/To))/E)*(1-p)**2.5

        k_int = 1.755*(Ti-To) + (ko_int - 1.755*(Ti-To)) * \
            np.exp(-self.BU/128.75)
        return k_int

    def T_melt(self):
        return 2964.92 + ((3147 - 364.85*self.wt_Pu - 1014.15*self.x) - 2964.92)*np.exp(-self.BU / 41.01)

    def evaluate_burnup(self, q_avg, t_irr):
        # q_lin: avg linear power [W m^-1]
        # Ha: active length [m]
        # t_irr: irradiation time [d]
        self.BU = (q_avg*self.Ha*t_irr * 1e-9) / (self.rho *
                                                  np.pi*self.R**2*self.Ha * 1e-3)  # GWd / tHM
        return self.BU

    def restructure(self, q3, T1=1800, T2=1600):
        # restructuring with average axial quantities
        self.restructured = True

        rho1 = 0.98 * self.TD
        rho2 = 0.96 * self.TD
        rho3 = self.rho

        q1 = q3 * rho1 / rho3
        q2 = q3 * rho2 / rho3

        # porosities
        p1 = 1 - rho1/self.TD
        p2 = 1 - rho2/self.TD
        p3 = 1 - rho3/self.TD

        Tfo = self.T[:, -1].mean()
        # conductivity integrals
        def I1(T0): return self.tc_integral(T0, T1, p1)
        I2 = self.tc_integral(T1, T2, p2)
        I3 = self.tc_integral(T2, Tfo, p3)

        r2 = np.sqrt(self.R**2 * (1 - 4*np.pi*I3/q3))

        # C2 = q3*r2**2*(rho2-rho3)/(4*np.pi*self.R**2*rho3)

        def f1(r1):
            return q2*(r2**2 - r1**2)/(4*np.pi*self.R**2) - q3*r2**2*(rho2-rho3)*np.log(r2/r1) - I2

        r1 = fsolve(f1, r2)

        self.r0 = np.sqrt((rho1-rho2)*r1**2/rho1 + (rho2-rho3)*r2**2/rho1)

        C1 = q1*self.r0**2/(4*np.pi*self.R**2)

        def f2(T0):
            return q1 * (r1**2 - self.r0**2) / (4*np.pi*self.R**2) - C1*np.log(r1/self.r0) - I1(T0)

        T0 = fsolve(f2, T1)

        print("\n\nRestructuring")
        print(f"r0 = {self.r0*1000}, r1 = {r1 *
              1000}, r2 = {r2*1000}, R = {self.R*1000}")
        print(f"T0 = {T0}, T1 = {T1}, T2 = {T2}, Tfo = {Tfo}")

    def plot_axial_T(self, r):
        if r > self.R or r < self.r0:
            print("r not in range in clad.plot_radial_T")
            return

        id = np.argmin(abs(r - np.linspace(self.r0, self.R, self.nr)))
        z = np.linspace(-self.Ha/2, self.Ha/2, self.nz)
        plt.figure()
        plt.plot(self.T[:, id], z*1000)
        plt.title(f"Fuel axial T [°C] @ r = {r*100} mm")
        plt.ylabel("z [mm]")

    def plot_radial_T(self, z):
        id = np.argmin(abs(z - np.linspace(-self.Ha/2, self.Ha/2, self.nz)))
        r = np.linspace(self.r0, self.R, self.nr)
        plt.figure()
        plt.plot(r*1000, self.T[id, :])
        plt.title(f"Fuel radial T [°C] @ z = {z*1000} mm")
        plt.xlabel("r [mm]")

    def plot_T_colormap(self):
        print(self.r0)
        z = np.linspace(-self.Ha/2, self.Ha/2, self.nz)
        r = np.linspace(self.r0, self.R, self.nr)
        z, r = np.meshgrid(r, z, indexing="xy")
        plt.subplot(111)
        plt.title("Fuel temperature")
        plt.pcolormesh(z*1000, r*1000, self.T, shading='auto',
                       cmap='hot')
        plt.colorbar(label='T [°C]')
        plt.xlabel('r [mm]')
        plt.ylabel('z [mm]')


class HeGap():
    def __init__(self, t, Ha, nz):
        self.t = t
        self.Ha = Ha
        self.T = np.zeros(nz)

    def tc(self, T):
        # thermal conductivity [W m^-1 K^-1]
        T = T + 273.15  # K
        return 15.8e-4 * T**0.79

    def plot_T(self):
        plt.figure("Gap axial T [°C]")
        z = np.linspace(-self.Ha/2, self.Ha/2, len(self.T))
        plt.plot(z, self.T)


class Cladding():
    def __init__(self, Ri, Ro, Ha, nz=100, nr=100):
        self.Ro = Ro
        self.Ri = Ri
        self.Tmelt = 1673 - 273.15  # °C
        self.T = np.zeros((nz, nr))
        self.Ha = Ha
        self.nz = nz
        self.nr = nr

    def display_geom(self):
        print("\nCladding geometry")
        print(f"r in [{self.Ri*1000}, {self.Ro*1000} mm]")

    def tc(self, T):
        # thermal conductivity [W m^-1 K^-1]
        return 13.95 + 0.01163 * T

    def lin_th_exp(self, T):
        # linear thermal expansion []
        return -3.101e-4 + 1.545e-5 * T + 2.75e-9 * T**2

    def rho(self, T):
        # density  [kg m^-3]
        return 7900 * (1 + self.lin_th_exp(T))**(-3)

    def plot_axial_T(self, r):
        if r > self.Ro or r < self.Ri:
            print("r not in range in clad.plot_radial_T")
            return

        id = np.argmin(abs(r - np.linspace(self.Ri, self.Ro, self.nr)))
        z = np.linspace(-self.Ha/2, self.Ha/2, self.nz)
        plt.figure()
        plt.plot(self.T[:, id], z*1000)
        plt.title(f"Cladding axial T [°C] @ r = {r*100} mm")
        plt.ylabel("z [mm]")

    def plot_radial_T(self, z):
        id = np.argmin(abs(z - np.linspace(-self.Ha/2, self.Ha/2, self.nz)))
        r = np.linspace(self.Ri, self.Ro, self.nr)
        plt.figure()
        plt.plot(r, self.T[id, :])
        plt.title(f"Cladding radial T [°C] @ z = {z*1000} mm")
        plt.xlabel("r [mm]")

    def plot_T_colormap(self):
        z = np.linspace(-self.Ha/2, self.Ha/2, self.nz)
        r = np.linspace(self.Ri, self.Ro, self.nr)
        z, r = np.meshgrid(r, z, indexing="xy")
        plt.subplot(111)
        plt.title("Cladding temperature")
        plt.pcolormesh(z*1000, r*1000, self.T, shading='auto',
                       cmap='hot')
        plt.colorbar(label='T [°C]')
        plt.xlabel('r')
        plt.ylabel('z')


class NaCoolant():
    def __init__(self, Ha, nz):
        self.T_in = 395
        self.Tb = 882  # °C
        self.Tm = 98  # °C
        self.T = np.zeros(nz)
        self.G = 0.049  # kg s^-1
        self.Ha = Ha
        self.nz = nz

    def cp(self, T):
        # isobaric specific heat [J kg^-1 K^-1]
        T = T + 273.15  # K
        return 1608 - 0.7481 * T + 3.929e-4 * T**2

    def mu(self, T):
        # dynamic viscosity [Pa s]
        T = T + 273.15  # K
        return np.exp(813.9/T - 2.530)*1e-3

    def tc(self, T):
        # thermal conductivity [W m^-1 K^-1]
        T = T + 273.15  # K
        return 110 - 0.0648 * T + 1.16e-5 * T**2

    def rho(self, T):
        # density [kg m^-3]
        T = (T * (9/5)) + 32  # °F
        return 954.1579 + T * (T * (T * 0.9667 * 1e-9 - 0.46 * 1e-5) - 0.1273534)

    def conv_HTC(self, T, Ap, Dh):
        # Nusselt number
        # u: velocity [m s^-1]
        # Dh: hydraulic diameter [m]
        u = self.G / (Ap * self.rho(T))
        self.u_max = np.max(u)
        Re = (self.rho(T) * u * Dh) / self.mu(T)
        Pr = (self.cp(T) * self.mu(T)) / self.tc(T)
        Pe = Re * Pr  # Pèclet number
        Nu = 7 + 0.025 * (Pe**(0.8))
        h = Nu * self.tc(T) / Dh
        return h

    def plot_T(self):
        z = np.linspace(-self.Ha/2, self.Ha/2, self.nz)
        plt.figure("Coolant axial T [°C]")
        plt.plot(z, self.T)


def thermal_analysis(q_lin, f, pitch, fuel, gap, clad, cool):

    # channel geometry
    Ap = 0.25*np.sqrt(3)*pitch**2 - 0.5*np.pi*clad.Ro**2
    Dh = 4*Ap / (np.pi*clad.Ro)

    # coolant
    def dTcool_dz(z, T):
        return f * q_lin(z) / (cool.G * cool.cp(T))

    z = np.linspace(-cool.Ha/2, cool.Ha/2, cool.nz)
    solution = solve_ivp(
        dTcool_dz, (-cool.Ha/2, cool.Ha/2), [cool.T_in], t_eval=z)
    cool.T = solution.y[0]
    h = cool.conv_HTC(cool.T, Ap, Dh)

    # cladding outer
    clad.T[:, -1] = cool.T + f * q_lin(z) / (2*np.pi*clad.Ro*h)

    # cladding inner

    def fun1(Tcl_i):
        return Tcl_i - clad.T[:, -1] - q_lin(z) * np.log(clad.Ro/clad.Ri) / (2*np.pi*clad.tc(0.5*(Tcl_i+clad.T[:, -1])))

    clad.T[:, 0] = fsolve(fun1, clad.T[:, -1])

    # clad T matrix
    r_clad = np.linspace(clad.Ri, clad.Ro, clad.nr)
    for i in range(clad.nz):
        clad.T[i, :] = clad.T[i, -1] + (clad.T[i, 0] - clad.T[i, -1]) * \
            np.log(r_clad/clad.Ro) / np.log(clad.Ri/clad.Ro)

    # gap
    def fun2(Tfo):
        return Tfo - clad.T[:, 0] - q_lin(z) * gap.t / (2*np.pi*fuel.R*gap.tc(0.5*(Tfo + clad.T[:, 0])))
    Tfo = fsolve(fun2, clad.T[:, 0])
    gap.T = 0.5*(Tfo + clad.T[:, 0])

    # fuel

    def dTf_dr(r, T):
        return -q_lin(z) * r / (2*np.pi*fuel.tc(T)*fuel.R**2)

    r_fuel = np.linspace(fuel.R, 0, fuel.nr)
    sol = solve_ivp(dTf_dr, (fuel.R, 0), Tfo, t_eval=r_fuel)
    fuel.T = np.fliplr(sol.y)  # orient from 0 to fuel.R

    return fuel, gap, clad, cool


def thermal_expansion(fuel, gap, clad):
    pass


# WHAT VARIES ALONG Z????
