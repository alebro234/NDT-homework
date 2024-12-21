import numpy as np
import nuclei_func as nu
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import functions as fn

    
def main(t_gap, V_plenum):
    # fast neutron flux ( E > 100 keV ) [n cm^-2 s^-1]
    neutron_flux = 6.1e15

    # number of axial nodes
    nz = 100
    # number of radial nodes in the fuel
    nr = 100
    Ha = 0.85

    # cold geometry
    Rf = 5.42e-3/2 * np.ones(nz)
    Rci = Rf + t_gap
    Rco = 6.55e-3/2 * np.ones(nz)
    R0 = np.zeros(nz)

    V_fuel = np.pi*Ha*Rf[0]**2
    TD = 11310
    rho_f = 0.945 * TD

    # irradiation time [d]
    t_irr = 365

    # axial linear power evaluation
    q_0 = 38.7e3  # W m^-1
    c = np.array([0.572, 0.737, 0.868, 0.958, 1, 0.983, 0.912, 0.802, 0.658, 0.49])
    z_fit = np.array([42.5, 127.5, 212.5, 297.5, 382.5, 467.5,552.5, 637.5, 722.5, 807.5])*0.001 - Ha/2 # m
    q_fit = q_0 * c
    polyn = np.poly1d(np.polyfit(z_fit, q_fit, q_fit.size-1))
    z = np.linspace(-Ha/2, +Ha/2, nz)
    def q_lin(z): return polyn(z)  # q = q(z) [W m^-1]
    # total rod power [W]
    Q = np.trapz(q_lin(z), z)
    # burnup [GWd tHM^-1]
    BU = 1e-6 * t_irr * Q / (V_fuel * rho_f)

    print(f"Total rod power: {Q:.3f} W")
    print(f"Burnup at {t_irr} d: {BU:.3f} GWD/tHM")


    # 1. Thermal analysis at cold geometry
    Tcool, Tco, Tci, Tgap, Tfo = fn.thermal_up_to_fuel(q_lin, Rco, Rci, Rf)
    Tfuel, Rfuel = fn.thermal_in_fuel(q_lin, BU, Rf, Tfo, nr, restructured=False)
    # fn.plot_T_colormap(Tfuel, 0, Rf[0], nr, Ha)
    fn.plot_axial_temperatures(z, Tcool, Tco, Tci, Tgap, Tfuel)

    print("\nCold geometry")
    print(f"Rf: {Rf[0]*1000} mm, Rci: {Rci[0]*1000} mm, Rco: {Rco[0]} mm")
    print("Max temperatures [°C]")
    print(f"fuel: {np.max(Tfuel):.2f}, gap: {np.max(Tgap):.2f}, cladding: {np.max(Tci):.2f}, coolant: {np.max(Tcool):.2f}")

    # 2. Expansion and thermal analysis at hot geometry

    # fuel linear thermal expansion
    eps_th_f = lambda T: 1.2e-5 * (T - 25)
    # clad linear thermal expansion
    eps_th_cl = lambda T: -3.101e-4 + 1.545e-5 * T + 2.75e-9 * T**2 
    # Volumetric cladding swelling due to irradiation 
    eps_irr_clad = lambda T: 1.5e-5 * np.exp(-0.25 * (T - 450)**2) * (neutron_flux * t_irr * 24 * 3600 / 1e22)**2.75
    # Volumetric fuel swelling due to irradiation
    eps_irr_fuel = 0.01 * 0.07 * BU

    toll = 1e-9
    err = toll+1
    max_iter = 500
    n = 0
    Rf_prev = np.copy(Rf)
    Rci_prev = np.copy(Rci)
    Rco_prev = np.copy(Rco)
    while err > toll and n <= max_iter:
        Rf_hot  = Rf * (1 + eps_th_f(Tfo)   + (1/3)*eps_irr_fuel)
        Rco_hot = Rco * (1 + eps_th_cl(Tco))
        Rci_hot = Rci * (1 + eps_th_cl(Tci) )

        # thermal analysis at new geometry
        Tcool, Tco, Tci, Tgap, Tfo = fn.thermal_up_to_fuel(q_lin, Rco_hot, Rci_hot, Rf_hot)

        err = np.max([
            np.abs(np.mean(Rf_hot) - np.mean(Rf_prev)),
            np.abs(np.mean(Rci_hot) - np.mean(Rci_prev)),
            np.abs(np.mean(Rco_hot) - np.mean(Rco_prev))
        ])
        Rf_prev = np.copy(Rf_hot)
        Rci_prev = np.copy(Rci_hot)
        Rco_prev = np.copy(Rco_hot)

        n += 1
        
    Tfuel, Rfuel = fn.thermal_in_fuel(q_lin, BU, Rf_hot, Tfo, nr, restructured=False)

    print(f"\nHot geometry converged in {n} iterations")
    print(f"Rf: {np.max(Rf_hot)*1000:.2f} mm, Rci: {np.max(Rci_hot)*1000:.2f} mm, Rco: {np.max(Rco_hot)*1000:.2f} mm")
    print("Max temperatures [°C]")
    print(f"fuel: {np.max(Tfuel):.2f}, gap: {np.max(Tgap):.2f}, cladding: {np.max(Tci):.2f}, coolant: {np.max(Tcool):.2f}")
    
    # Plot expanded radii
    fn.plot_expanded_radii(z, Rf_hot, Rci_hot, Rco_hot, Rf, Rci, Rco)
    # plot_axial_temperatures(z, Tcool, Tco, Tci, Tgap, Tfuel)

    Tf_max = np.max(Tfuel)

    # 3. Restructuring

    Tfuel, Rfuel = fn.thermal_in_fuel(q_lin, BU, Rf_hot, Tfo, nr, restructured=True)
    fn.plot_Tfuel(Tfuel, Rfuel, Ha, restructured=True)

    # 4. FGR and plenum dimensioning

    R_grain = 1e-5 #[m]
    V_grain = 4/3 * np.pi * R_grain**3
    y_Fg = 0.30

    # fission rate [# s^-1]
    F_rate = fn.fission_rate(neutron_flux, rho_f)
    print(f"\nFission rate: {F_rate}")

    # fission gas production [atoms]
    p = lambda t: y_Fg * F_rate * t * V_grain

    # effective diffusion coefficient [m^2 s^-1]
    Tf_max_K = Tf_max + 273.15 #[K]
    D = 5 * 1e-8 * np.exp(- 40262 / Tf_max_K)
    print(f"Deff: {D} m^2 s^-2")

    # fission gas retention in the grain
    g = lambda t: y_Fg * F_rate * 8 * R_grain**5 / (np.pi**3 * D) * (1 - np.exp(-np.pi**2 * D * t / R_grain**2))

    # release
    FG_release = lambda t: p(t) - g(t)

    # fractional release
    F_release = lambda t: FG_release(t) / p(t)
    # t_irr = 10
    print(f"Fractional release at {t_irr} d: {F_release(t_irr*24*3600):.10f}")


    


    # 5. Mechanical checks


if __name__ == "__main__":
    main(0.16e-3, 1)
