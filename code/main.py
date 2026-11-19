import numpy as np 
import functions as fn 
import matplotlib.pyplot as plt
import multiprocessing
import csv

def main(t_gap, H_plenum):
    
    print(f"Gap thickness: {t_gap*1000} mm, Plenum height: {H_plenum} m\n\n")

    # fast neutron flux ( E > 100 keV ) [n cm^-2 s^-1]
    neutron_flux = 6.1e15

    # number of axial nodes
    nz = 10
    # number of radial nodes in the fuel
    nr = 50

    # active height [m]
    Ha = 0.85

    # cold geometry
    Rf = 5.42e-3/2 * np.ones(nz)
    Rci = Rf + t_gap
    Rco = 6.55e-3/2 * np.ones(nz)

    V_fuel = np.pi*Ha*Rf[0]**2
    TD = 11310
    rho_f = 0.945 * TD

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

            # 1. THERMAL ANALYSIS AT COLD GEOMETRY 
    BU = 0 
    Tcool, T_clad_cold, Tco, Tci, Tgap, Tfo, u_max, p_contact = fn.thermal_up_to_fuel(q_lin, Rco, Rci, Rf, BU)
    Tfuel_cold, Rfuel = fn.thermal_in_fuel(q_lin, BU, Rf, Tfo, nr)
    # fn.plot_axial_temperatures(z, Tcool, Tco, Tci, Tgap, Tfuel)

    print("\nCold geometry")
    print(f"Rf: {Rf[0]*1000:.2f} mm, Rci: {Rci[0]*1000:.2f} mm, Rco: {Rco[0]*1000:.2f} mm")
    print("Max temperatures [°C]")
    print(f"fuel: {np.max(Tfuel_cold):.2f}, gap: {np.max(Tgap):.2f}, cladding: {np.max(Tci):.2f}, coolant: {np.max(Tcool):.2f}")

            # 2. EXPANSION, THERMAL ANALYSIS AND RESTRUCTURING AT T_IRR = 7 DAYS
    t_irr = 7
    BU = 1e-6 * t_irr * Q / (V_fuel * rho_f)

    print(f"\n\n\t\t Irradiation time: {t_irr} days, Burnup: {BU:.3f} GWD/tHM\n\n")

    # fuel linear thermal expansion
    eps_th_f = lambda T: 1.2e-5 * (T - 25)
    # clad linear thermal expansion
    eps_th_cl = lambda T: -3.101e-4 + 1.545e-5 * T + 2.75e-9 * T**2 
    # Volumetric cladding swelling due to irradiation 
    eps_irr_clad = lambda T, t_irr: 1.5e-5 * np.exp(-2.5e-4 * (T - 450)**2) * (neutron_flux * t_irr * 24 * 3600 / 1e22)**2.75
    # Volumetric fuel swelling due to irradiation
    eps_irr_fuel = lambda BU: 0.01 * 0.07 * BU

    toll = 1e-6
    err = toll+1
    max_iter = 500
    n = 0
    Rf_prev = np.copy(Rf)
    Rci_prev = np.copy(Rci)
    Rco_prev = np.copy(Rco)
    while err > toll and n <= max_iter:
        Rf_hot  = Rf *  (1 + eps_th_f(Tfo)  + (1/3)*eps_irr_fuel(BU))
        Rco_hot = Rco * (1 + eps_th_cl(Tco) + (1/3)*eps_irr_clad(Tco, t_irr))
        Rci_hot = Rci * (1 + eps_th_cl(Tci) + (1/3)*eps_irr_clad(Tci, t_irr))

        # thermal analysis at new geometry
        Tcool, T_clad, Tco, Tci, Tgap, Tfo, u_max, p_contact = fn.thermal_up_to_fuel(q_lin, Rco_hot, Rci_hot, Rf_hot, BU, x_He=1)

        err = np.max([
            np.abs(np.mean(Rf_hot) - np.mean(Rf_prev)),
            np.abs(np.mean(Rci_hot) - np.mean(Rci_prev)),
            np.abs(np.mean(Rco_hot) - np.mean(Rco_prev))
        ])
        Rf_prev = np.copy(Rf_hot)
        Rci_prev = np.copy(Rci_hot)
        Rco_prev = np.copy(Rco_hot)


        n += 1

    # Tfuel, Rfuel = fn.thermal_in_fuel(q_lin, BU, Rf_hot, Tfo, nr)
    print(f"\nHot geometry converged in {n} iterations")
    print(f"Max coolant velocity: {u_max:.3f} m/s")
    print("Expanded max radii [mm]")
    print(f"Rf: {np.max(Rf_hot)*1000:.2f} mm, Rci: {np.max(Rci_hot)*1000:.2f} mm, Rco: {np.max(Rco_hot)*1000:.2f} mm")
    print("Max temperatures [°C]")
    # print(f"fuel: {np.max(Tfuel):.2f}, gap: {np.max(Tgap):.2f}, cladding: {np.max(Tci):.2f}, coolant: {np.max(Tcool):.2f}")
    
    # Plot expanded radii
    # plot_axial_temperatures(z, Tcool, Tco, Tci, Tgap, Tfuel)
    # plt.figure()
    # plt.plot(Tci, 100*eps_irr_clad(Tci), label='Cladding swelling inner')
    # plt.plot(Tco, 100*eps_irr_clad(Tco), label='Claadding swelling outer')
    # plt.legend()
    # plt.show()

    print("\nRestructuring the fuel...")
    Tfuel, Rfuel, r0, r1, r2 = fn.restructure(q_lin, Tfo, Rf, nr, BU)
    # fn.plot_Tfuel(Tfuel, Rfuel, Ha, restructured=True)
    # fn.plot_restructuring_radii(z, r0, r1, r2, Rf)
    z0_idx = np.round(nz/2).astype(int)
    r_clad = np.linspace(Rci_hot[z0_idx], Rco_hot[z0_idx], nr)
    # plt.figure()
    # plt.plot(r_clad*1000, Tfuel_cold[z0_idx,:], label='Cold geometry')
    # plt.plot(r_clad*1000, Tfuel[z0_idx,:], label='Hot geometry')
    # plt.legend()
    # plt.title("Fuel radial temperature")
    # plt.ylabel('Temperature [°C]')
    # plt.xlabel('r [mm]')
    # plt.grid()
    # plt.show()
    print(f"Restructured fuel: r0 = {np.max(r0):.3e}, r1 = {np.max(r1):.3e}, r2 = {np.max(r2):.3e}, T0 = {np.max(Tfuel):.2f} °C")

    # fission gas yield and fission rate
    y_Fg = 0.30
    F_rate = fn.fission_rate(q_lin, z, Rf[0])
    # initial He moles
    p_He = 0.1e6 #[Pa]
    T_He = 20 + 273.15 #[K]
    V_plenum = np.pi * Rci[0]**2 * H_plenum
    V_gas = V_plenum + np.pi * (Rci[0]**2 - Rf[0]**2) * Ha
    n_He = p_He * V_gas / (8.31446 * T_He)
    print(f"\nInitial He moles: {n_He:.3e}")

    

                # TIME LOOP
    time_irr = np.arange(7, 365)

    pressure = []
    Tmax_fuel = []
    Tmax_nodes = np.zeros((nz, len(time_irr)))
    Tmax_clad = []
    Tmax_gap = []
    tgap_min = []
    tgap_max = []
    max_swelling = []
    Tmelt = []

    time_n = 0
    for t_irr in time_irr:
        BU = 1e-6 * t_irr * Q / (V_fuel * rho_f)
        print(f"\n\n\t\t Irradiation time: {t_irr} days, Burnup: {BU:.3f} GWD/tHM\n\n")

        print(f"\nFission rate: {F_rate:.3e}")

        # fission gas production and release [atoms]
        FGR = y_Fg * F_rate * t_irr * 24 * 3600 * V_fuel #[atoms]

        # fission gas moles
        NA = 6.022e23
        n_fg = FGR / NA 
        print(f"Released FG moles: {n_fg:.3e}")
        x_He = n_He / (n_He + n_fg)
        print(f"He molar fraction: {x_He}")

        # expand
        print("\nExpanding...")
        toll = 0.3e-4
        err = toll+1
        max_iter = 100
        n = 0
        Rf_prev = np.copy(Rf)
        Rci_prev = np.copy(Rci)
        Rco_prev = np.copy(Rco)
        i_closed = []
        Rf_new = np.copy(Rf)
        Rci_new = np.copy(Rci)
        Rco_new = np.copy(Rco)
        while err > toll and n <= max_iter:
            for i in range(nz):
                if i not in i_closed:
                    Rf_new[i]  = Rf[i] *  (1 + eps_th_f(Tfo[i])  + (1/3)*eps_irr_fuel(BU))
                    Rco_new[i] = Rco[i] * (1 + eps_th_cl(Tco[i]) + (1/3)*eps_irr_clad(Tco[i], t_irr))
                    Rci_new[i] = Rci[i] * (1 + eps_th_cl(Tci[i]) + (1/3)*eps_irr_clad(Tci[i], t_irr))

            # thermal analysis at new geometry
            Tcool, T_clad, Tco, Tci, Tgap, Tfo, u_max, p_contact = fn.thermal_up_to_fuel(q_lin, Rco_new, Rci_new, Rf_new, BU, x_He=x_He)
                

            for i in range(nz):
                if Rf_new[i] >= Rci_new[i]:
                    # print(f"GAP CHIUSO a z = {z[i]}")
                    # Rf_new[i] = Rci_new[i]
                    if i not in i_closed:
                        i_closed.append(i)

            # if len(i_closed) >  0.85 * nz:
            #     break

            err = np.max([
                np.abs(np.mean(Rf_new) - np.mean(Rf_prev)),
                np.abs(np.mean(Rci_new) - np.mean(Rci_prev)),
                np.abs(np.mean(Rco_new) - np.mean(Rco_prev)),
            ])

            Rf_old = np.copy(Rf_new)
            Rci_old = np.copy(Rci_new)
            Rco_old = np.copy(Rco_new)

            # fn.plot_tgap(z, Rf_new, Rci_new, Rf, Rci, live=True)

            n += 1
        # plt.close()
        for i in range(nz):
            if Rf_new[i] > Rci_new[i]:
                Rf_new[i] = Rci_new[i]

        # write p_contact vector to csv file, append row for each time step
        # file_name = 'p_contact.csv'
        # with open(file_name, mode='a') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(p_contact)


        Tfuel, Rfuel = fn.thermal_in_fuel(q_lin, BU, Rf_new, Tfo, nr, r0, r1, r2)
        # fn.plot_tgap(z, Rf_new, Rci_new, Rf, Rci)
        # fn.plot_Tfuel(Tfuel, Rfuel, Ha, restructured=True)
        print(f"\nHot geometry converged in {n} iterations")
        print(f"Max coolant velocity: {u_max:.3f} m/s")
        print("Expanded max radii [mm]")
        print(f"Rf: {np.max(Rf_new)*1000:.4f} mm, Rci: {np.max(Rci_new)*1000:.4f} mm, Rco: {np.max(Rco_new)*1000:.4f} mm")
        print("Max temperatures [°C]")
        print(f"fuel: {np.max(Tfuel):.2f}, gap: {np.max(Tgap):.2f}, cladding: {np.max(Tci):.2f}, coolant: {np.max(Tcool):.2f}")
        print(f"Max contact pressure: {np.max(p_contact):.3f} MPa")
        # plenum dimensioning
        Tgap_max_K = np.max(Tgap) + 273.15 
        p_tot = (n_He + n_fg) * 8.31446 * Tgap_max_K / V_gas
        print(f"Plenum pressure: {p_tot/1e6:.3f} MPa")
        # plt.show()

        t_gap_min = 1000*np.min(Rci_new - Rf_new)
        t_gap_max = 1000*np.max(Rci_new - Rf_new)

        for i in range(nz):
            Tmax_nodes[i, time_n] = np.max(Tfuel[i,:])
        Tmax_fuel.append(np.max(Tfuel))
        Tmax_clad.append(np.max(Tci))
        Tmax_gap.append(np.max(Tgap))
        Tmelt.append(fn.T_melt_fuel(BU))
        tgap_min.append(t_gap_min)
        tgap_max.append(t_gap_max)
        pressure.append(p_tot/1e6)
        max_swelling.append(np.max(100*eps_irr_clad(Tci, t_irr)))

        time_n += 1




    print(f"\nLimits check")
    print(f"Max fuel temperature: {np.max(Tfuel):.2f} °C < 2600 °C")
    print(f"Max cladding temperature: {np.max(T_clad):.2f} °C < 650 °C")
    print(f"Plenum pressure: {p_tot/1e6:.3f} MPa < 5 MPa")
    print(f"Max coolant velocity: {u_max:.3f} m/s < 8 m/s")
    print(f"Max cladding volumetric swelling: {np.max(100*eps_irr_clad(T_clad, t_irr)):.2f}% < 3%")


    # write Tmax fuel to csv file to plot with other gap thicknesses
    file_name = f'Tmax_fuel_{t_gap*1e6:.0f}_micron.csv'
    with open(file_name, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['time', 'Tmax'])
        for i in range(len(time_irr)):
            writer.writerow([time_irr[i], Tmax_fuel[i]])

    plt.figure()
    plt.plot(time_irr, pressure, label='Plenum pressure [MPa]')
    plt.axhline(y=5, color='r', linestyle='--', label='Limit pressure')
    plt.xlabel('Irradiation time [days]')
    plt.ylabel('Pressure [MPa]')
    plt.grid()
    plt.legend()

    fig, ax = plt.subplots(2,1)
    ax[0].plot(time_irr, Tmax_fuel, label='Max fuel temperature [°C]')
    ax[0].plot(time_irr, Tmelt, linestyle='--', label='Fuel melting temperature [°C]')
    ax[0].axhline(y=2600, color='r', linestyle='--', label='Limit temperature')
    ax[0].legend()
    ax[0].grid()
    ax[1].plot(time_irr, Tmax_clad, label='Max cladding temperature [°C]')
    ax[1].axhline(y=650, color='r', linestyle='--', label='Limit temperature')
    ax[1].legend()
    ax[1].grid()

    plt.figure()
    for i in range(nz):
        plt.plot(time_irr, Tmax_nodes[i,:], label=f'Node at {z[i]:.2f} m')
    plt.xlabel('Irradiation time [days]')
    plt.ylabel('Max temperature [°C]')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.plot(time_irr, tgap_min, label='Min gap thickness [mm]')
    plt.plot(time_irr, tgap_max, label='Max gap thickness [mm]')
    plt.xlabel('Irradiation time [days]')
    plt.ylabel('Gap thickness [mm]')
    plt.legend()
    plt.grid()

    # fn.plot_Tfuel(Tfuel, Rfuel, Ha, restructured=True)

    plt.figure()
    plt.plot(time_irr, max_swelling, label='Max cladding swelling [%]')
    plt.axhline(y=3, color='r', linestyle='--', label='Limit swelling')
    plt.xlabel('Irradiation time [days]')
    plt.ylabel('Cladding swelling [%]')
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    t_gap = 40e-6
    H_plenum = 1.3
    main(t_gap, H_plenum)
