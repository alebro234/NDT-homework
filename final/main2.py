import numpy as np 
import functions as fn 
import matplotlib.pyplot as plt

def main(t_gap, H_plenum):
    
    print(f"Gap thickness: {t_gap*1000} mm, Plenum height: {H_plenum} m\n\n")

    # fast neutron flux ( E > 100 keV ) [n cm^-2 s^-1]
    neutron_flux = 6.1e15

    # number of axial nodes
    nz = 25
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
    Tcool, T_clad, Tco, Tci, Tgap, Tfo, u_max = fn.thermal_up_to_fuel(q_lin, Rco, Rci, Rf)
    Tfuel, Rfuel = fn.thermal_in_fuel(q_lin, BU, Rf, Tfo, nr)
    # fn.plot_axial_temperatures(z, Tcool, Tco, Tci, Tgap, Tfuel)

    print("\nCold geometry")
    print(f"Rf: {Rf[0]*1000:.2f} mm, Rci: {Rci[0]*1000:.2f} mm, Rco: {Rco[0]*1000:.2f} mm")
    print("Max temperatures [°C]")
    print(f"fuel: {np.max(Tfuel):.2f}, gap: {np.max(Tgap):.2f}, cladding: {np.max(Tci):.2f}, coolant: {np.max(Tcool):.2f}")

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
        Tcool, T_clad, Tco, Tci, Tgap, Tfo, u_max = fn.thermal_up_to_fuel(q_lin, Rco_hot, Rci_hot, Rf_hot, x_He=1)

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
    print(f"fuel: {np.max(Tfuel):.2f}, gap: {np.max(Tgap):.2f}, cladding: {np.max(Tci):.2f}, coolant: {np.max(Tcool):.2f}")
    
    # Plot expanded radii
    # plot_axial_temperatures(z, Tcool, Tco, Tci, Tgap, Tfuel)
    # plt.figure()
    # plt.plot(Tci, 100*eps_irr_clad(Tci), label='Cladding swelling inner')
    # plt.plot(Tco, 100*eps_irr_clad(Tco), label='Claadding swelling outer')
    # plt.legend()
    # plt.show()

    print("\nRestructuring the fuel...")
    Tfuel, Rfuel, r0, r1, r2 = fn.restructure(q_lin, Tfo, Rf, nr, BU)
    fn.plot_Tfuel(Tfuel, Rfuel, Ha, restructured=True)
    fn.plot_restructuring_radii(z, r0, r1, r2, Rf)
    plt.show()
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

    pressure = []
    Tmax_fuel = []
    Tmax_clad = []
    Tmax_gap = []
    tgap_min = []
    tgap_max = []
    max_swelling = []
    Tmelt = []

    # time_irr = np.arange(7, 365, 30)
    time_irr = np.arange(7, 365, 7)

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
        toll = 0.3e-4
        err = toll+1
        max_iter = 50
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
            Tcool, T_clad, Tco, Tci, Tgap, Tfo, u_max = fn.thermal_up_to_fuel(q_lin, Rco_new, Rci_new, Rf_new, x_He=x_He)

            for i in range(nz):
                if Rf_new[i] >= Rci_new[i]:
                    # print(f"GAP CHIUSO a z = {z[i]}")
                    # Rf_new[i] = Rci_new[i]
                    if i not in i_closed:
                        i_closed.append(i)



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

        Tfuel, Rfuel = fn.thermal_in_fuel(q_lin, BU, Rf_new, Tfo, nr, r0, r1, r2)
        # fn.plot_tgap(z, Rf_new, Rci_new, Rf, Rci)
        # fn.plot_Tfuel(Tfuel, Rfuel, Ha, restructured=True)
        print(f"\nHot geometry converged in {n} iterations")
        print(f"Max coolant velocity: {u_max:.3f} m/s")
        print("Expanded max radii [mm]")
        print(f"Rf: {np.max(Rf_hot)*1000:.2f} mm, Rci: {np.max(Rci_hot)*1000:.2f} mm, Rco: {np.max(Rco_hot)*1000:.2f} mm")
        print("Max temperatures [°C]")
        print(f"fuel: {np.max(Tfuel):.2f}, gap: {np.max(Tgap):.2f}, cladding: {np.max(Tci):.2f}, coolant: {np.max(Tcool):.2f}")
        # plenum dimensioning
        Tgap_max_K = np.max(Tgap) + 273.15 
        p_tot = (n_He + n_fg) * 8.31446 * Tgap_max_K / V_gas
        print(f"Plenum pressure: {p_tot/1e6:.3f} MPa")
        # plt.show()


        pressure.append(p_tot/1e6)
        Tmax_fuel.append(np.max(Tfuel))
        Tmax_clad.append(np.max(Tci))
        Tmax_gap.append(np.max(Tgap))
        tgap_min.append(1000*np.min(Rci_new - Rf_new))
        tgap_max.append(1000*np.max(Rci_new - Rf_new))
        max_swelling.append( np.max(100*eps_irr_clad(Tci, t_irr)) )
        Tmelt.append( fn.T_melt_fuel(BU) )




                # 6. Mechanical checks

            # 5.1. Cladding thermal stresses

        # Plot the T distribution inside the cladding
        # z = np.linspace(-Ha/2, Ha/2, T_clad.shape[0])
        # Rco_avg = np.mean(Rco_hot)
        # Rci_avg = np.mean(Rci_hot)
        # r = np.linspace(Rci_avg, Rco_avg, T_clad.shape[1])
        # R, Z = np.meshgrid(r, z, indexing='xy')
        # plt.figure(figsize=(10, 6))
        # plt.pcolormesh(R*1000, Z*1000, T_clad, shading='auto', cmap='hot')
        # plt.colorbar(label='Temperature [°C]')
        # plt.title('Cladding Temperature Distribution')
        # plt.xlabel('r [mm]')
        # plt.ylabel('z [mm]')
        # plt.grid(True)      
        # plt.show()
        #
        # # Thermal stresses evaluation
        #
        # alpha_clad = 1.545e-5 # [1/°C] #thermal expansion coefficient from derivative of epsilon_th_cl approximated (source: Simone Coccuz)
        # E_clad = lambda T: (202.7-0.08167*T)*1e9 # [Pa]
        # nu_clad = lambda T : 0.277 + T*6e-5 # [-]
        # S_th_r = np.zeros((nz, nr ))
        # S_th_theta = np.zeros((nz, nr ))
        # S_th_z = np.zeros((nz, nr ))
        #
        # for i in range(nz):
        #
        #     T = (Tco[i] - Tci[i])/(Rco_hot[i] - Rci_hot[i]) # [°C/m]
        #     O = -Rci_hot[i]/(Rco_hot[i] - Rci_hot[i])*(Tco[i] - Tci[i]) + Tci[i]  # [°C]
        #     r_cladding = np.array(np.linspace(Rci_hot[i], Rco_hot[i], nr))
        #     for j in range(nr):               
        #         numerator1 = (((r_cladding[j]**2 - Rci_hot[i]**2) / (Rco_hot[i]**2 - Rci_hot[i]**2))* (T / 3 * (Rco_hot[i]**3 - Rci_hot[i]**3) + O / 2 * (Rco_hot[i]**2 - Rci_hot[i]**2))- (T / 3 * (r_cladding[j]**3 - Rci_hot[i]**3) + O / 2 * (r_cladding[j]**2 - Rci_hot[i]**2)))
        #         S_th_r[i,j] = alpha_clad * E_clad(T_clad[i,j]) / (1 - nu_clad(T_clad[i,j])) / (r_cladding[j]**2) * numerator1
        #         numerator2 = (((r_cladding[j]**2 + Rci_hot[i]**2) / (Rco_hot[i]**2 - Rci_hot[i]**2))* (T / 3 * (Rco_hot[i]**3 - Rci_hot[i]**3) + O / 2 * (Rco_hot[i]**2 - Rci_hot[i]**2)) + (T / 3 * (r_cladding[j]**3 - Rci_hot[i]**3) + O / 2 * (r_cladding[j]**2 - Rci_hot[i]**2)) - T_clad[i,j] * r_cladding[j]**2)
        #         S_th_theta[i,j] = alpha_clad * E_clad(T_clad[i,j]) / (1 - nu_clad(T_clad[i,j])) / (r_cladding[j]**2) * numerator2
        #         #unsing the simplified formula for sigma_theta
        #         # S_th_theta[i,j] = alpha_clad * E_clad(T_clad[i,j]) / (1 - nu_clad(T_clad[i,j])) * (np.mean(T_clad[i,:]) - T_clad[i,j]) 
        #         S_th_z[i,j] = alpha_clad * E_clad(T_clad[i,j]) / (1 - nu_clad(T_clad[i,j])) * (2/(Rco_hot[i]**2 - Rci_hot[i]**2) * (T / 3 * (Rco_hot[i]**3 - Rci_hot[i]**3) + O / 2 * (Rco_hot[i]**2 - Rci_hot[i]**2)) - T_clad[i,j])
        #
        #
        # # Create a 2x2 grid for subplots
        # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        #
        # # Plot stress r distribution
        # im1 = axs[0, 0].pcolormesh(R * 1000, Z * 1000, S_th_r * 1e-6, shading='auto', cmap='Blues_r')
        # fig.colorbar(im1, ax=axs[0, 0], label='stresses [MPa]')
        # axs[0, 0].set_title('Cladding Sigma_r Distribution')
        # axs[0, 0].set_xlabel('r [mm]')
        # axs[0, 0].set_ylabel('z [mm]')
        # axs[0, 0].grid(True)
        #
        # # Plot stress theta distribution
        # im2 = axs[0, 1].pcolormesh(R * 1000, Z * 1000, S_th_theta * 1e-6, shading='auto', cmap='bwr')
        # fig.colorbar(im2, ax=axs[0, 1], label='stresses [MPa]')
        # axs[0, 1].set_title('Cladding Sigma_theta Distribution')
        # axs[0, 1].set_xlabel('r [mm]')
        # axs[0, 1].set_ylabel('z [mm]')
        # axs[0, 1].grid(True)
        #
        # # Plot stress z distribution
        # im3 = axs[1, 0].pcolormesh(R * 1000, Z * 1000, S_th_z * 1e-6, shading='auto', cmap='bwr')
        # fig.colorbar(im3, ax=axs[1, 0], label='stresses [MPa]')
        # axs[1, 0].set_title('Cladding Sigma_z Distribution')
        # axs[1, 0].set_xlabel('r [mm]')
        # axs[1, 0].set_ylabel('z [mm]')
        # axs[1, 0].grid(True)
        #
        # # Plot stress distribution for a fixed z
        # i_max = np.argmax(np.max(T_clad, axis=1))
        # r_values = np.linspace(Rci[i_max], Rco[i_max], 100) * 1000
        #
        # axs[1, 1].plot(r_values, S_th_z[i_max, :] * 1e-6, linestyle='-', label='Sigma_z')
        # axs[1, 1].plot(r_values, S_th_r[i_max, :] * 1e-6, linestyle='--', label='Sigma_r')
        # axs[1, 1].plot(r_values, S_th_theta[i_max, :] * 1e-6, linestyle='-.', label='Sigma_theta')
        # axs[1, 1].set_title('Thermal Stresses Distribution for a Fixed z')
        # axs[1, 1].set_xlabel('r [mm]')
        # axs[1, 1].set_ylabel('Stresses [MPa]')
        # axs[1, 1].grid(True)
        # axs[1, 1].legend()
        #
        # # Adjust layout for better appearance
        # plt.tight_layout()
        # plt.show()
        #
        #     # 5.2. Cladding mechanical stresses
        # S_mech_r = np.zeros((nz, nr ))
        # S_mech_theta = np.zeros((nz, nr ))
        # S_mech_z = np.zeros((nz, nr ))
        # pressure = 5e6 # [Pa]
        # #evaluate Lame solution 
        # for i in range(nz):
        #     r_cladding = np.array(np.linspace(Rci_hot[i], Rco_hot[i], nr))
        #     for j in range(nr):               
        #         S_mech_r[i,j] = -Rci_hot[i]**2 / (Rco_hot[i]**2 - Rci_hot[i]**2) * (Rco_hot[i]**2/r_cladding[j]**2 - 1) * pressure
        #         S_mech_theta[i,j] = Rci_hot[i]**2 / (Rco_hot[i]**2 - Rci_hot[i]**2) * (Rco_hot[i]**2/r_cladding[j]**2 + 1) * pressure
        #
        # #evaluate the Mariotte solution
        # S_mar_r = np.zeros((nz, nr ))
        # S_mar_theta = np.zeros((nz, nr ))
        # S_mar_z = np.zeros((nz, nr ))
        # R_mar = Rci_hot
        # t_mar = Rco_hot - Rci_hot
        # for i in range(nz):
        #     r_cladding = np.array(np.linspace(Rci_hot[i], Rco_hot[i], nr))
        #     for j in range(nr):               
        #         S_mar_r[i,j] = -pressure/2
        #         S_mar_theta[i,j] = pressure * R_mar[i] / t_mar[i]
        #         S_mar_z[i,j] = pressure * R_mar[i] / t_mar[i] / 2
        #
        #
        # # Create a 2x2 grid for subplots
        # fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        #
        # # First plot: Radial mechanical stress (Sigma_r) distribution
        # im1 = axs[0, 0].pcolormesh(R * 1000, Z * 1000, S_mech_r * 1e-6, shading='auto', cmap='Blues_r')
        # fig.colorbar(im1, ax=axs[0, 0], label='Stresses [MPa]')
        # axs[0, 0].set_title('Cladding Sigma_r Distribution')
        # axs[0, 0].set_xlabel('r [mm]')
        # axs[0, 0].set_ylabel('z [mm]')
        # axs[0, 0].grid(True)
        #
        # # Second plot: Circumferential (hoop) mechanical stress (Sigma_theta) distribution
        # im2 = axs[0, 1].pcolormesh(R * 1000, Z * 1000, S_mech_theta * 1e-6, shading='auto', cmap='Reds')
        # fig.colorbar(im2, ax=axs[0, 1], label='Stresses [MPa]')
        # axs[0, 1].set_title('Cladding Sigma_theta Distribution')
        # axs[0, 1].set_xlabel('r [mm]')
        # axs[0, 1].set_ylabel('z [mm]')
        # axs[0, 1].grid(True)
        #
        # # Third plot: Axial mechanical stress (Sigma_z) distribution
        # im3 = axs[1, 0].pcolormesh(R * 1000, Z * 1000, S_mar_z * 1e-6, shading='auto', cmap='Reds')
        # fig.colorbar(im3, ax=axs[1, 0], label='Stresses [MPa]')
        # axs[1, 0].set_title('Cladding Sigma_z Distribution')
        # axs[1, 0].set_xlabel('r [mm]')
        # axs[1, 0].set_ylabel('z [mm]')
        # axs[1, 0].grid(True)
        #
        #
        # # Fourth plot: Compare mechanical stresses with the Mariotte solution for a fixed z
        # r_values = np.linspace(Rci[i_max], Rco[i_max], 100) * 1000  # Radial positions for the plot
        #
        # axs[1, 1].plot(r_values, S_mech_z[i_max, :] * 1e-6, linestyle='-', label='Sigma_z')
        # axs[1, 1].plot(r_values, S_mech_r[i_max, :] * 1e-6, linestyle='--', label='Sigma_r')
        # axs[1, 1].plot(r_values, S_mech_theta[i_max, :] * 1e-6, linestyle='-.', label='Sigma_theta')
        # axs[1, 1].plot(r_values, S_mar_z[i_max, :] * 1e-6, linestyle='-', label='Sigma_z Mariotte')
        # axs[1, 1].plot(r_values, S_mar_r[i_max, :] * 1e-6, linestyle='--', label='Sigma_r Mariotte')
        # axs[1, 1].plot(r_values, S_mar_theta[i_max, :] * 1e-6, linestyle='-.', label='Sigma_theta Mariotte')
        # axs[1, 1].set_title('Mechanical Stress Distribution (Fixed z)')
        # axs[1, 1].set_xlabel('r [mm]')
        # axs[1, 1].set_ylabel('Stresses [MPa]')
        # axs[1, 1].grid(True)
        # axs[1, 1].legend()
        #
        # # Adjust layout for better visualization
        # plt.tight_layout()
        # plt.show()
        #
        #
        # #EVALUATE THAT MECHANICAL STRESSES RESPECT TRESCA CRITERION
        # print(f"S yield: {fn.S_yield(np.max(T_clad))} MPa")
        # S_allowable = 1/3*fn.S_yield(np.max(T_clad)) # [MPa]
        # print(f"Allowable stress: {S_allowable} MPa")
        # S_equivalent_Tresca_mech = max(np.max(np.abs(S_mech_r - S_mech_theta)), np.max(np.abs(S_mech_r - S_mech_z)), np.max(np.abs(S_mech_theta - S_mech_z))) #Pa
        # print(f"Max equivalent mechanical stress Lame: {S_equivalent_Tresca_mech* 1e-6} MPa")
        # print(f"Max equivalent mechanical stress Mariotte: {np.max(np.abs(S_mar_r - S_mar_theta))*1e-6} MPa")
        # S_equivalent_Tresca_thermal = max(np.max(np.abs(S_th_r - S_th_theta)), np.max(np.abs(S_th_r - S_th_z)), np.max(np.abs(S_th_theta - S_th_z))) #Pa
        # print(f"Max equivalent thermal stress: {S_equivalent_Tresca_thermal* 1e-6} MPa")
        #
        #
        # # # 5.3. Design the minimum cladding thickness
        # # Design to withstand mechanical stresses, verify then thermal stresses
        # # TRESCA failure criterion(easy, conservative) + MARIOTTE solution(conservative w/ respect to yielding of the whole section)
        #
        # Stress_int = 1/3 * fn.S_yield(np.max(T_clad))*1e6 # [Pa]
        # t_clad_min = pressure * Rco_avg / (Stress_int + 0.5 * pressure) #m
        # print(f"Minimum cladding thickness: {t_clad_min*1000:.3f} mm")
        #
        #
        # # 5.4. Creep time to rupture verification
        # S_tot_r = S_th_r + S_mech_r
        # S_tot_theta = S_th_theta + S_mech_theta
        # S_tot_z = S_th_z + S_mech_z
        # # plot stress distribution for a fixed z
        # plt.figure()
        # plt.plot(np.linspace(Rci[i_max], Rco[i_max], 100)*1000, S_tot_z[i_max,:]*1e-6, linestyle='-', label='Sigma_z')
        # plt.plot(np.linspace(Rci[i_max], Rco[i_max], 100)*1000, S_tot_r[i_max,:]*1e-6, linestyle='--', label='Sigma_r')
        # plt.plot(np.linspace(Rci[i_max], Rco[i_max], 100)*1000, S_tot_theta[i_max,:]*1e-6, linestyle='-.', label='Sigma_theta')
        # plt.title('Cladding Total Stress Distribution for a fixed z')
        # plt.xlabel('r [mm]')
        # plt.ylabel('Stresses [MPa]')
        # plt.grid(True)
        # plt.legend()
        # plt.show()
        #
        # # Equivalent Stress calculation for each z node (von Mises)
        # S_vm = []
        # for i in range(nz):
        #     S_tot_max_r = np.max(S_tot_r[i,:])
        #     S_tot_max_theta = np.max(S_tot_theta[i,:])
        #     S_tot_max_z = np.max(S_tot_z[i,:])
        #     S_vm.append(1/np.sqrt(2) * np.sqrt( (S_tot_max_r - S_tot_max_theta)**2 + (S_tot_max_theta - S_tot_max_z)**2 + (S_tot_max_z - S_tot_max_r)**2 )) # [Pa]
        #
        # # Larson Miller Parameter evaluation for each z node
        # LMP = (2060 - np.array(S_vm) * 1e-6) / 0.095
        #
        # # Minimum time to rupture using Filacchioni et al correlation ([T]=K, [p]=MPa, [t]=h)
        # time_rupture = 10**(LMP/(np.mean(T_clad, axis=1) + 273.15) - 17.125)  #[h]
        # min_time_rupture = np.min(time_rupture)
        # print(f"Minimum time to rupture: {min_time_rupture:.3f} h, i.e {min_time_rupture/(24*365):.3f} years")
        #
        #
        # Limits check
    print(f"\nLimits check")
    print(f"Max fuel temperature: {np.max(Tfuel):.2f} °C < 2600 °C")
    print(f"Max cladding temperature: {np.max(T_clad):.2f} °C < 650 °C")
    print(f"Plenum pressure: {p_tot/1e6:.3f} MPa < 5 MPa")
    print(f"Max coolant velocity: {u_max:.3f} m/s < 8 m/s")
    print(f"Max cladding volumetric swelling: {np.max(100*eps_irr_clad(T_clad, t_irr)):.2f}% < 3%")

    plt.figure()
    plt.plot(time_irr, pressure, label='Plenum pressure [MPa]')
    plt.xlabel('Irradiation time [days]')
    plt.ylabel('Pressure [MPa]')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(time_irr, Tmax_fuel, label='Max fuel temperature [°C]')
    plt.plot(time_irr, Tmelt, linestyle="--", label='Fuel melting temperature [°C]')
    plt.plot(time_irr, Tmax_gap, label='Max gap temperature [°C]')
    plt.xlabel('Irradiation time [days]')
    plt.ylabel('Temperature [°C]')
    plt.grid()
    plt.legend()

    plt.figure()
    plt.plot(time_irr, tgap_min, label='Min gap thickness [mm]')
    plt.plot(time_irr, tgap_max, label='Max gap thickness [mm]')
    plt.xlabel('Irradiation time [days]')
    plt.ylabel('Gap thickness [mm]')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time_irr, max_swelling, label='Max swelling [%]')
    plt.xlabel('Irradiation time [days]')
    plt.ylabel('Swelling [%]')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    t_gap = 40e-6
    H_plenum = 1.3
    main(t_gap, H_plenum)
