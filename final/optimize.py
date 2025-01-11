import numpy as np 
import json
import functions as fn 
import matplotlib.pyplot as plt
from BreedingProgram import Breeder, check_multiproc
import multiprocessing

def object_func(vars):

    t_gap, H_plenum = vars

    # fast neutron flux ( E > 100 keV ) [n cm^-2 s^-1]
    neutron_flux = 6.1e15

    # number of axial nodes
    nz = 10
    # number of radial nodes in the fuel
    nr = 100

    # active height [m]
    Ha = 0.85

    # cold geometry
    Rf = 5.42e-3/2 * np.ones(nz)
    Rci = Rf + t_gap
    Rco = 6.55e-3/2 * np.ones(nz)

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
    BU = 1e-6 * 7 * Q / (V_fuel * rho_f)

            # 1. Thermal analysis at cold geometry

    Tcool, T_clad, Tco, Tci, Tgap, Tfo, u_max = fn.thermal_up_to_fuel(q_lin, Rco, Rci, Rf)
    Tfuel, Rfuel = fn.thermal_in_fuel(q_lin, BU, Rf, Tfo, nr, restructured=False, log=False)
    # fn.plot_axial_temperatures(z, Tcool, Tco, Tci, Tgap, Tfuel)


            # 2. FGR calculation for gap thermal conductivity

    y_Fg = 0.30
    F_rate = fn.fission_rate(q_lin, z, Rf[0])

    # fission gas production and release [atoms]
    FGR = y_Fg * F_rate * t_irr * 24 * 3600 * V_fuel #[atoms]

    # initial He moles
    p_He = 0.1e6 #[Pa]
    T_He = 20 + 273.15 #[K]
    V_plenum = np.pi * Rci[0]**2 * H_plenum
    V_gas = V_plenum + np.pi * (Rci[0]**2 - Rf[0]**2) * Ha
    n_He = p_He * V_gas / (8.31446 * T_He)
    # fission gas moles
    NA = 6.022e23
    n_fg = FGR / NA 
    x_He = n_He / (n_He + n_fg)

            # 3. Expansion and thermal analysis at hot geometry

    # fuel linear thermal expansion
    eps_th_f = lambda T: 1.2e-5 * (T - 25)
    # clad linear thermal expansion
    eps_th_cl = lambda T: -3.101e-4 + 1.545e-5 * T + 2.75e-9 * T**2 
    # Volumetric cladding swelling due to irradiation 
    eps_irr_clad = lambda T: 1.5e-5 * np.exp(-2.5e-4 * (T - 450)**2) * (neutron_flux * t_irr * 24 * 3600 / 1e22)**2.75
    # Volumetric fuel swelling due to irradiation
    eps_irr_fuel = 0.01 * 0.07 * BU

    toll = 1e-6
    err = toll+1
    max_iter = 500
    n = 0
    Rf_prev = np.copy(Rf)
    Rci_prev = np.copy(Rci)
    Rco_prev = np.copy(Rco)
    while err > toll and n <= max_iter:
        Rf_hot  = Rf *  (1 + eps_th_f(Tfo)  + (1/3)*eps_irr_fuel)
        Rco_hot = Rco * (1 + eps_th_cl(Tco) + (1/3)*eps_irr_clad(Tco))
        Rci_hot = Rci * (1 + eps_th_cl(Tci) + (1/3)*eps_irr_clad(Tci))

        # damping factor to stabilize iteration
        alpha = 0.5
 
        Rf_hot = alpha * Rf_hot + (1 - alpha) * Rf_prev
        Rco_hot = alpha * Rco_hot + (1 - alpha) * Rco_prev
        Rci_hot = alpha * Rci_hot + (1 - alpha) * Rci_prev

        # thermal analysis at new geometry
        Tcool, T_clad, Tco, Tci, Tgap, Tfo, u_max = fn.thermal_up_to_fuel(q_lin, Rco_hot, Rci_hot, Rf_hot, x_He=x_He)

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

    if any(Rf_hot > Rci_hot):
        print(f"Gap closure at tgap = {t_gap}")

            # 4. Restructuring

    Tfuel, Rfuel = fn.thermal_in_fuel(q_lin, BU, Rf_hot, Tfo, nr, restructured=True, log=False)

         # 5. Plenum dimensioning

    Tgap_max_K = np.max(Tgap) + 273.15 
    p_tot = (n_He + n_fg) * 8.31446 * Tgap_max_K / V_gas

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

    return np.max(Tfuel), np.max(T_clad), p_tot, u_max, 100*np.max(eps_irr_clad(T_clad))


if __name__ == "__main__":




    cpus = multiprocessing.cpu_count()
    T_fuel = []
    T_clad = []
    Ntest = 100
    t_gap_values = np.linspace(5e-6, 50e-6, Ntest)

    args = [(t_gap, 1) for t_gap in t_gap_values]
    with multiprocessing.Pool(cpus) as pool:
        results = pool.map(object_func, args)

    T_fuel = [result[0] for result in results]
    T_clad = [result[1] for result in results]
    pressure = [result[2]/1e6 for result in results]


    # fig, axs = plt.subplots(2)
    # fig.suptitle('Temperature vs gap thickness')
    # axs[0].plot(1e6*np.linspace(20e-6, 200e-6, 100), T_fuel, label="Fuel")
    # axs[0].axhline(2600, color="r", linestyle="--", label="Fuel limit")
    # axs[0].set_ylabel("Fuel temperature [°C]")
    # axs[0].legend()
    # axs[0].grid()
    # axs[1].plot(1e6*np.linspace(20e-6, 200e-6, 100), T_clad, label="Cladding")
    # axs[1].axhline(650, color="g", linestyle="--", label="Cladding limit")
    # axs[1].set_xlabel("Gap thickness [microns]")
    # axs[1].set_ylabel("Cladding temperature [°C]")
    # axs[1].legend()
    # axs[1].grid()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(1e6*t_gap_values, pressure)
    # plt.axhline(5, color="r", linestyle="--")
    # plt.xlabel("Gap thickness [microns]")
    # plt.ylabel("Plenum pressure [MPa]")
    # plt.title("Plenum pressure at H_plenum = 1m")
    # plt.grid()
    # plt.show()
    #
    #
    # # CHOICE OF t_gap
    # t_gap = 125e-6
    #
    # H_plenum_values = np.linspace(0.5, 1.2, Ntest)
    #
    # pressure = []
    #
    # args = [(t_gap, H_plenum) for H_plenum in H_plenum_values]
    # with multiprocessing.Pool(cpus) as pool:
    #     results = pool.map(object_func, args)
    # pressure = [result[2]/1e6 for result in results] # MPa
    #
    # plt.figure()
    # plt.plot(H_plenum_values, pressure)
    # plt.axhline(5, color="r", linestyle="--")
    # plt.xlabel("Plenum height [m]")
    # plt.ylabel("Plenum pressure [MPa]")
    # plt.title("Plenum pressure at t_gap = 125 microns")
    # plt.grid()
    # plt.show()
    #
    # # Limits check
    # # print(f"\nLimits check")
    # # print(f"Max fuel temperature: {Tfuel_max:.2f} °C < 2600 °C")
    # # print(f"Max cladding temperature: {T_clad_max:.2f} °C < 650 °C")
    # # print(f"Plenum pressure: {p_tot/1e6:.3f} MPa < 5 MPa")
    # # print(f"Max coolant velocity: {u_max:.3f} m/s < 8 m/s")
    # # print(f"Max cladding volumetric swelling: {eps_irr_clad_max:.2f}% < 3%")
    #
    #
