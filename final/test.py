
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
        #
        #
