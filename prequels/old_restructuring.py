
        # solution = []
        # # compute for each axial node
        # for i in range(nz):
        #     # guess internal radius [m]
        #     r0 = 0.1 * Rf[i]
        #
        #     toll = 1e-9
        #     max_iter = 1000
        #     err = toll + 1
        #     n = 0
        #     while err > toll and n <= max_iter:
        #         # vol power densities [W m^-3]
        #         chi, I = radial_redstr(Rf[i], r0)
        #         chi_avg_10 = lambda r1: np.trapz(chi(np.linspace(r0, r1, 100)), np.linspace(r0, r1, 100)) / (r1 - r0)
        #         Omega = np.pi * (Rf[i]**2 - r0**2)
        #         q3 = q_lin(z[i]) / Omega
        #         q2 = q3 * rho2 / rho3
        #         # q1 = lambda r1: q3 * rho1 / rho3 * chi_avg_10(r1)
        #         q1 = lambda r1: q3 * rho1 / rho3
        #
        #         # conduction integrals [W m^-1]
        #         I1 = lambda T0: tc_fuel_integral(T0, T1, 2 - 1.945, y0, p1, BU)
        #         I2 = tc_fuel_integral(T1, T2, 2 - 1.945, y0, p2, BU)
        #         I3 = tc_fuel_integral(T2, T_fo[i], 2 - 1.945, y0, p3, BU)
        #
        #         def func(vars):
        #             T0, r1, r2 = vars
        #             return [
        #                 I1(T0) - 0.25*q1(r1) * (r1**2 - r0**2) + 0.5*q1(r1)*r0**2*np.log(r1/r0),
        #                 I2 - 0.25*q2 * (r2**2 - r1**2) + 0.5*(q2*r1**2 + q1(r1)*(r0**2 - r1**2))*np.log(r2/r1),
        #                 I3 - 0.25*q3 * (Rf[i]**2 - r2**2)
        #             ]
        #
        #         sol = fsolve(func, [T1, 0.9*Rf_avg, 0.8*Rf_avg])
        #         T0 = sol[0]
        #         r1 = sol[1]
        #         r2 = sol[2]
        #         r0_new = np.sqrt( (rho1-rho2)*r1**2 / rho1 + (rho2-rho3)*r2**2 / rho1 )
        #
        #         err = np.linalg.norm(r0_new - r0)
        #         r0 = r0_new
        #         n += 1
        #     solution.append([r0, r1, r2, T0])
        #     print(f"chi_avg_10: {chi_avg_10(np.max(r1))}")
        #
        # T_f = np.zeros((nz, nr))
        # r0 = np.array([sol[0] for sol in solution])
        # r1 = np.array([sol[1] for sol in solution])
        # r2 = np.array([sol[2] for sol in solution])
        # T0 = np.array([sol[3] for sol in solution])
        #
        # # generate fuel temp matrix
        # for i in range(nz):
        #     def kf(T):
        #         if T > T1:
        #             return tc_fuel(T, x, y0, p1, BU)
        #         elif T > T2:
        #             return tc_fuel(T, x, y0, p2, BU)
        #         else:
        #             return tc_fuel(T, x, y0, p3, BU)
        #
        # print("Restructuring")
        # print(f"r0: {np.max(r0)*1000:.2f} mm")
        # print(f"r1: {np.max(r1)*1000:.2f} mm")
        # print(f"r2: {np.max(r2)*1000:.2f} mm")
        # print(f"T0: {np.max(T0):.2f} °C")
        #
        # plt.figure()
        # plt.title("Fuel restructuring")
        # plt.plot(r0*1000, z, label='r0')
        # plt.plot(r1*1000, z, label='r1')
        # plt.plot(r2*1000, z, label='r2')
        # plt.plot(Rf*1000, z, label='Rf')
        # plt.legend()
        # plt.show()
