
            Rf[i] = Rci[i]
            a = r0[i]
            b = Rci[i]
            c = Rco[i]
            p_contact = lambda Tfo: ( -t_gap[i]/b ) /( ( (b**2+c**2)/(c**2-b**2) - 1/nu_clad(T_ci[i]) )/E_clad(T_ci[i]) + (
                (a**2+b**2)/(b**2-a**2) - 1/nu_fuel )/Efuel(Tfo) )
