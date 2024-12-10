'''
HELIUM EMBRITTLEMENT EVALUATION @PEAK POWER NODE on 15-15, Titanium stabilized, cold worked stainless steel (AIM1)
@author :  SciancatoAbilissimo 22/11/2024

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#%% DATA CELL 
q_avg = 38.7e3 # W m^-1
c =  np.array([0.572, 0.737, 0.868, 0.958, 1, 0.983, 0.912, 0.802, 0.658, 0.49])
# max power
q0 = q_avg*10 / c.sum() * 1e-6  #[MW]
TD = 11.310 # ton m^-3
rho_MOX = 0.945 * TD 
pellet_section = np.pi*5.42e-3**2 #[m2]
fi_th = 1e12  # [n/(cm2*s)] thermal neutrons flux
fi_f = 6.1e15   # [n/(cm2*s)] fast neutrons flux (nell'homework lo danno a 6.1e15 al peak power node)
# CROSS SECTIONS [cm2] taken from Olander, sembrano un po' grossolane?
sigt_B = 3800e-24         # Boro10
sigt_Ni58 = 4.4e-24      # 
sigt_Ni59 =  13e-24    #
sigf_Fe =  0.23e-27       # Ferro56
sigf_Cr =  0.2e-27       # Cromo52
sigf_N =    41e-27         #   Azoto14
sigf_B =    623e-27           #  Boro10
sigf_Ni = 4.2e-27

# MASS PERCENTAGE CLADDING COMPOSITION [%wt] DA CONTROLLARE AZOTO!!!
Cr_wpc = 15
Ni_wpc = 15
Mo_wpc = 1.5
Mn_wpc = 1.5
Si_wpc = 0.9
Ti_wpc = 0.4
C_wpc = 0.09
N_wpc = 0.015
B_wpc = 60e-6
Fe_wpc = 100 - (Cr_wpc + Ni_wpc + Mo_wpc + Mn_wpc + Si_wpc + Ti_wpc + C_wpc + N_wpc + B_wpc)
rhoAIM1 = 7850 #[kg/m3]

# INITIAL CONCENTRATIONS [atoms/m3]

Nav = 6.02214076e23 # [at/mol] Avogadro
u = 1.660538921e-24    #[grams] atomic mass unit
Ni58_isoAbb = 0.68077  #da nuchart
B10_isoAbb = 0.199
#qua sotto andrebbero prese tutte da wiki e basta mi sa
MM_B = 10.012936862  # [g/mol] molar mass  (da nuchart per il boro10)
MM_N = 14.0067  #wiki (già media degli isotopi)
MM_Cr = 51.9961  #wiki (già media degli isotopi)
MM_Fe = 55.934935537   #(da nuchart per il ferro56)
MM_Ni = 58.6934      #wiki (già media degli isotopi)
MM_Mo = 95.94
MM_Mn = 54.938049
MM_Si = 28.0855
MM_Ti = 47.867

def wt_to_conc(wpc, rho, MM):
    conc = wpc * rho * 10 * Nav / MM
    return conc

B10_0 = wt_to_conc(B_wpc, rhoAIM1, MM_B) * B10_isoAbb
N14_0 = wt_to_conc(N_wpc, rhoAIM1, MM_N)
Cr52_0 = wt_to_conc(Cr_wpc, rhoAIM1, MM_Cr)
Fe56_0 = wt_to_conc(Fe_wpc, rhoAIM1, MM_Fe)
Ni58_0 = wt_to_conc(Ni_wpc, rhoAIM1, MM_Ni) * Ni58_isoAbb
Ni59_0 = 0
ceterNi_0 = wt_to_conc(Ni_wpc, rhoAIM1, MM_Ni) * (1-Ni58_isoAbb)
He_B_0 = 0
He_N_0 = 0
He_Cr_0 = 0 
He_Fe_0 = 0 
He_Ni_0 = 0
He_0 = 0

y0 = np.array([B10_0, N14_0, Cr52_0, Fe56_0, Ni58_0, Ni59_0, ceterNi_0, He_B_0, He_N_0, He_Cr_0, He_Fe_0, He_Ni_0, He_0])

Mn = wt_to_conc(Mn_wpc, rhoAIM1, MM_Mn)
Mo = wt_to_conc(Mn_wpc, rhoAIM1, MM_Mn)
Si = wt_to_conc(Si_wpc, rhoAIM1, MM_Si)
Ti = wt_to_conc(Ti_wpc, rhoAIM1, MM_Ti)
# TIME 
t0 = 0
t_fin = 60 * 60 * 24 * 360 #[s] in un anno 
t_span = (t0, t_fin)
t_eval = np.linspace(t0, t_fin, 100000)

#%% CORPUS


def Bateman_set(t, y):
    B10, N14, Cr52, Fe56, Ni58, Ni59, ceterNi, He_B, He_N, He_Cr, He_Fe, He_Ni, He = y
    #BORO
    ddt_B10 = - sigt_B * fi_th * B10 - sigf_B * fi_f * B10 + sigf_N * fi_f * N14
    ddt_He_B = sigt_B * fi_th * B10 + sigf_B * fi_f * B10
    #AZOTO
    ddt_N14 = - sigf_N * fi_f * N14 
    ddt_He_N = + sigf_N * fi_f * N14
    #CROMO
    ddt_Cr52 = - sigf_Cr * fi_f * Cr52 + sigf_Fe * fi_f * Fe56
    ddt_He_Cr = + sigf_Cr * fi_f * Cr52
    #FERRO
    ddt_Fe56 = sigt_Ni59 * fi_th * Ni59 - sigf_Fe * fi_f * Fe56 + sigf_Ni * fi_f * (Ni58+Ni59)
    ddt_He_Fe = sigf_Fe * fi_f * Fe56
    #NICKEL
    ddt_Ni58 = - sigt_Ni58 * fi_th * Ni58 - sigf_Ni * fi_f * Ni58
    ddt_Ni59 = sigt_Ni58 * fi_th * Ni58 - sigt_Ni59 * fi_th * Ni59 - sigf_Ni * fi_f * Ni59
    ddt_ceterNi = -sigf_Ni * fi_f * ceterNi  
    ddt_He_Ni = sigt_Ni59 * fi_th * Ni59 + sigf_Ni * fi_f * (Ni58+Ni59+ceterNi)
    #TOTALE
    ddt_He = ddt_He_B + ddt_He_N + ddt_He_Cr + ddt_He_Fe + ddt_He_Ni

    return np.array([ddt_B10, ddt_N14, ddt_Cr52, ddt_Fe56, ddt_Ni58, ddt_Ni59, ddt_ceterNi, ddt_He_B, ddt_He_N, ddt_He_Cr, ddt_He_Fe, ddt_He_Ni, ddt_He])

bateman_solution = solve_ivp(Bateman_set, t_span, y0, method = 'RK45', t_eval=t_eval)
print(bateman_solution.message) 

t, B10, N14, Cr52, Fe56, Ni58, Ni59, ceterNi, He_B, He_N, He_Cr, He_Fe, He_Ni, He = bateman_solution.t, *bateman_solution.y #L'asterisco *  è un operatore di unpacking a quanto pare

total_conc_AIM1 = B10[-1] + N14[-1] + Cr52[-1] + Fe56[-1] + Ni58[-1] + Ni59[-1] + ceterNi[-1] + Mo + Mn + Si + Ti 
He_ppm = He / (total_conc_AIM1/1e6)  
B10_ppm = B10 / (total_conc_AIM1/1e6) 
N14_ppm = N14 / (total_conc_AIM1/1e6) 
Ni58_ppm = Ni58 / (total_conc_AIM1/1e6) 
Ni59_ppm = Ni59 / (total_conc_AIM1/1e6) 

Bu = q0/(rho_MOX*pellet_section) * t/(24*60**2) #[MWd/ton]
print(f'La concentrazione di Elio nella guaina dopo un anno di irraggiamento risulta valere {He[-1]} at/m3, cioè {He_ppm[-1]} ppm ')

plt.figure(1)
plt.semilogy(t/(24*60**2), He, label='[ELIO TOTALE]', linewidth=2)
plt.semilogy(t/(24*60**2), He_B,linestyle='-.', label='[da Boro-10]')
plt.semilogy(t/(24*60**2), He_N,linestyle='-.', label='[da Azoto-14]')
plt.semilogy(t/(24*60**2), He_Ni,linestyle='-.', label='[da Nichel]')
plt.semilogy(t/(24*60**2), He_Fe,linestyle='-.', label='[da Ferro-56]')
plt.title('Evoluzione concetrazione di Elio nella guaina')
plt.xlabel('Tempo di esercizio [giorni]')
plt.ylabel('Elio [at/m3]')
plt.legend(loc='best')
plt.grid(True)

plt.figure(2)
plt.semilogy(Bu, He, label='[ELIO TOTALE]', linewidth=2)
plt.semilogy(Bu, He_B,linestyle='-.', label='[da Boro-10]')
plt.semilogy(Bu, He_N,linestyle='-.', label='[da Azoto-14]')
plt.semilogy(Bu, He_Ni,linestyle='-.', label='[da Nichel]')
plt.semilogy(Bu, He_Fe,linestyle='-.', label='[da Ferro-56]')
plt.title('Evoluzione concetrazione di Elio nella guaina')
plt.xlabel('Burnup [MWd/tonMOX]')
plt.ylabel('Elio [at/m3]')
plt.legend(loc='best')
plt.grid(True)

plt.figure(3)
plt.plot(t/(24*60**2), He_ppm, label='[Elio]')
# plt.plot(t/(24*60**2), B10_ppm, label='[Boro-10]')
# plt.plot(t/(24*60**2), N14_ppm, label='[Azoto-14]')
# plt.plot(t/(24*60**2), Ni58_ppm, label='[Nichel-58]')
# plt.plot(t/(24*60**2), Ni59_ppm, label='[Nichel-59]')
plt.title('Evoluzione concetrazioni nella guaina')
plt.xlabel('Tempo di esercizio [giorni]')
plt.ylabel('ppm')
plt.grid(True)
plt.legend(loc='best')


#%% TEST SECTION

