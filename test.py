import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
import functions as fn


# test integrali

Ti = 1800
To = 1600

y_Pu  = 0.2
Bu = 0
x = 0.03
p = 0.5

k_int_an = fn.tc_integral(Ti, To, x, y_Pu, p, Bu)
k_int_sc = quad(fn.tc, To, Ti, args=(x, y_Pu, p, Bu))[0]

print("\nThermal conductivity integral")
print("Analytical: ", k_int_an)
print("Scipy: ", k_int_sc)

print("\nPlutonium redistribution")
R = 2.71e-3
R0 = 0
Ha = 0.85
V = np.pi*(R**2 - R0**2)*Ha
wt_Pu = 0.29
y0 = wt_Pu / V
D = 0.01
alpha = 10

print("r_star computation")
r, y_Pu_an, r_star_an = fn.radial_y_Pu(wt_Pu, R, R0)
print("Analytical: ", r_star_an)

y_Pu_sc = lambda r_star: y0 + y0*D*( np.exp(-2*alpha*(r-r_star)/R) - 2*np.exp(-alpha*(r-r_star)/R) )
def g(r_star):
    return np.trapz(r*y_Pu_sc(r_star), r) - y0*(R**2 - R0**2)/2

r_star_sc = fsolve(g, 0.01)[0]
print("Scipy: ", r_star_sc)


plt.plot(r, y_Pu_an, label="Analytical")
plt.plot(r, y_Pu_sc(r_star_sc), label="Scipy")
plt.legend()
plt.show()
