import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.optimize import root_scalar


def print_gay():
    print("gayyy")

# Define the function
R_f = sp.symbols('R_f', real=True, positive=True)
R_f_num = 1
D = sp.symbols('D', real=True, positive=True)
D_num = 0.01
alfa = sp.symbols('alpha', real=True, positive=True)
alfa_num = 10
r_star = sp.symbols('r*', real=True, positive=True)
r = sp.symbols('r', real=True, positive=True)

Pu_r = 1 + D * (sp.exp(-2 * alfa * (r - r_star) / R_f ) - 2 * sp.exp(-alfa * (r - r_star) / R_f))

int = sp.integrate(Pu_r * 2 * sp.pi * r, (r, 0, R_f))
int = int.subs([(D, D_num), (R_f, R_f_num), (alfa, alfa_num)])
eq = sp.Eq(int / (np.pi * R_f_num**2), 1)
r_star_num = sp.nsolve(eq, r_star, R_f_num / 2)
print(r_star_num)

lambda_r = sp.lambdify(r, Pu_r.subs([(D, D_num), (R_f, R_f_num), (alfa, alfa_num), (r_star, r_star_num)]), 'numpy')
lambda_r(0)
r = np.linspace(0, R_f_num, 1000)
plt.plot(r, lambda_r(r))
plt.xlabel('r')
plt.ylabel(r'$\frac{[Pu](r)}{Pu_0}$')
plt.show()
