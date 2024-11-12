import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

n_z = 100
n_t = 100

z = np.linspace(0,2,n_z)
t = np.linspace(0,1,n_t)

def dydt(t,y):
    return z*y

y0 = np.ones(n_z)
t_span = (0,1)
sol = solve_ivp(dydt, t_span, y0, t_eval = t)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
z, t = np.meshgrid(z,t)
ax.plot_surface(z,t,sol.y)
plt.xlabel("z")
plt.ylabel("t")
plt.legend()

plt.show()


