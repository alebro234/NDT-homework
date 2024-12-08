import functions as fn
import numpy as np
import matplotlib.pyplot as plt

Ha = 0.85
R = 2.71e-3
R0 = 0.5e-3
wt_Pu = 0.29
r, y_Pu, r_star = fn.radial_y_Pu(wt_Pu, R, R0)

print("r_star: ", r_star)

y_avg = fn.y_Pu_avg(y_Pu, R0, R/2)
print("y_avg: ", y_avg)

plt.plot(r*1000, y_Pu)
plt.xlabel("r [mm]")
plt.ylabel("y_Pu")
plt.show()
