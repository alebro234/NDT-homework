import numpy as np
import matplotlib.pyplot as plt
from functions import *

L = 2

phi = lambda x, n: np.sqrt(2/L)*np.sin(n*np.pi/L*x)


x = np.linspace(0, L, 1000)

# check orthogonality
I12 = np.trapz(phi(x, 1)*phi(x, 5), x)
I11 = np.trapz(phi(x, 5)*phi(x, 5), x)

print(I12, I11)
