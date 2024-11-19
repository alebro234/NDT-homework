import numpy as np

BC = {"type": "D", "value": (1, 2)}
BCtype, BCvalues = BC.values()
u0, U0 = BCvalues

N = 5
C = 1
A = np.diag(np.ones(N) * C)
for i in range(N):
    if i > 0:
        A[i, i - 1] = C + 1
    if i < (N - 1):
        A[i, i + 1] = C + 2

print(A)

