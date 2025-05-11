import numpy as np
import matplotlib.pyplot as plt
import time

def Compute1PolyTraj(t, q0, dq0, qf, dqf, t0, tf):
    A = np.array([
        [1, t0, t0**2, t0**3],
        [0, 1, 2*t0, 3*t0**2],
        [1, tf, tf**2, tf**3],
        [0, 1, 2*tf, 3*tf**2]
    ])
    B = np.array([q0, dq0, qf, dqf])
    a0, a1, a2, a3  = np.linalg.solve(A, B)
    print("linalg: ", np.linalg.solve(A, B))
    return a0 + a1*t + a2*t**2 + a3*t**3

d = 3

Z = -0.5
v_0 = np.sqrt(9.81*d**2/(d - Z))
t = np.arange(0, 4, 0.01)
x = v_0*np.sqrt(2)/2.0*t
y = v_0*np.sqrt(2)/2.0*t - 1/2.0*9.81*t**2

while (True):
    print(time.time())
# plt.plot(x,y)
# plt.xlim((0, 3))
# plt.ylim((0, 1))
# plt.show()