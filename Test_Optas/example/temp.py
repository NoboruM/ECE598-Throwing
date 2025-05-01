import numpy as np

x = 6
y = 2
theta = np.pi/4.0
g = 9.81
v = np.sqrt((x**2 * g)/(x*np.sin(2*theta) - 2*y*(np.cos(theta))**2))
print("v: ", v)
