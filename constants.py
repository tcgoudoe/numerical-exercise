#list of constants
import numpy as np
k_0 = 20.
hbar = 1.
m = 1.
L = 20.
N_x = int(2.5*k_0*L) + 1

del_x = L /(N_x-1)

x_s = 5.
v_g = hbar**2*k_0**2/m
sigma = 0.5
rho = 1    #scaling dt woop woop

c = 2
E = hbar**2*k_0**2/2*m
w = E/hbar
dt = rho  *  2*m*hbar*del_x**2 / (2*m*c*E*del_x**2 + hbar**2)
#dt = 0.1 * 2 * m * (del_x ** 2)  # for stabilitet
x = np.linspace(0 , L, N_x)