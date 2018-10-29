import matplotlib.pyplot as plt
import numpy as np
import math as ma
import numpy.core.umath as np
import pylab


class Wave_function_handler:
    def __init__(self, k_0, x, x_s, m, L, w, sigma, hbar, E, dt, del_x):
        self.__k_0 = k_0
        self.__x = x
        self.__x_s = x_s
        self.__m = m
        self.__L = L
        self.__w = w
        self.__sigma = sigma
        self.__hbar = hbar
        self.__E = E
        self.__dt = dt
        self.__del_x = del_x

    def get_del_x(self):
        return self.__del_x

    def get_k_0(self):
        return self.__k_0

    def get_E(self):
        return self.__E
    def get_x(self):
        return self.__x

    def get_hbar(self):
        return self.__hbar

    def get_x_s(self):
        return self.__x_s

    def get_m(self):
        return self.__m

    def get_L(self):
        return self.__L

    def get_w(self):
        return self.__w

    def get_sigma(self):
        return self.__sigma

    def get_dt(self):
        return self.__dt

    def set_sigma(self, sigma):
        self.__sigma = sigma

    def set_del_x(self, del_x):
        self.__del_x = del_x

    def set_k_o(self, k_o):
        self.__k_o = k_o

    def set_w(self, w):
        self.__w = w

    def set_x_s(self, x_s):
        self.__x_s = x_s

    def set_hbar(self, hbar):
        self.__hbar = hbar

    def set_x(self, x):
        self.__x = x

    def set_e(self, E):
        self.__E = E

    def set_m(self, m):
        self.__m = m

    def c_factor(self):

        return 1/(np.pi*self.get_sigma()**2)**(1/4)



#Funksjoner ============================================================================================================#


    def gaussian(self):
        return self.c_factor() * np.exp(-(self.get_x() - self.get_x_s())**(2)/self.get_sigma()**(2))

    def calc_psi_r(self):

        return 0.05*self.gaussian() * np.cos(self.get_k_0()*self.get_x())

    def calc_psi_i(self):

        return 0.05*self.gaussian()*np.sin(self.get_k_0()*self.get_x() - self.get_w()*self.get_dt()/2)


    def calc_matrix(self):

        new_psi_r = self.calc_psi_r()
        new_psi_i = self.calc_psi_i()

        new_psi_r[0] = 0
        new_psi_r[-1] = 0
        new_psi_i[0] = 0
        new_psi_i[-1] = 0

        return new_psi_r + 1j * new_psi_i

    def plot_functions(self):
        psi_R = self.calc_psi_r()
        psi_I = self.calc_psi_i()
        plt.figure()
        plt.plot(self.get_x(), psi_R, 'm', label = '$\Psi(x, t)_{R}$' )
        plt.plot(self.get_x(), psi_I, 'lightgreen', label = '$\Psi(x, t)_{I}$')
        plt.title('$\Psi(x, t)_{I}, \Psi(x, t)_{I}$')
        plt.legend()
        plt.show()



    def Psi_dt(self,a, number_of_times):

        a_new = a
        import numpy as np
        for i in range(number_of_times):

            for n in range(np.size(a) - 2):
                n += 1

                a_new.imag[n] -= self.get_dt() * (a.real[n] - (1 / (2 * self.get_m() * (self.get_del_x() ** 2))) *

                                       (a.real[n + 1] - 2 * a.real[n] + a.real[n - 1]))

                a_new.real[n] += self.get_dt() * (a.imag[n] - (1 / (2 * self.get_m() * (self.get_del_x() ** 2))) *

                                       (a.imag[n + 1] - 2 * a.imag[n] + a.imag[n - 1]))

        return a_new




    def animate_functions(self, psi2):




        plt.clf()

        plt.plot(self.get_x(), psi2.real,label='$\Psi_I$', color = 'm')
        plt.plot(self.get_x(), psi2.imag,label='$\Psi_R$', color = 'lightgreen')
        plt.xlim((0, 20))
        plt.legend()
        plt.ylim((-0.1, 0.1))

        plt.pause(0.0001)

        plt.show()



    def animate(self):

        psi = self.calc_matrix()

        for i in range(1000):

            psi2 = self.Psi_dt(psi,10)

            self.animate_functions(psi2)
            plt.ion()  # Endrer bildet.


    def problem_2(self):

        for sigmax in [0.1, 0.2, 0.5, 1.0, 2.0]:
            self.animate_functions(sigmax)




'''
        plt.figure()
        plt.plot(self.get_x(), psi_R)
        plt.xlabel('$x$')
        plt.title('$\Psi(x, t)_{R}$')
        plt.figure()
        plt.plot(self.get_x(), psi_I)
        plt.xlabel('$x$')
        plt.title('$\Psi(x, t)_{I}$')
        plt.figure()
        plt.plot(self.get_x(), psi)
 - self.get_w()*self.get_dt()/2
'''


