import constants as c
from Wave_function_handler_one import Wave_function_handler
import time

start_time = time.clock()

#problem 1:
wave_obj = Wave_function_handler(c.k_0, c.x, c.x_s, c.m, c.L, c.w, c.sigma, c.hbar, c.E, c.dt, c.del_x)
wave_obj.animate()
wave_obj.plot_functions()

print(start_time)













#wave_obj.animate_functions()

#wave_obj_2 = Wave_functiob\n_handler/\(x,y )
#wave-obj-2.plot()
