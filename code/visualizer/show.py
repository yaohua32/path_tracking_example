import matplotlib.pyplot as plt
import numpy as np

class show():

    def __init__(self):
        plt.ion()

    def set_figure(self, x, y, slope=None, ax=None, ay=None, figsize=(10,5)):
        #
        plt.figure(figsize=figsize)
        #
        plt.plot(x, y, color='b', linewidth='4', label='ref path')
        # (ax, ay) are points which used to formulate the reference path
        if (ax is not None and ay is not None):
            plt.scatter(ax, ay, s=100, c='r', marker='*')
        # (slope) add the slope
        if slope is not None:
            plt.plot(slope[:,0], slope[:,1], color='r', label='slope')
        plt.legend()
        
    def update_figure(self, pos, pause_time=0.00001):
        #
        x, y = pos[0], pos[1]
        plt.scatter(x, y, color='r')
        plt.pause(pause_time)
    
    def add_arrow(self, pos, pos_d):
        #
        dx, dy = pos_d - pos 
        plt.quiver(pos[0], pos[1], dx, dy, angles='xy', scale=1, scale_units='xy')
    
    def hold_figure(self):
        #
        input('Press ENTER to continue!')
    
    def plot_u(self, u):
        #
        n_u = np.shape(u)[0]
        #
        plt.figure()
        for i in range(n_u):
            plt.plot(u[i], label='u%d'%i)
        plt.legend()
        plt.xlabel('time steps')
        plt.ylabel('controls')

    def plot_x(self, x):
        #
        n_s = np.shape(x)[0]
        #
        plt.figure()
        for i in range(n_s):
            plt.plot(x[i], label='X%d'%i)
        plt.legend()
        plt.xlabel('time steps')
        plt.ylabel('X')

    def plot_path(self, state, ctrl_name, ref_path=None):
        #
        plt.figure()
        plt.plot(state[0], state[1], label=ctrl_name)
        if ref_path is not None:
            plt.plot(ref_path[0], ref_path[1], label='reference')
        plt.legend()
        plt.xlabel('x(t)')
        plt.ylabel('y(t)')





