import numpy as np
import scipy.io

from model.omni_car import OmniCar
from controller import LQR_fin_dist
from path.refer_path import Path
from visualizer import show

def u_constraint(u_opt):
    # 对控制的限制
    u_opt[0] = np.clip(u_opt[0], -0.2, 0.2)
    u_opt[1] = np.clip(u_opt[1], -0.2, 0.2)

    return u_opt

def solver(model, lqr, x0, N):
    #
    gain_K, P = lqr.lqr(model, N)
    #
    u_opt = np.zeros((N, model.nu))
    x_opt = np.zeros((N+1, x0.shape[0]))
    #
    x_opt[0] = x0
    x_diff = np.concatenate([x0-model.x_ref[0], np.array([1.])])
    #
    cost_rec = []
    for k in range(N): 
        # forward
        u_opt[k] = - np.dot(gain_K[k], x_diff)
        u_opt[k] = u_constraint( u_opt[k] )
        x_opt[k+1] = model.dynamic(x_opt[k], u_opt[k])
        #
        x_diff = np.concatenate([x_opt[k+1]-model.x_ref[k+1], np.array([1.])])
        # cost
        cost = np.sum(x_diff *  (P[k+1] @ x_diff) )
        cost_rec.append(cost)

    return x_opt, u_opt, cost_rec

def main():
    # omni car model
    car = OmniCar(mass=1., friction=0.1, dt=0.1)
    Q_w = [1., 1., 1., 1., 0.]
    R_w = [1., 1.]
    ctrl_lqr = LQR_fin_dist.LQR_controller(Q_w, R_w)

    # 参考路径 （可以考虑路径追踪的问题）
    N = 2500
    path = Path()
    path.set_ref_path(N=N, path_inx = 3)
    refer_path = path.ref_path
    #（xd）
    x_ref = np.zeros((refer_path.shape[0], 4))
    x_ref[:,0:2] = refer_path 
    car.set_x_ref(x_ref)

    # 求解问题
    # x0 = np.array([10, 30, 0., 0.])  # path_inx=2
    x0 = np.array([0, 1, 0, 0]) # path_inx =1
    x_opt, u_opt, cost_list = solver(car, ctrl_lqr, x0, N)
    #
    print('x_opt:\n', x_opt.shape)
    print('u_opt:\n', u_opt.shape)

    # 存储数据
    data_lqr = {}
    data_lqr['x_opt'] = x_opt 
    data_lqr['u_opt'] = u_opt 
    data_lqr['cost'] = cost_list
    scipy.io.savemat('data_lqr.mat', data_lqr)
    
    ###### visualization (dynamics with arrow)
    drawer = show.show()
    # drawer.set_figure(refer_path[:,0], refer_path[:,1])
    # #
    # for k in range(N+1):
    #     drawer.update_figure(x_opt[k])
    #     drawer.add_arrow(x_opt[k,0:2], refer_path[k])
    ###### visualization (static p & u)
    drawer.plot_u(u_opt.T)
    drawer.plot_path(x_opt.T)
    drawer.plot_path(x_opt.T, refer_path.T)
    drawer.hold_figure()


if __name__=='__main__':
    main()