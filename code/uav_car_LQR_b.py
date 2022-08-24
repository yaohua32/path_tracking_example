import numpy as np

from model.uav_car import UAV 
from controller import LQR_inf_dist
from path.refer_path import Path
from visualizer import show

def u_constraint(u_opt):
    # 对u=[a,\delta]的限制
    u_opt[0] = np.clip(u_opt[0], -5, 5)
    u_opt[1] = (u_opt[1]+np.pi)%(2*np.pi) - np.pi

    return u_opt

def solver(model, lqr, x0, N):
    #
    u_opt = np.zeros((N, model.nu))
    x_opt = np.zeros((N+1, x0.shape[0]))
    #
    x_opt[0] = x0
    x_diff = x0 - model.x_ref[0]
    #
    cost_rec = []
    for k in range(N): 
        #
        A = model.get_Abar(k)
        B = model.get_Bbar(k)
        gain_K, P = lqr.lqr(A, B)
        # forward
        delta_u = - np.dot(gain_K, x_diff)
        u_opt[k] = u_constraint(delta_u)
        x_opt[k+1] = model.dynamic(x_opt[k], u_opt[k])
        #
        x_diff = x_opt[k+1] - model.x_ref[k+1]
        # cost
        cost = np.sum(x_diff *  (P @ x_diff) )
        cost_rec.append(cost)

    return x_opt, u_opt, cost_rec

def main():
    #
    car = UAV(L=1., dt=0.02)
    Q_w = [1., 1., 0., 0.]
    R_w = [0.5, 0.5]  # 重要注释：调大R_w可以对x0[1]较大时有很大拉回作用！！！
    ctrl_lqr = LQR_inf_dist.LQR_controller(Q_w, R_w)
    
    # 参考路径 （可以考虑路径追踪的问题）
    N = 500
    path = Path()
    path.set_ref_path(N=N, path_inx = 1)
    refer_path = path.ref_path
    psi, slope = path.get_slope(refer_path)
    #（Xd=[x_d, y_d, \psi_d, v_d], ud=[0,0]）
    x_ref = np.zeros((refer_path.shape[0], 4))
    x_ref[:,0:2] = refer_path  
    x_ref[:,2:3] = psi
    x_ref[:,3:] =  1.5 * np.ones((N+1, 1))
    car.set_x_ref(x_ref)

    # 求解问题
    x0 = np.array([0, .5, 0., 0.])
    x_opt, u_opt, cost_list = solver(car, ctrl_lqr, x0, N)

    ###### visualization (dynamics with arrow)
    drawer = show.show()
    drawer.set_figure(refer_path[:,0], refer_path[:,1], slope)
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