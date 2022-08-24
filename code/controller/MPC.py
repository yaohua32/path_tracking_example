import numpy as np
import quadprog
import scipy.optimize
'''
from quadprog import solve_qp
#
min 0.5 *(xT @ G @ x) - aT @ x
s.t. - CT @ x + b <= 0
'''

def solve_qp_scipy(G, a, C, b, meq):
    def f(x):
        return 0.5 * np.dot(x, G).dot(x) - np.dot(a, x)
    #
    constraints = []
    if C is not None:
        constraints = [{
            'type': 'eq' if i < meq else 'ineq',
            'fun': lambda x, C=C, b=b, i=i: (np.dot(C.T, x) - b)[i]
        } for i in range(C.shape[1])]
    #
    result = scipy.optimize.minimize(
        f, x0=np.zeros(len(G)), method='SLSQP', constraints=constraints,
        tol=1e-10, options={'maxiter': 2000})
    #
    return result

class MPC_controller():
    
    def __init__(self, model, N, P, Q_w, R_w):
        #
        self._model = model
        self.N = N # control horizon
        self.P = P # predict horizon
        #
        self.Q = np.diag(Q_w * P)
        self.R = np.diag(R_w * N)

    def get_AA(self, A_tilde_list):
        #
        ns, nu = self._model.ns, self._model.nu
        ns_tilde = ns+1+nu
        AA = np.zeros((self.P * ns_tilde, ns_tilde))
        #
        IA_block = np.eye(ns_tilde)
        for k in range(self.P):
            IA_block = IA_block @ A_tilde_list[k]
            AA[k*ns_tilde:(k+1)*ns_tilde,:] = IA_block

        return AA

    def get_BB(self, A_tilde_list, B_tilde_list):
        #
        ns, nu = self._model.ns, self._model.nu
        ns_tilde = ns+1+nu
        BB = np.zeros((self.P*ns_tilde, self.N*nu))
        # 我们(按行来)来实现BB
        for k in range(self.P):
            if k<self.N:
                # 处理对角线上的
                BB[k*ns_tilde:(k+1)*ns_tilde, k*nu:(k+1)*nu] = B_tilde_list[k]
                # 处理非对角线上的(按行来)
                for j in range(0, k-1):
                    B_block = BB[(k-1)*ns_tilde:k*ns_tilde, j*nu:(j+1)*nu]
                    A_block = A_tilde_list[k]
                    BB[k*ns_tilde:(k+1)*ns_tilde, j*nu:(j+1)*nu] = A_block @ B_block
            else:
                # 处理非对角线上的（按行来）
                for j in range(0, self.N):
                    B_block = BB[(k-1)*ns_tilde:k*ns_tilde, j*nu:(j+1)*nu]
                    A_block = A_tilde_list[k]
                    BB[k*ns_tilde:(k+1)*ns_tilde, j*nu:(j+1)*nu] = A_block @ B_block  
        # print('BB:\n', BB)

        return BB

    def get_constraint(self, AA, BB, Constraint_dict, X0_tilde):
        ##### CC.shape = (2*P*ns_tilde+2*N*nu+2*N*nu, N*nu)
        ##### bb.shape = (2*P*ns_tilde+2*N*nu+2*N*nu, 1)
        ns, nu = self._model.ns, self._model.nu
        U_old = X0_tilde[ns+1:]
        Xmax = Constraint_dict['Xmax']
        Xmin = Constraint_dict['Xmin']
        Umax = Constraint_dict['Umax']
        Umin = Constraint_dict['Umin']
        dUmax = Constraint_dict['dUmax']
        dUmin = Constraint_dict['dUmin']

        # constraint for Y
        Ymax = np.vstack([Xmax, np.array([[1.]]), Umax] * self.P )
        Ymin = np.vstack([Xmin, np.array([[1.]]), Umin] * self.P )
        CC_Y = np.vstack([BB, -BB])
        bb_Y = np.vstack([Ymax - AA @ X0_tilde,
                          AA @ X0_tilde - Ymin])

        # constraint for U
        Cu = np.tril(np.ones(self.N * nu), k=0)
        CC_U = np.vstack([Cu, -Cu])
        bb_U = np.vstack([np.vstack([Umax-U_old]*self.N),
                          np.vstack([U_old-Umin]*self.N)])

        # constraint for dU
        I = np.eye(self.N * nu) 
        CC_I = np.vstack([I, -I])
        bb_I = np.vstack([np.vstack([dUmax]*self.N),
                          np.vstack([-dUmin]*self.N)])

        # 汇总
        CC = np.vstack([CC_Y, CC_U, CC_I])
        bb = np.vstack([bb_Y, bb_U, bb_I])
        # print('Shape of CC:', CC.shape)
        # print('CC:\n', CC)
        # print('shape of bb:', bb.shape)
        # print('bb:\n', bb)
        #
        return CC, bb

    def get_control(self, A_tilde_list, B_tilde_list, 
                          X0_tilde, Xd_tilde, Constraint_dict):
        # Here, we solve the optimal control problem
        AA = self.get_AA(A_tilde_list)
        # print('AA:\n', AA)
        BB = self.get_BB(A_tilde_list, B_tilde_list)
        # print('BB:\n', BB)
        #
        GG =  2 * BB.T @ self.Q @BB + self.R
        aa = - 2 * BB.T @ self.Q @ (AA @ X0_tilde - Xd_tilde)
        #
        CC, bb = self.get_constraint(AA, BB, Constraint_dict, X0_tilde)
        CC =  - CC.T
        bb = - bb
        #############  利用quadprog进行求解
        dU_opt = quadprog.solve_qp(GG, aa.flatten(), CC, bb.flatten())[0]
        ############# 利用scipy.optimize进行求解
        # result = solve_qp_scipy(GG, aa.flatten(), CC, bb.flatten(), meq=0)
        # dU_opt = result.x
        #
        # print('shape of dU_opt', dU_opt.shape)
        # print('dU_opt:\n', dU_opt)

        return dU_opt