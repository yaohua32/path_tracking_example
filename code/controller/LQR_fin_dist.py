import numpy as np

class LQR_controller():

    def __init__(self, Q_w, R_w):
        #
        self.Q = np.diag(Q_w)
        self.R = np.diag(R_w)

    def lqr(self, model, N):
        '''
        适用于有限时域、离散系统; N为预测总步数
        '''
        #
        P = (N+1) * [self.Q]
        K = (N) * [0]
        for n in range(N, 0, -1):
            #
            Abar = model.get_Abar(n-1)
            Bbar = model.get_Bbar(n-1)
            # 
            K[n-1] = Bbar.T @ P[n] @ Abar
            F = np.linalg.inv(self.R + Bbar.T @ P[n] @ Bbar)
            K[n-1] = F @ K[n-1]
            #
            C = Abar + Bbar @ (-K[n-1])
            E = (-K[n-1].T) @ self.R @ (-K[n-1])
            P[n-1] = self.Q + E + C.T @ P[n] @ C
        
        return K, P
