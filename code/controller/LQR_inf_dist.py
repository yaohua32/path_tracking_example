import numpy as np

class LQR_controller():

    def __init__(self, Q_w, R_w, max_iter=100, eps=1e-3):
        #
        self.Q = np.diag(Q_w)
        self.R = np.diag(R_w)
        #
        self._max_iter = max_iter
        self._eps = eps

    def lqr(self, A, B):
        '''
        无限时间、离散系统
        '''
        P = self.Q 
        P_next = self.Q
        for _ in range(self._max_iter):
            P_next = (A.T @ P @ A -
                      A.T @ P @ B @ 
                      np.linalg.inv(self.R + B.T @ P @ B) @
                      B.T @ P @ A + self.Q)
            if (abs(P_next - P)).max() < self._eps:
                break 
            P = P_next
        #
        K = np.linalg.inv(B.T @ P @ B + self.R) @ (B.T @ P @ A)

        return K, P