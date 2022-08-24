import numpy as np

class UAV(object):

    def __init__(self, L, dt):
        #
        self._L = L
        self._dt = dt 
        #
        self._ns = 4
        self._nu = 2
    
    @property 
    def ns(self):
        return self._ns 

    @property 
    def nu(self):
        return self._nu 
    
    @property
    def dt(self):
        return self._dt
    
    ###########################
    def set_x_ref(self, x_ref):
        # size: (N, ns)
        self.x_ref = x_ref 
        #
        self.x_d = x_ref[:,0]
        self.y_d = x_ref[:,1]
        self.psi_d = x_ref[:,2]
        self.v_d =  x_ref[:,3]

    def get_Abar(self, i):
        #
        Abar = np.eye(self._ns)
        Abar[0,2] = - self.v_d[i] * np.sin(self.psi_d[i]) * self._dt 
        Abar[0,3] = np.cos(self.psi_d[i]) * self._dt
        Abar[1,2] = self.v_d[i] * np.cos(self.psi_d[i]) * self._dt
        Abar[1,3] = np.sin(self.psi_d[i]) * self._dt

        return Abar
    
    def get_Bbar(self, i):
        #
        Bbar = np.zeros((self._ns, self._nu))
        Bbar[2,1] = self.v_d[i] * self._dt / self._L 
        Bbar[3,0] = self._dt

        return Bbar
    
    def dynamic(self, xt, ut, dt=None):
        #
        if dt is not None:
            x_new = np.zeros(xt.shape)
            #
            x_new[0] = xt[0] + xt[3]*np.cos(xt[2]) * dt 
            x_new[1] = xt[1] + xt[3]*np.sin(xt[2]) * dt 
            x_new[2] = xt[2] + xt[3]*np.tan(ut[1]) * dt / self._L
            x_new[3] = xt[3] + ut[0] * dt 
        else:
            x_new = np.zeros(xt.shape)
            #
            x_new[0] = xt[0] + xt[3]*np.cos(xt[2]) * self._dt 
            x_new[1] = xt[1] + xt[3]*np.sin(xt[2]) * self._dt 
            x_new[2] = xt[2] + xt[3]*np.tan(ut[1]) * self._dt / self._L
            x_new[3] = xt[3] + ut[0] * self._dt         

        return x_new

