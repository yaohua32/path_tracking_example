import numpy as np

class UAV(object):

    def __init__(self, L, dt):
        #
        self._L = L
        self._dt = dt 
        #
        self._ns = 3
        self._nu = 2
    
    @property 
    def L(self):
        return self._L
    
    @property 
    def ns(self):
        return self._ns 

    @property 
    def nu(self):
        return self._nu 
    
    @property
    def dt(self):
        return self._dt
    
    @property
    def u_ref(self):
        return self._u_ref
    
    @u_ref.setter 
    def u_ref(self, u_ref):
        self._u_ref = u_ref
    
    @property 
    def x_ref(self):
        return self._x_ref 
    
    @x_ref.setter 
    def x_ref(self, x_ref):
        self._x_ref = x_ref
    
    ###########################
    def set_x_ref(self, x_ref):
        # size: (N, ns)
        self._x_ref = x_ref 
    
    def set_u_ref(self, u_ref):
        # size: (nu,)
        self._u_ref = u_ref

    def get_Abar(self, i):
        #
        v_d = self.u_ref[i,0]
        psi_d = self.x_ref[i,2]
        #
        Abar = np.eye(self._ns)
        Abar[0,2] = - v_d * np.sin(psi_d) * self._dt 
        Abar[1,2] = v_d * np.cos(psi_d) * self._dt

        return Abar
    
    def get_Bbar(self, i):
        #
        v_d = self.u_ref[i,0]
        psi_d = self.x_ref[i,2]
        #
        Bbar = np.zeros((self._ns, self._nu))
        Bbar[0,0] = np.cos(psi_d) * self._dt 
        Bbar[1,0] = np.sin(psi_d) * self._dt
        Bbar[2,0] = 0.
        Bbar[2,1] = v_d * self._dt / self._L 

        return Bbar
    
    def get_Cbar(self, i):
        #
        v_d = self.u_ref[i,0]
        psi_d = self.x_ref[i,2]
        #
        Cbar = np.zeros((self._ns, 1))
        Cbar[0,0] = v_d * psi_d * np.sin(psi_d) * self._dt 
        Cbar[1,0] = -v_d * psi_d * np.cos(psi_d) * self._dt 
        Cbar[2,0] = 0.

        return Cbar

    
    def dynamic(self, xt, ut, dt=None):
        #
        if dt is not None:
            x_new = np.zeros(xt.shape)
            x_new[0] = xt[0] + ut[0]*np.cos(xt[2]) * dt 
            x_new[1] = xt[1] + ut[0]*np.sin(xt[2]) * dt 
            x_new[2] = xt[2] + ut[0]*np.tan(ut[1]) * dt / self._L
        else:
            x_new = np.zeros(xt.shape)
            x_new[0] = xt[0] + ut[0]*np.cos(xt[2]) * self._dt 
            x_new[1] = xt[1] + ut[0]*np.sin(xt[2]) * self._dt 
            x_new[2] = xt[2] + ut[0]*np.tan(ut[1]) * self._dt / self._L           

        return x_new

