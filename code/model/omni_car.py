import autograd.numpy as anp
import numpy as np

class OmniCar(object):
    
    def __init__(self, mass, friction, dt):
        # 
        self._m = mass
        self._alpha = friction
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
    
    @property 
    def f(self):
        return self._f

    @property
    def c(self):
        return self._c
    
    @property
    def cf(self):
        return self._cf

    ############################# Xbar = Abar * Xbar + Bbar * Ubar (Used for control)
    ############################# where Xbar = [x - x_d; 1], Ubar = U
    def get_Abar(self, i):
        # 扩展后用于控制算法的 Abar
        A = np.array([[1, 0, self._dt, 0],
                      [0, 1, 0, self._dt],
                      [0, 0, 1 - self._dt * self._alpha/self._m, 0],
                      [0, 0, 0, 1- self._dt * self._alpha/self._m]])
        #
        Abar = np.eye(5)
        Abar[:4,:4] = A
        Abar[:4,4] = np.dot(A, self.x_ref[i]) - self.x_ref[i+1] 
        Abar[4,4] = 1

        return Abar
    
    def get_Bbar(self, i):
        # 扩展后用于控制算法的 Bbar
        B = np.array([[0, 0],
                      [0, 0],
                      [self._dt/self._m, 0],
                      [0, self._dt/self._m]])
        #
        Bbar = np.zeros((5,2))
        Bbar[:4,:] = B

        return Bbar

    def set_x_ref(self, x_ref):
        #
        self.x_ref = x_ref
    
    def dynamic(self, xt, ut):
        #
        p = xt[0:2] 
        v = xt[2:self._ns]
        #
        p_dot = v 
        v_dot = (ut - self._alpha * v) / self._m
        #
        X_dot = np.concatenate([p_dot, v_dot])
     
        return X_dot * self._dt + xt

   ################################## X_new = f(X_old, U, t)

    def set_running_cost(self, U_ref=None, X_ref=None):
        '''
        Define running cost
        '''
        def running_cost(XU):
            #
            X = XU[0:self._ns].reshape((self._ns, 1))
            U = XU[self._ns:].reshape((self._nu, 1))
            #
            p = X[0:2,:]       
            v = X[2:self._ns,:] 
            #
            Loss = anp.sum( (X)**2 ) + 100 * anp.sum( (U)**2 )
            return Loss 
        
        self._c = running_cost

    def set_final_cost(self, X_ref=None):
        '''
        Define the final cost!
        '''
        def final_cost(Xf):
            # 
            Xf = Xf.reshape((self._ns, 1))
            #
            pf = Xf[0:3,:]   
            vf = Xf[3:6,:]
            #
            Lf = 0. * anp.sum(Xf**2)
            return Lf
        
        self._cf = final_cost


    def set_dynamic(self):
        '''
        Define the dynamics here!
        '''
        def dynamic(XU):
            X = XU[0:self._ns].reshape((self._ns, 1))
            U = XU[self._ns:].reshape((self._nu, 1))
            #
            p = X[0:2,:]        
            v = X[2:self._ns,:]
            #
            p_dot = v 
            v_dot = (U - self._alpha * v) / self._m
            #
            X_dot = anp.vstack([p_dot, v_dot])

            return X_dot * self._dt + X

        self._f = dynamic
