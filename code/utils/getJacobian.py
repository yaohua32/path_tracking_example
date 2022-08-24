import autograd.numpy as np
from autograd import grad, jacobian, hessian

class GetJacob(object):

    def __init__(self, fun, cost, costf):
        # The return of fun is a vector
        self._jacob_f = jacobian(fun)
        # The return of cost is a scalar
        self._grad_c = grad(cost)
        self._hess_c = hessian(cost)
        # The return of costf is a scalar
        self._grad_cf = grad(costf)
        self._hess_cf = hessian(costf)

    def grad_f(self, xbar, ubar):
        # Compute the grad of fun w.r.t (x,u)
        ns, nu = xbar.shape[0], ubar.shape[0]
        xu = np.vstack([xbar.reshape(-1,1), ubar.reshape(-1,1)])
        #
        df = self._jacob_f(xu).reshape((ns, ns+nu))
        #
        f_x = df[0:ns, 0:ns]
        f_u = df[0:ns, ns:ns+nu]

        return f_x, f_u

    def grad_c(self, xbar, ubar):
        # Compute the grad of cost w.r.t (x,u)
        ns, nu = xbar.shape[0], ubar.shape[0]
        xu = np.vstack([xbar.reshape(-1,1), ubar.reshape(-1,1)])
        #
        dc = self._grad_c(xu)
        #
        c_x = dc[0:ns,:]
        c_u = dc[ns:ns+nu,:]

        return c_x, c_u 

    def hess_c(self, xbar, ubar):
        # Compute the hessian of cost w.r.t (x,u)
        ns, nu = xbar.shape[0], ubar.shape[0]
        xu = np.vstack([xbar.reshape(-1,1), ubar.reshape(-1,1)])
        #
        Hc = self._hess_c(xu).reshape((ns+nu, ns+nu))
        #
        Hc_xx = Hc[0:ns,0:ns]
        Hc_ux = Hc[ns:ns+nu,0:ns]
        Hc_uu = Hc[ns:ns+nu, ns:ns+nu]

        return Hc_xx, Hc_ux, Hc_uu

    def grad_cf(self, xbar):
        # Compute the grad of costf w.r.t (x)
        xbar = xbar.reshape(-1,1)
        #
        cf_x = self._grad_cf(xbar)

        return cf_x

    def hess_cf(self, xbar):
        # Compute the hessian of costf w.r.t (x,x)
        ns = xbar.shape[0]
        xbar = xbar.reshape(-1,1)
        #
        Hcf = self._hess_cf(xbar).reshape((ns, ns))
        #
        Hcf_xx = Hcf

        return Hcf_xx

    def _test_fun(self):
        '''
        An example for testing autograd for "vector function"
        '''
        def fun(xu):
            # vector function
            # xu (3, n)
            # x: (2, n)
            # u: (1, n)
            x = xu[0:2,:]
            u = xu[2,:]
            #
            z1 = x[0,:]**2 
            z2 = x[1,:]*u
            
            return  np.vstack([z1, z2])
        #####################
        grad_fun = jacobian(fun)
        hess_fun = hessian(fun)
        x0 = np.array([[1.], [2.], [0.5]])
        ##################### Compute Jacobian (scalar)
        J = grad_fun(x0).reshape(2,3)
        print('Grad scalar:\n', J)
        J_x = J[0:2,0:2]
        print('J_x:\n', J_x)
        J_u = J[0:2,2:3]
        print('J_u:\n', J_u)

    def _test_cost(self):
        '''
        An example for testing autograd for "scalar function"
        '''
        def cost(xu):
            # scalar function
            # xu (3, n)
            # x: (2, n)
            # u: (1, n)
            x = xu[0:2,:]
            u = xu[2,:]
            #
            z = x[0,:]**2 + x[1,:]*u

            return  z
        #####################
        grad_cost = grad(cost)
        hess_cost = hessian(cost)
        x0 = np.array([[1.], [2.], [0.5]])
        ##################### Compute gradient (scalar)
        J = grad_cost(x0)
        print('Grad scalar:\n', J)
        J_x = J[0:2,:]
        print('J_x:\n', J_x)
        J_u = J[2:3,:]
        print('J_u:\n', J_u)   
        ##################### Compute hessian (scalar)
        H = hess_cost(x0).reshape(3,3)
        print('hessian scalar:\n', H)
        H_xx = H[0:2,0:2]
        print('H_xx:\n', H_xx)
        H_xu = H[0:2,2:3]
        print('H_xu:\n', H_xu)
        H_uu = H[2:3,2:3]
        print('H_uu:\n', H_uu)   

if __name__=='__main__':
    GetJacob(None, None)._test_fun()
    GetJacob(None, None)._test_cost()