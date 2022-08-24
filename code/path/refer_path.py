import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import splrep
from scipy.interpolate import splev

class Path():
    
    def __init__(self):
        #
        pass
    
    @property
    def ref_path(self):
        return self._ref_path

    @property
    def spline(self):
        return self._spline
        
    @property 
    def ref_head(self):
        return self._ref_head

    @property
    def ax(self):
        return self._ax 
    
    @property 
    def ay(self):
        return self._ay

    def set_ref_path(self, N=2, path_inx=0):
        #
        if path_inx==1:
            ############## path 1: spline function
            self._ax = np.array([0., 6., 10, 12.5, 17.5, 20., 25., 30.])
            self._ay = np.array([0., -1., -1., 1, 1., 0., 0., 0.])
            self._spline = splrep(self._ax, self._ay)
            x = np.linspace(0, 20, N+1)
            y = splev(x, self._spline)
        elif path_inx==2:
            ##############  path 2
            n1, n2 = int(N/2), N - int(N/2)+1
            x = np.concatenate([np.linspace(10, 20, n1), 
                                np.array([20 for _ in range(n2)])],
                                axis=0 )
            y = np.concatenate([np.array([25. for _ in range(n1)]), 
                                25.0 - np.linspace(0, 10, n2)],
                                axis=0 )
        elif path_inx==3:
            ############# path 3: sin function
            x = np.linspace(0, 20, N+1)
            y = np.sin(x)
        else:
            ##############  default path
            x = np.linspace(0, 10, N+1)
            y = np.array([0 for _ in range(N+1)])
        ###############
        self._ref_path = np.vstack((x,y)).T
        self._ref_tree = KDTree(self._ref_path)

    def get_slope(self, path):
        # (N,2)
        slope = np.zeros(path.shape)
        slope[:,0] = path[:,0]
        #
        for k in range(path.shape[0]-1):
            slope_k = (path[k+1][1] - path[k][1]) / (path[k+1][0] - path[k][0])
            slope[k,1] = slope_k 
        #
        slope[-1,1] = slope_k
        #
        psi = np.arctan(slope[:,1:])

        return psi, slope


    def get_ref_point_pure_pursuit(self, pind, p, ld):
        #
        for j in range(pind, len(self._ref_path)):
            dist = np.linalg.norm(p - self._ref_path[j])
            if dist >= ld:
                pind = j
                break
        return pind
    
    def get_ref_point_stanley(self, pind, p):
        #
        _, ind = self._ref_tree.query(p)
        if pind > ind:
            return pind
        else:
            return ind
    
    def get_ref_point_mpc(self, ind, p):
        #
        return ind




