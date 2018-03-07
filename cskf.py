import numpy as np
import scipy.io as sio
from time import time
from scipy.sparse.linalg import eigsh
import math
#from IPython.core.debugger import Tracer; debug_here = Tracer()
from pdb import set_trace
from scipy.io import savemat, loadmat

__all__ = ['CSKF']

class CSKF:
        
    """
            Find the optimal estimate for the inversion problem with the given 
            noisy set of observation and the approximate forward model 
    """

    def __init__(self, forward_model, observation_model, initial_dist, params, H, nx, n_step, rank, trend=None,true_sol = None, grid = None, R = None, Q = None, P = None, obs = None, lin = None):


        ### Forward model
        ## CSKF treats forward model as a given black box
        
        self.forward_model = forward_model

        ## Observation model, can be a linear matrix or a non linear operator

        self.observation_model = observation_model
        

        # Observation

        if obs is None:
            self.Syntethic = True
        else:
            if obs.ndim ==1:
                obs = np.array(obs)
                self.obs = obs.reshape(-1,1)
            elif obs.ndim ==2:
                if obs.shape[1] != n_step:
                    raise ValueError('the shape of observation shoul be n by n_step')
                self.obs = np.array(obs)
            self.Syntethic =False

        self.initial_dist = initial_dist
        self.H = H

        if grid is not None:
            self.grid = None

        if Q is not None:
            self.Q = Q
        elif Q is None and grid is None:
            raise ValueError('Either Q or grid points should be given')
        else:

            if 'kernel' in params:
                kernel = params['kernel']
            else:
                kernel = None

            if 'beta' in params:
                beta = params['beta']
            else:
                beta = 1

            self.Q = ConstructCovarianceMatrix(self.grid, kernel, beta)

        if R is None:
            self.R = np.eye(np.obs.shape[0])
        elif not isinstance(R, (float, np.ndarray)):
            raise ValueError('R should be a float number or numpy array')
        else:
            self.R = R

        #if self.R.ndim ==0 or self.R.ndim ==1:
         #   self.R = self.R.reshape(-1)
        
        if self.R.ndim>2:
            raise ValueError('R should be a scalar or a 2D array')

        if (self.R<=0).all():
            raise ValueError('R should be positive semidefinite')
        
        if P is None:
            self.P = self.Q
        else:
            self.P = P

        
        self.nx = nx
        self.trend = trend
        self.n_step = n_step
        self.rank = rank
        self.result =[]
        self.prior = initial_dist
        self.params = params
        self.lin = lin
        self.m = int(np.prod(nx))
        self.uncert = []
        
        if 'parallel' in params:
            self.parallel = params['parallel']
            
            if 'ncores' in params:
                self.ncores = params['ncores']
            else:
                from psutil import cpu_count
                self.ncores = cpu_count(logical = False)
        else:
            self.parallel = False
            self.ncores = 1

        if obs is None:
            self.n = 0
        else:
            self.n = obs.shape[0]
            if obs.ndim != 2:
                raise ValueError('obs should be n by n_step')
            if obs.shape[1] != n_step:
                raise ValueError('obs should be n by n_step = ', n_step, " obs_1 = ", obs.shape[1])
        
        ### generate synthetic measurements
        if true_sol is not None and self.Syntethic:
            print(' Measurements are not given and CSKF generates synthetic measurements based on the true solution')
            noise = True
            R_diag = True
            self.obs = self.create_synthetic_obs(self.R, true_sol, self.n_step, noise, R_diag)

        print("------------ CSKF Parameters -------------------------")
        print("    Number of unknowns                     :  %d" %(self.m))
        print("    Number of steps                        :  %d" %(self.n_step))
        print("    Number of principle components         :  %d" %(self.rank))
        print("    Number of observations                 :  %d" %(self.R.shape[0]))
        print("    Number of CPU cores                    :  %d" %(self.ncores))
        



    def forward_run(self, s):
        par = False
        simul_obs = self.forward_model(s,par)

        return simul_obs


    def parallel_forward_run(self, V):
        par =True
        simul_obs_parallel = self.forward_model(V,par,ncores = self.ncores)	

        return simul_obs_parallel

    def ConstructCovarianceMatrix(self, kernel = 'Gaussian', beta = None, grid = None):
        print('###----- Constructing prior covariance matrix-------####')
        start = time()
        Q = self.CovarianceMatrix(grid, kernel, beta)
        finish = time()
        print('Time for covariance matrix construction is %f seconds' %(finish - start))
        return Q



    def cov_update_pred(self, C_p, C_q):
        ### to be implemented: check the rank at each step
        self.C_p = C_p+C_q

    def compute_low_rank(self, rank = None, method ='SVD'):
        """
        Compute low rank decomposition of the model error covariance matrix
        """
        m = self.m
        n = self.n

        print("computing low rank representation of the model error covariance matrix")
        if rank is None:
            if self.rank is None:
                raise ValueError('Rank should be assigned')
            rank = self.rank

        if self.Q is None:
            raise ValueError('Q should be assigned')


        start = time()
        if method == 'SVD':

            u,C_p,v= np.linalg.svd(self.P)
            self.U = u[:,:rank]
            self.C_p = np.diag(C_p[:rank])
            self.C_q = np.diag(C_p[:rank])
        
        elif method == 'arpack':
            C_p, u = eigsh(self.Q, k = rank)
            self.U = u[:,:rank]
            self.C_p = np.diag(C_p[:rank])
            self.C_q = np.diag(C_p[:rank])

            if (C_p >0).sum() <rank:
                self.rank = (C_p >0).sum()
                self.U = self.U[:,:self.rank]
                self.C_p = np.diag(C_p[:rank])
                self.C_q = np.diag(C_p[:rank])
                print("Warning: rank changed to %d for positive eigenvalues" % (self.rank))

        elif method == 'randomized':
            raise NotImplementedError

        elif method == 'DCT':
            raise NotImplementedError

        else:
            raise NotImplementedError

        print('time for obtaining low rank approximation with rank = %d was %g' %(rank, time()-start))

        return self.U, self.C_p, self.C_q

    
    def create_synthetic_obs(self, R,  n_step = None, true_sol = None, noise = None, R_diag = None):
        """
        Create synthetic observations for synthetic problems
        """

        if n_step is None:
            raise ValueError('Please specify the number of data assimilation steps')

        if true_sol is None:
            raise ValueError(' Please specify the true solution to generate the observation')

        if R_diag:
            diag_R = np.diag(R)
            R_sqrt = np.sqrt(diag_R)
            R_sqrt = np.diag(R_sqrt)
            
        elif np.size(R,0) == np.size(R,1):
            print('shape R is %d by %d' %(np.size(R,0), np.size(R,1)))
            raise ValueError('R should be a square matrix')
        else:
            from scipy.linalg import sqrtm
            R_sqrt = sqrtm(R)

        n = self.n
        if n==0:
            n = self.m     # for the case that the number of observation is not given
        # generate measurements without noise
        y = self.forward_run(true_sol)
        hy = self.observation_model(y, self.H, self.lin)
        hy = hy.reshape(-1,1)
        obs = np.zeros((n, n_step))
        if noise:
            for i in range(n_step):
               obs[:,i] =  hy + np.multiply(R_sqrt, np.random.randn(n,1))
        else:
            obs = np.tile(hy, (1, n_step))

        return obs


    def CovarianceMatrix(self, grid=None, kernel='Gaussian', beta = None):
        ### Construct covariance matrix

        if grid:
            raise NotImplementedError
        xmin = self.params['xmin']
        dx = self.params['dx']
        length_x = self.params['lx']
        nx = self.params['nx']
        xr = np.linspace(xmin[0]+.5*dx[0],  xmin[0]+length_x[0]-.5*dx[0], nx[0])
        yr = np.linspace(xmin[1]+.5*dx[1],  xmin[1]+length_x[1]-.5*dx[1],nx[1])
        x, y = np.meshgrid(xr, yr)
        x1, x2 = np.meshgrid(x, x)
        y1, y2 = np.meshgrid(y,y)
        distance_x = (x1-x2)**2
        distance_y = (y1-y2)**2
        distance = distance_x/(L[0]**2)+distance_y/(L[1]**2)

        if beta == None:
            beta = 1
        if kernel == 'Gaussian':
            Q = beta*np.exp(-distance)
        elif kernel == 'Exponential':
            Q = beta*np.exp(-np.sqrt(distance))

        return Q

    def get_PH_Parallel(self, s, U, delta = None, precision=None):
        """
        Computing the product of the Jacobian matrix with the covariance matrix
        """

        s = np.array(s)
        s= s.reshape(-1,1)
        norm_s = np.linalg.norm(s)
        n_runs = np.size(U,1)
        if np.size(s,0) != np.size(U,0):
            raise ValueError('vector s and matrix U should have the same hight')

        y = self.forward_run(s)
        hy = self.observation_model(y, self.H, self.lin)

        if delta is None or math.isnan(delta) or delta == 0:
            deltas = np.zeros((nruns,1),'d') 
            for i in range(n_runs):
                mag = np.dot(s.T, U[:,i])
                abs_mag = np.dot(abs(s.T), abs(U[:,i]))

                sign_mag = 1 if mag>=0 else -1

                deltas[i] = sign_mag*sqrt(precision)*(max(abs(mag),abs_mag))/((np.linalg.norm(x[:,i]))**2)
                
                if deltas[i] ==0:
                    raise ValueError('Check the vector s to see if it is reasonable')

                U[:,i] =  s + deltas[i]*norm_s*U[:,i]
                U[U<.01] = 0.01
                
                if self.parallel:
                    simul_obs_parallel = self.parallel_forward_run(U)
                    hU = self.observation_model(simul_obs_parallel, self.H, self.lin)
                
                else:
                    hU = np.zeros_like(U)
                    
                    for i in range(n_runs):
                        simul_obs= self.forward_run(U[:,i])
                        hU[:,i] = self.observation_model(simul_obs, self.H, self.lin)

                for i in range(n_runs):
                    HU[:,i] = (hU[:,i] - hy)/(deltas[i]*norm_s)

        else:
            U = s + delta*norm_s*U
            U[U<.01] = 0.01
        ## transforming negative values of bathymetry to a small positive value close to zero
        

        if self.parallel:
            simul_obs_parallel = self.parallel_forward_run(U)
            hU = self.observation_model(simul_obs_parallel, self.H, self.lin)
        
        else:
            hU = np.zeros_like(U)
            
            for i in range(n_runs):
                simul_obs= self.forward_run(U[:,i])
                hU[:,i] = self.observation_model(simul_obs, self.H, self.lin)
        
        HU = (hU - hy)/(delta*norm_s)

        return HU
    def add_basis(self, bU):
        ### needs to be modified
        bU = bU/np.linalg.norm(bU)
        updatedU = [bU]
        a = np.zeros(size(self.U))
        a[:,0] = bU

        for i in range(self.U.shape[1]-1):
            print(i)
            tmp = self.U[:,i]
            tmp = tmp-np.sum(np.dot(tmp.T, updatedU[i])*updatedU[i] for i in range(len(updatedU)))
            updatedU.append(tmp/np.linalg.norm(tmp))
            a[:,i+1] = tmp

        print(" shape U is", len(updatedU[:-1]))
        #U = np.array(updatedU[:])
        
        self.U = a

    def compute_Tp(self, HU, R, C_p, R_diag = None):
        if R_diag:
            R = np.diag(R)
            inv_R = np.divide(1.,R)
            inv_R = np.diag(inv_R)
        else:
            inv_R = np.linalg.inv(R)
        inv_C = np.linalg.inv(C_p)
        RHU = inv_R.dot(HU)
        A = np.linalg.inv(inv_C+ HU.T.dot(RHU))
        T_p = inv_R - RHU.dot(A).dot(RHU.T)
        return T_p

    def compute_K(self, T_p, HU):
        UC = self.U.dot(self.C_p)
        HUT = HU.T.dot(T_p)
        K = UC.dot(HUT)
        return K

    def update_s(self, K, obs_t, prev_s):
        s = prev_s
        obs_x = self.forward_run(s)
        H = self.H
        lin = self.lin
        obs_x = self.observation_model(obs_x, H, lin)
        dev = obs_t - obs_x
        s_new = s + K.dot(dev)
        s_new[s_new<0.01] = 0.01

        self.result.append(s_new)
        return s_new

    def cov_update_mod(self, HU, T_p):
        HUT = HU.T.dot(T_p)
        HUC = HU.dot(self.C_p)
        self.C_p = self.C_p - self.C_p.dot(HUT.dot(HUC))
        self.uncert.append(np.diag(np.dot(np.dot(self.U,self.C_p), self.U.T)))
        

    def Run(self):

        start = time()
        if self.Q is None:
            self.Q = ConstructCovarianceMatrix(kernel = 'Gaussian', beta = 1)
        rank = self.rank
        method = 'SVD'
        U, C_p, C_q = self.compute_low_rank(rank, method)
        #savemat('U.mat',{'U':self.U, 'C_p':C_p, 'C_q':C_q })
        #data = loadmat('U.mat', squeeze_me=True)
        #self.U = data['U']  
        #self.C_p = data['C_p']
        #self.C_q = data['C_q']
        #self.add_basis(self.trend)
        s = self.prior
        for i in range(self.n_step):
        	obs_t = self.obs[:,i].reshape(self.obs.shape[0],1)
	        self.cov_update_pred(self.C_p, self.C_q)
        	HU = self.get_PH_Parallel(s, self.U, delta=0.001)
        	Tp = self.compute_Tp( HU, self.R, self.C_p)
        	K = self.compute_K( Tp, HU)
        	s=self.update_s(K, obs_t, s)
        	self.cov_update_mod(HU, Tp)

        return self.result, self.uncert
