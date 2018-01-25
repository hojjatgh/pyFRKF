import numpy as np
import scipy.io as sio
#from IPython.core.debugger import Tracer; debug_here = Tracer()
from pdb import set_trace


__all__ = ['CSKF']

class CSKF:

    def __init__(self, forward_model, observation_model, initial_dist, params, H, R, Q, P, nx, n_step, rank, obs = None):
        self.forward_model = forward_model
        self.observation_model = observation_model
        self.obs = obs
        self.initial_dist = initial_dist
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.nx = nx
        self.n_step = 1
        self.rank = rank
        self.result =[]
        self.result.append(initial_dist)
        self.params = params
        self.m = 110*83
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
				raise ValueError('obs should be n by n_step')

    def forward_run(self, s):
        par = False
        simul_obs = self.forward_model(s,par)
        obs_x = self.observation_model(simul_obs, self.H)

        return obs_x


    def parallel_forward_run(self, V):
        par =True
        simul_obs_parallel = self.forward_model(V,par,ncores = self.ncores)	

        return simul_obs_parallel

    def cov_update_pred(self):
        self.C_p = self.C_p+self.C_q

    def compute_low_rank(self):
        u,C_p,v= np.linalg.svd(self.P)
        self.U = u[:,:self.rank]
        self.C_p = np.diag(C_p[:self.rank])
        self.C_q = np.diag(C_p[:self.rank])
        print(self.C_p.shape)
            

    def get_PH_Parallel(self, delta=0.001):
        s = self.result[-1]
        s= s.reshape(len(s),1)
        y_0 = self.forward_run(s)
        hy = self.observation_model(y_0, self.H)
        norm_s = np.linalg.norm(s)
        V = s + delta*norm_s*self.U
        V[V<.01] = 0.01
        simul_obs_parallel = self.parallel_forward_run(V)
        hU = self.observation_model(simul_obs_parallel, self.H)
        HU = (hU - hy)/(delta*norm_s)

        return HU

    def compute_Tp(self, HU):
        inv_R = np.linalg.inv(self.R)
        inv_C = np.linalg.inv(self.C_p)
        RHU = inv_R.dot(HU)
        A = np.linalg.inv(inv_C+ HU.T.dot(RHU))
        T_p = inv_R - RHU.dot(A).dot(RHU.T)
        return T_p

    def compute_K(self, T_p, HU):
        UC = self.U.dot(self.C_p)
        HUT = HU.T.dot(T_p)
        K = UC.dot(HUT)
        return K

    def update_s(self, K, obs_t):
        s = self.result[-1]
        print(s.shape)
        obs_x = self.forward_run(s)
        dev = obs_t - obs_x
        s_new = s + K.dot(dev)
        print(s_new.shape)
        s_new[s_new<0.01] = 0.01

        self.result.append(s_new)

    def cov_update_mod(self, HU, T_p):
        HUT = HU.T.dot(T_p)
        HUC = HU.dot(self.C_p)
        self.C_p = self.C_p - self.C_p.dot(HUT.dot(HUC))
        self.uncert.append(np.diag(self.C_p))
        	

    def CSKF_main(self):

        self.compute_low_rank()
        for i in range(self.n_step):
        	obs_t = self.obs[:,i].reshape(self.obs.shape[0],1)
	        self.cov_update_pred()
        	HU = self.get_PH_Parallel( delta=0.001)
        	Tp = self.compute_Tp( HU)
        	K = self.compute_K( Tp, HU)
        	self.update_s(K, obs_t)
        	self.cov_update_mod(HU, Tp)

        return self.result, self.uncert
