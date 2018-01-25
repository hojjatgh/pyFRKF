import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import stwave as st
import math
from cskf import CSKF
#import linear_dist

def generate_Q(xmin, dx,length_x, nx, L, kernel):

	xr = np.linspace(xmin[0]+.5*dx[0],  xmin[0]+length_x[0]-.5*dx[0], nx[0])
	yr = np.linspace(xmin[1]+.5*dx[1],  xmin[1]+length_x[1]-.5*dx[1],nx[1])
	x, y = np.meshgrid(xr, yr)
	x1, x2 = np.meshgrid(x, x)
	y1, y2 = np.meshgrid(y,y)
	distance_x = (x1-x2)**2
	distance_y = (y1-y2)**2
	distance = distance_x/(L[0]**2)+distance_y/(L[1]**2)
	if kernel == 'Gaussian':
		Q = np.exp(-distance)
	elif kernel == 'Exponential':
		Q = np.exp(-np.sqrt(distance))
	return Q

def initializer_dist(bc_l, bc_r, nx, distribution):
	if distribution == 'linear':
		linear_variation = np.linspace(bc_l, bc_r, nx[0]).reshape(1,nx[0])
		Initial_dist = np.ones((nx[1],nx[0]))*linear_variation
		Initial_dist = Initial_dist.reshape(nx[0]*nx[1],1)

	return Initial_dist
	
def forward_model(s, parallel, ncores=None):
	model = st.Model()
	if parallel:
		simul_obs = model.run(s, parallel, ncores)
	else:
		simul_obs = model.run(s, parallel)
	return simul_obs

def observation_model(x, H):
	return np.dot(H, x)


xmin = np.array([0,0])
nx =  np.array([110, 83])
dx = np.array([5., 5.])
length_x = nx*dx
data = loadmat('true_depth.mat', squeeze_me=True)
ref_bathy = data['true']
ref_bathy[ref_bathy<0.01] = 0.01
ref_bathy = ref_bathy.reshape(np.prod(nx),1)
theta = np.array([1, 4])
kernel =  'Gaussian'
Q = theta[0]*generate_Q(xmin, dx,length_x, nx, length_x/10, kernel) #generating Q matrix
P = theta[1]* Q
bc_l = 7
bc_r = 0
distribution = "linear"
initial_dist = initializer_dist(bc_l, bc_r, nx, distribution)

plt.figure()
plt.imshow(initial_dist.reshape(nx[1],nx[0]))
plt.xlabel('Offshore distance')
plt.ylabel('Cross-shore distance')
plt.title('prior bathymetry')
plt.colorbar()
plt.savefig('initial.png')

obs_x = np.arange(0, nx[0])
obs_y = np.arange(0, nx[1])
theta_r = .01
R = theta_r*np.eye(len(obs_x)*len(obs_y))
H = np.eye(len(obs_x)*len(obs_y))
np.random.seed(100)
simul_obs = forward_model(ref_bathy, parallel=False)
obs = observation_model(ref_bathy, H)
obs = obs.reshape(len(obs),1)
n_step = 5
obs = obs.dot(np.ones((1, n_step)))
print(ref_bathy.shape, simul_obs.shape, initial_dist.shape, np.max(obs))

plt.figure()
plt.imshow(np.flip(obs[:,0].reshape(nx[1],nx[0]),1))
plt.xlabel('Offshore distance')
plt.ylabel('Cross-shore distance')
plt.title('Observation')
plt.colorbar()
plt.savefig('Observation.png')
params = {}
params['parallel'] = True
rank = 96
prob = CSKF(forward_model, observation_model, initial_dist, params, H, R, Q, P, nx, n_step, 96, obs)
results, uncert = prob.CSKF_main()
print("length result is and each value has %d elements" %len(uncert[0]))
print(results[1].shape)
print(uncert[0].shape)
print(results[1])
plt.figure()
plt.imshow(np.flip(results[1].reshape(nx[1],nx[0]),1))
plt.xlabel('Offshore distance')
plt.ylabel('Cross-shore distance')
plt.title('Observation')
plt.colorbar()
plt.savefig('step1.png')

plt.figure()
plt.imshow(np.flip(results[0].reshape(nx[1],nx[0]),1))
plt.xlabel('Offshore distance')
plt.ylabel('Cross-shore distance')
plt.title('Observation')
plt.colorbar()
plt.savefig('Prior.png')
savemat('results1.mat',{'results':results, 'uncert': uncert})