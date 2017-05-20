from normal import *
from utils import *
from tqdm import tqdm
import math

def gibbs(Z, Y, beta, tau, gamma, alpha_tau, beta_tau, iters):
	"""
	Params
	Z: It is normalised observed variable
	Y: Output
	beta: vector of lenght 3 determing the linear mapping in the regression
	tau: precison for the gaussian noise
	alpha_tau: Gamma distribution parameters on precision
	beta_tau: Gamma distribution parameters on precision

	Returns a matrix containing the values of all the parameters values at each iterations.
	"""
	trace = np.zeros((iters,5))

	for i in tqdm(range(iters)):
		for j in range(len(beta)):
			beta[j] = sample_beta_ind(Z, Y, beta, tau, gamma, j)  # sample beta1, beta2, beta3 from the conditional joint
		tau = sample_tau(Z, Y, beta, alpha_tau, beta_tau) # sample tau from the conditional joint which will be a gamma distribution
		trace[i,:] = np.append(beta,tau)  

	return trace


if __name__ == "__main__":
	
	[beta, tau, phi] = init()		#initialize as per given in the stacks psuedo code
	gamma = 1e-5 
	alpha_tau = 1e-3
	beta_tau = 1e-3
	[X,Y] = get_data('data.txt')		#data file
	Z = normalize(X)
	Z = np.column_stack((np.ones(len(Z)),Z))
	iters = 11000
	burn_in = 1000

	trace = gibbs(Z, Y, beta, tau, gamma, alpha_tau, beta_tau, iters) # return the matrix containing the parameters values at each iteration
	trace_burnt = trace[burn_in:] 		                              # We will run 1000 as burn iterations.				
	b = np.zeros((len(beta), len(trace_burnt)))
	for i in range(1,len(beta)):
		print('beta_' + str(i))
		b[i] = trace_burnt[:,i]/np.std(X[:,i-1])
		print('Mean : ' + str(np.mean(b[i])))
		print('Std: ' + str(np.std(b[i])))
		print('-'*80)
	
	b[0] = trace_burnt[:,0] - b[1]*np.mean(X[:,0]) - b[2]*np.mean(X[:,1]) - b[3]*np.mean(X[:,2])  
	print('beta_0')
	print('Mean : ' + str(np.mean(b[0])))
	print('Std: ' + str(np.std(b[0])))
	print('-'*80)

	sigma = np.sqrt(np.divide(1.0, trace_burnt[:,-1]))
	print('sigma')
	print('Mean : ' + str(np.mean(sigma)))
	print('Std : ' + str(np.std(sigma)))
