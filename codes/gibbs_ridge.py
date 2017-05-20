from normal import *
from utils import *
from tqdm import tqdm
import math

def gibbs(Z, Y, beta, tau, gamma, phi, alpha_tau, beta_tau, iters):
	
	trace = np.zeros((iters,6))

	for i in tqdm(range(iters)):
		beta[0] = sample_beta_ind(Z, Y, beta, tau, gamma, 0)
		for j in range(1,len(beta)):
			beta[j] = sample_beta_ind(Z, Y, beta, tau, phi, j)
		tau = sample_tau(Z, Y, beta, alpha_tau, beta_tau)
		phi = sample_tau(Z, Y, beta, alpha_tau, beta_tau)
		trace[i,:] = np.append(np.append(beta,tau), phi)

	return trace


if __name__ == "__main__":
	
	[beta, tau, phi] = init()
	gamma = 1e-5
	alpha_tau = 1e-3
	beta_tau = 1e-3
	[X,Y] = get_data('data.txt')
	Z = normalize(X)
	Z = np.column_stack((np.ones(len(Z)),Z))
	iters = 10000
	burn_in = 1000

	trace = gibbs(Z, Y, beta, tau, gamma, phi, alpha_tau, beta_tau, iters)
	trace_burnt = trace[burn_in:]
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
