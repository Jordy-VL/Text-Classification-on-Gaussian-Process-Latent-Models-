from utils import *
from scipy.stats import t
from tqdm import tqdm
from scipy import stats
def joint_dist(Z, Y, state, gamma, alpha_tau, beta_tau):
	result = 0.0
	tau = state[4]
	phi = state[5]
	for i in range(len(Y)):
		result += t.logpdf(Y[i], 4, loc=Z[i].dot(state[:4]), scale = 1/np.sqrt(tau))
	
	for j in range(3):
		result += stats.norm.logpdf(state[j+1], scale=1/np.sqrt(phi))
	
	result += stats.norm.logpdf(state[0], scale=1/np.sqrt(gamma))
	result += stats.gamma.logpdf(state[4], alpha_tau, scale=1/beta_tau)	## tau
	result += stats.gamma.logpdf(state[5], alpha_tau, scale=1/beta_tau)	## tau
	return np.exp(result)
	
def metropolis_sampler(Z, Y, state, cov, gamma, alpha_tau, beta_tau, iters):

	trace = np.zeros((iters, len(state)))
	state_new = np.zeros(len(state))
	print(state)
	for i in tqdm(range(iters)):

		for j in range(len(state)):
			if(j<4):
				state_new[j] = np.random.normal(state[j], np.sqrt(cov[j][j]))
			else:
				while(True):
					state_new[j] = np.random.normal(state[j]*alpha_tau, np.sqrt(cov[j][j]))
					if(state_new[j]>0):
						break
				
			den = joint_dist(Z, Y, state, gamma, alpha_tau, beta_tau)
			aprob = min(1.0, joint_dist(Z, Y, state_new, gamma, alpha_tau, beta_tau)/den)
			u = np.random.uniform(0,1)
			if aprob > u:
				state[j] = np.copy(state_new[j])
		trace[i] = state

	return trace
	
if __name__ == "__main__":
	
	[X,Y] = get_data('data.txt')
	Z = normalize(X)
	Z = np.column_stack((np.ones(len(Z)),Z))
	[beta, tau, phi] = init()
	gamma = 1e-5
	alpha_tau = 1e-3
	beta_tau = 1e-3
	phi_mean = alpha_tau / beta_tau
	phi_var = alpha_tau / (beta_tau**2) 
	cov = np.eye(6)#np.diag([1/gamma, phi_mean, phi_mean, phi_mean, phi_var, phi_var])
	state = np.append(np.append(beta, tau),phi)
	iters = 10000
	burn_in = 1000
	trace = metropolis_sampler(Z, Y, state, cov, gamma, alpha_tau, beta_tau, iters)
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
