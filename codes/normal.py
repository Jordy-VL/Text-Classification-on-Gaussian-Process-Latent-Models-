import numpy as np

def sample_beta0(Z, Y, beta, tau, gamma):

	N = len(Y)
	precision = gamma + N*tau
	mean = tau * np.sum(Y - (Z.dot(beta) - beta[0]))
	mean /= precision
	return np.random.normal(mean, 1/ np.sqrt(precision))

def sample_beta_ind(Z, Y, beta, tau, gamma, ind):
	N = len(Y)
	precision = tau*np.sum(Z[:,ind]**2) + gamma
	mean = tau * np.sum(Z[:,ind]*(Y - (Z.dot(beta) - Z[:,ind]*beta[ind])))
	mean /= precision
	return np.random.normal(mean, 1 / np.sqrt(precision))

def sample_tau(Z, Y, beta, alpha_tau, beta_tau):
	N = len(Y)
	alpha_new = alpha_tau + N / 2.0
	resid = Y - Z.dot(beta)
	beta_new = beta_tau + np.sum(resid**2)/2
	return np.random.gamma(alpha_new, 1/ beta_new)

def sample_phi(Z, Y, beta, alpha_tau, beta_tau):
	alpha_new  = alpha_tau + 1.5
	beta_new = beta_tau + np.sum(beta[1:]**2)/2
	return np.random.gamma(alpha_new, 1 / beta_new)
	return
