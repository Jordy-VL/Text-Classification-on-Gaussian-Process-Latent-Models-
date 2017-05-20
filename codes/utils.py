import numpy as np

def init():
	beta = np.array([10.0, 0, 0, 0])
	tau = 0.1
	phi = 0.1
	return [beta,tau,phi]

def get_data(filepath):

	f = open(filepath).readlines()
	Y = np.array([float(y) for y in f[0].split(',')])
	X = np.array([[float(x) for x in line.split(',')] for line in f[1:]])
	
	return [X,Y]

def normalize(X):
	Z = np.zeros_like(X)
	for i in range(len(X[0])):
		Z[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
	
	return Z
