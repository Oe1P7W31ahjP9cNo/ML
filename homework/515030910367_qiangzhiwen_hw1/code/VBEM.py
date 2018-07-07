import numpy as np
import matplotlib.pyplot as plt
from sklearn import mixture


def experimentVBEM(data):
	max_cluster_num = 7
	concentration_types = ['dirichlet_process', 'dirichlet_distribution']
	weight = [0.01,0.1,1,10,100,1000]
	for ctype in concentration_types:
		for w in weight:
			gmm = mixture.BayesianGaussianMixture(max_cluster_num, max_iter = 100000, weight_concentration_prior_type = ctype, weight_concentration_prior = w)
			gmm.fit(data)
			pred = np.array(gmm.predict(data))[:,np.newaxis]
			p = np.concatenate((data,pred),axis=1)
			plt.figure()
			plt.title('VBEM_model with weight = ' + str(w) + ', in ' + ctype + ' mode with max cluster number ' + str(max_cluster_num))
			plt.scatter(p[:,0],p[:,1],marker = '+', c = p[:,2])
			plt.savefig('w:'+str(w)+'t:'+ctype+'.png')

n_samples = 500
np.random.seed(0)
C = np.array([[0., -0.1], [1.7, .4]])
X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
		  0.7 * np.random.randn(n_samples, 2) + np.array([-6, 3]),
		  0.4 * np.random.randn(n_samples, 2) + np.array([6, -3]),
		  0.2 * np.random.randn(n_samples, 2) + np.array([0, -3])]
experimentVBEM(X)
plt.show()
