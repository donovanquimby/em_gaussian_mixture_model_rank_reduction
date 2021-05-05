# -*- coding: utf-8 -*-
"""
# =============================================================================
Created on Wed Jun  3 08:48:43 2020
# =============================================================================
"""

import numpy as np
from os.path import abspath, exists
from scipy.io import loadmat
from scipy.linalg import eig, svd
import matplotlib.pyplot as plt
import lowRank

# %%

def import_data(x):
    # read the graph from 'play_graph.txt'
    f_path = abspath(x)
    if exists(f_path):
         file = loadmat(x)
           
    return file

data = import_data("data/data.mat")['data'].T
labels = import_data('data/label.mat')['trueLabel']

X = data
n_observations, n_features = data.shape

# %% model running parameters
K = 2 # number of clusters, gaussien normal distributions
r = 100 # number of reduced features

# %% functions
def lowRankCalc( cov, y, p,  U_tilda, Mu_tilda, Lambda ):
    """
    Parameters
    ----------
    
    cov : numpy matrix. Covariance matrix
    y : numpy vector. Vector of data point
    pi : np vector. Vector of weighting factors
    Mu_tilda : np array. projected avg vector for cluster
    Lambda : np array. Eigan values

    Returns
    -------
    Tau, log-likelyhood

    """
    import numpy as np
    m = len(y)
    
    # calculate x_tilda and mu_tilda     
    X_tilda = np.dot(U_tilda.T,y)
    
    # calculate m_k and D for sigma calc slide 7 gmm speedup
    diff =  X_tilda-Mu_tilda.T
    
    m_k = np.sum(np.power((diff),2)/Lambda)
    D = np.prod(1/np.sqrt(Lambda))
    
    # compute tau_tilda for k slide 7 gmm speed-up
    Tau_tilda = p*D*np.exp(-.5*m_k)
     
    # log-likelihood Calc (alternate derivation)
    one = -0.5*m_k
    two = -0.5*(np.sum(np.log(Lambda)))
    three = -0.5*m*(np.log(2)+np.log(np.pi))
    log_prob_density = one + two + three
    
    return(Tau_tilda,log_prob_density)


def initialize(x=X, k = K):
        
    mu1 = np.random.normal(0.0,0.5,(n_features,1))
    mu2 = np.random.normal(0.0,0.5,(n_features,1))
    mu_mat = np.hstack((mu1,mu2))
    mu = mu_mat   

    cov1 = np.identity(n_features)
    cov = [cov1]*k
 
    pi = np.repeat(1/k,k)
    return(mu,cov,pi)

def e_step(mu,cov,pi, k=2, r =100):
    
    tau_tilda = np.zeros((n_observations,len(pi)))
    log_pdf = np.zeros((n_observations,len(pi)))
    
    
    for k_i in range(k):
        evecs, evals, Vh = svd(cov[k_i])
        Lambda =  evals.real[0:r]
        U_tilda =  evecs.real[:,0:r]
        Mu_tilda = np.dot(U_tilda.T,mu[:,k_i])
        
        for i in range(len(data)):
            t,l = lowRank.lowRankCalc( cov = cov[k_i], y = X[i,:], p = pi[k_i], U_tilda =U_tilda, Mu_tilda = Mu_tilda, Lambda=Lambda)
            tau_tilda[i,k_i]=t
            log_pdf[i,k_i] = t
       
    
    
    # normalization as per slide 7 gmm speed-up        
    C = np.sum(tau_tilda,axis=1)  
    tau = tau_tilda/C[:,None]
    log_likleyhood = (np.sum(np.log(np.sum(tau_tilda, axis = 1))))
    return(tau, tau_tilda,log_likleyhood)
   

def m_step(tau, k):
    mu = []
    cov = []
    pi = []
     
    # calculate new pi values (should sum to one)
    tr = np.sum(tau, axis= 0)   
    pi = tr/n_observations      
    
    # calculate mu
    for k_i in range(k):
        mu.append((1/tr[k_i])*np.sum(tau[:,k_i]*X.T,axis=1))
    
    # calculate weighted covariance of assigned data
    for k_i in range(k):
        diff =  X - mu[k_i].T
        cov.append((1/tr[k_i])*np.dot(tau[:,k_i]*diff.T,diff))
        
    mu = np.asarray(mu).T
    
    return(mu,cov,pi)
# %% Run model

# initialize problem
mu, cov, pi = initialize()

log_likelyhood = []
convergence = 100

iter_n=0
while convergence >= 0.000025 and iter_n < 50:
    
    tau, tau_tilda, log_like_temp =  e_step(mu,cov,pi, k = K, r = r)
    log_likelyhood.append(log_like_temp)
    mu, cov, pi = m_step(tau, k = K)
    
    if iter_n == 0:
        convergence = 100
    else:
        convergence = abs((log_likelyhood[iter_n] - log_likelyhood[iter_n-1]))/abs(log_likelyhood[iter_n-1])

    iter_n += 1
    print(f"Iteration Number: {iter_n}")
    print(f"Change In Log-likelihood: {convergence}")

# %% False classification rate 

idx_six_start = 1032
n_twos = 1032
n_sixs =1990-n_twos

classification = tau[:,0]/tau[:,1]
classification[classification >= 1] = 2
classification[classification < 1] = 6
labels = labels.reshape(n_observations,)    

n_miss = np.count_nonzero(labels-classification)
rate_overall_miss = min(n_miss/n_observations, 1-(n_miss/n_observations))

# two miss-classification rate
n_miss_two = (classification[0:idx_six_start] != 2).sum()
rate_two_miss = min(n_miss_two/n_twos,1-(n_miss_two/n_twos))

# six miss-classification rate
n_miss_six = (classification[idx_six_start:] != 6).sum()
rate_six_miss = min(n_miss_six/n_sixs, 1-(n_miss_six/n_sixs))



# %% Plot log-likleyhood
    
import seaborn as sns
fig = plt.figure()
fig.set_size_inches(10, 8)
ax = sns.lineplot(x=range(len(log_likelyhood)-1), y=log_likelyhood[1:])
ax.axes.set_title("Log-likelihood Convergence GMM Model",fontsize=25)
ax.set_xlabel("Iteration Number (-)",fontsize=20)
ax.set_ylabel("Log-Likelihood (-)",fontsize=20)  
#plt.savefig("pics/log_likelihood.png")


# %% display avg vector images


def display_avg_image(data):
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    list2 = ['Average Cluster 1 Vector','Average Cluster 2 Vector']
    j =0
    for i in range(2):
        img = np.reshape(mu[:,i], (28,28), order = 'F')
        
        ax[j].imshow(img, cmap='gray_r')
        ax[j].set_title(list2[j])
        j += 1
    fig.suptitle("Final Average Vectors GMM Model",fontsize=25)
    #plt.savefig("pics/final_average_vectors.png")   
    
display_avg_image(mu)

# %% display example images from datas

two_idx = 1
six_idx = data.shape[0]-1


def display_image(data,two_idx,six_idx):
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    list1 = [two_idx,six_idx]
    list2 = ['Two','Six']
    j =0
    for i in list1:
        img = np.reshape(data[i], (28,28), order = 'F') 
        ax[j].imshow(img, cmap='gray_r')
        ax[j].set_title(list2[j])
        j += 1
        
    fig.suptitle("Two Example Digits From Data",fontsize=25)
   # plt.savefig('pics/two_Example_digits.png')
display_image(data, two_idx, six_idx)

