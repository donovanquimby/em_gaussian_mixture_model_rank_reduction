# -*- coding: utf-8 -*-
"""
# =============================================================================
Created on Tue Jun  2 16:40:32 2020
# =============================================================================
"""

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
    Stuff Tau tilda, pdf, log-likelyhood

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
    pdf = (1/(2*(np.pi**(m/2))))*D*np.exp(-(1/2)*m_k)
    weighted_prob = p * pdf
    
    # calculate Tau_tilda
    Tau_tilda = weighted_prob
     
    # use this for high dimesional rank reduction
    one = -0.5*m_k
    two = -0.5*np.log(np.prod(Lambda))
    three = -0.5*m*(np.log(2)+np.log(np.pi))
    log_prob_density = one + two + three
    
    return(Tau_tilda,log_prob_density)
    
    
    
    