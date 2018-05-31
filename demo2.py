import numpy as np
from scipy.stats import norm
from time import sleep

np.random.seed(1) # For reproducibility

"""
   BEGIN INPUT
"""

p = 1e-5             # Failure Probability
n = int(1e6)         # Number of samples
delta = 0.05         # Confidence parameter
replicas = int(1e4)  # Experiment replicas

"""
   END INPUT
"""

z = norm.ppf(1-delta/2)  # Z-score based on confidence level (1-delta)

def experiment(p, n, z):
    """ Test validity of CLT based confidence inervals for the mean of a Bernoulli random variable.
    
    Args:
        p: Success probability.
        n: Number of samples.
        z: Z-score for probability tails.
    
    Returns:
        True if confidence bound covers p, Fals otherwise.
    
    """
    
    # Sample mean and variance
    bmu = np.random.binomial(n, p)*1.0/n
    bvar = bmu * (1-bmu) * n/(n-1)
    
    # Cofidence Interval
    ci = [bmu - z*np.sqrt(bvar/n), bmu + z*np.sqrt(bvar/n)]
    
    # Test coverage
    return p >= ci[0] and p <= ci[1]

# Run Experiment 10 times more

confidence = []
print('Confidence (1-delta) in practice:\n')
for trial in range(10):
    
    # Replicas
    confidence.append(np.mean([experiment(p, n, z) for i in range(replicas)]))
    print('Trial %d: %.3f' % (trial+1, confidence[-1]) + (' Overconfident!' if confidence[-1]<(1-delta) else ''))
    
    sleep(0.5)

    
print('\nAgregating all trials: %.3f ' % (np.mean(confidence)))