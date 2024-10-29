import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define class prior probabilities
P_L0 = 0.35
P_L1 = 0.65

# Total number of samples
N = 10000

# Number of samples for each class
N_L0 = int(N * P_L0)
N_L1 = N - N_L0

# Define mean and covariance for each class
mu_0 = np.array([-1, -1, -1, -1])
Sigma_0 = np.array([[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]])

mu_1 = np.array([1, 1, 1, 1])
Sigma_1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])

# Generate samples
X_L0 = np.random.multivariate_normal(mu_0, Sigma_0, N_L0)
X_L1 = np.random.multivariate_normal(mu_1, Sigma_1, N_L1)

# Combine samples and labels
X = np.vstack((X_L0, X_L1))
L = np.hstack((np.zeros(N_L0), np.ones(N_L1)))

# Save samples and labels
np.savez('samples_and_labels.npz', X=X, L=L, mu_0=mu_0, Sigma_0=Sigma_0, mu_1=mu_1, Sigma_1=Sigma_1, P_L0=P_L0, P_L1=P_L1)

print("save to: samples_and_labels.npz")
