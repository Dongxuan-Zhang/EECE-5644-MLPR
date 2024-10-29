import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd

# -----------------------------
# 1. Generate dataset
# -----------------------------

# Set random seed to ensure reproducibility
np.random.seed(0)

# Define class prior probabilities
P_L0 = 0.35
P_L1 = 0.65

# Total number of samples
N = 10000

# Number of samples for each class
N_L0 = int(N * P_L0)
N_L1 = N - N_L0

# Define mean and covariance matrix for each class
mu_0 = np.array([0, 0, 0, 0])
mu_1 = np.array([1, 1, 1, 1])

# Randomly generate positive definite covariance matrices
A = np.random.randn(4,4)
Sigma_0 = np.dot(A, A.T)  # Ensure the covariance matrix is positive definite
B = np.random.randn(4,4)
Sigma_1 = np.dot(B, B.T)

# Generate samples
X_L0 = np.random.multivariate_normal(mu_0, Sigma_0, N_L0)
X_L1 = np.random.multivariate_normal(mu_1, Sigma_1, N_L1)

# Combine samples and labels
X = np.vstack((X_L0, X_L1))
L = np.hstack((np.zeros(N_L0), np.ones(N_L1)))

# -----------------------------
# 2. Minimum expected risk classifier for the correct model
# -----------------------------

# Calculate likelihood for all samples (correct model)
p_x_given_L0 = multivariate_normal.pdf(X, mean=mu_0, cov=Sigma_0)
p_x_given_L1 = multivariate_normal.pdf(X, mean=mu_1, cov=Sigma_1)

# Calculate likelihood ratio
likelihood_ratio = p_x_given_L1 / p_x_given_L0

# Define gamma values
gamma_values = np.logspace(-3, 3, num=1000)

# Lists to store results
TPR_list = []
FPR_list = []
FNR_list = []
P_error_list = []

for gamma in gamma_values:
    D = (likelihood_ratio > gamma).astype(int)
    
    TP = np.sum((D == 1) & (L == 1))
    FP = np.sum((D == 1) & (L == 0))
    FN = np.sum((D == 0) & (L == 1))
    TN = np.sum((D == 0) & (L == 0))
    
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    
    P_error = FPR * P_L0 + FNR * P_L1
    
    TPR_list.append(TPR)
    FPR_list.append(FPR)
    FNR_list.append(FNR)
    P_error_list.append(P_error)

# Convert to numpy arrays
gamma_values = np.array(gamma_values)
TPR_list = np.array(TPR_list)
FPR_list = np.array(FPR_list)
P_error_list = np.array(P_error_list)

# Find minimum error probability
min_error_index = np.argmin(P_error_list)
P_min_error = P_error_list[min_error_index]
gamma_min = gamma_values[min_error_index]
TPR_min = TPR_list[min_error_index]
FPR_min = FPR_list[min_error_index]

# Calculate theoretical optimal threshold
gamma_theoretical = P_L0 / P_L1

# Find the index of the theoretical optimal threshold
index_theoretical = np.argmin(np.abs(gamma_values - gamma_theoretical))
TPR_theoretical = TPR_list[index_theoretical]
FPR_theoretical = FPR_list[index_theoretical]

# -----------------------------
# 3. Implementation of Naive Bayes classifier
# -----------------------------

# Extract diagonal covariance matrices
Sigma_0_diag = np.diag(np.diag(Sigma_0))
Sigma_1_diag = np.diag(np.diag(Sigma_1))

# Calculate likelihood for all samples (Naive Bayes assumption)
p_x_given_L0_NB = multivariate_normal.pdf(X, mean=mu_0, cov=Sigma_0_diag)
p_x_given_L1_NB = multivariate_normal.pdf(X, mean=mu_1, cov=Sigma_1_diag)

# Calculate likelihood ratio
likelihood_ratio_NB = p_x_given_L1_NB / p_x_given_L0_NB

# Lists to store results
TPR_list_NB = []
FPR_list_NB = []
FNR_list_NB = []
P_error_list_NB = []

for gamma in gamma_values:
    D_NB = (likelihood_ratio_NB > gamma).astype(int)
    
    TP_NB = np.sum((D_NB == 1) & (L == 1))
    FP_NB = np.sum((D_NB == 1) & (L == 0))
    FN_NB = np.sum((D_NB == 0) & (L == 1))
    TN_NB = np.sum((D_NB == 0) & (L == 0))
    
    TPR_NB = TP_NB / (TP_NB + FN_NB)
    FPR_NB = FP_NB / (FP_NB + TN_NB)
    FNR_NB = FN_NB / (TP_NB + FN_NB)
    
    P_error_NB = FPR_NB * P_L0 + FNR_NB * P_L1
    
    TPR_list_NB.append(TPR_NB)
    FPR_list_NB.append(FPR_NB)
    FNR_list_NB.append(FNR_NB)
    P_error_list_NB.append(P_error_NB)

# Convert to numpy arrays
TPR_list_NB = np.array(TPR_list_NB)
FPR_list_NB = np.array(FPR_list_NB)
P_error_list_NB = np.array(P_error_list_NB)

# Find minimum error probability
min_error_index_NB = np.argmin(P_error_list_NB)
P_min_error_NB = P_error_list_NB[min_error_index_NB]
gamma_min_NB = gamma_values[min_error_index_NB]
TPR_min_NB = TPR_list_NB[min_error_index_NB]
FPR_min_NB = FPR_list_NB[min_error_index_NB]

# -----------------------------
# 4. Implementation of Fisher Linear Discriminant Analysis (LDA) classifier
# -----------------------------

# Get samples for each class separately
X_L0 = X[L == 0]
X_L1 = X[L == 1]

# Estimate class conditional means
mu_0_est = np.mean(X_L0, axis=0)
mu_1_est = np.mean(X_L1, axis=0)

# Estimate class conditional covariance matrices
Sigma_0_est = np.cov(X_L0, rowvar=False)
Sigma_1_est = np.cov(X_L1, rowvar=False)

# Number of samples in each class
n0 = X_L0.shape[0]
n1 = X_L1.shape[0]

# Within-class scatter matrix
S_W = (n0 - 1) * Sigma_0_est + (n1 - 1) * Sigma_1_est

# Between-class scatter matrix
mean_diff = (mu_0_est - mu_1_est).reshape(-1, 1)
S_B = np.dot(mean_diff, mean_diff.T)

# Calculate projection weight vector
S_W_inv = np.linalg.inv(S_W)
w_LDA = np.dot(S_W_inv, (mu_1_est - mu_0_est))

# Projection
y = np.dot(X, w_LDA)

# Define threshold range
tau_values = np.linspace(np.min(y), np.max(y), num=1000)

# Lists to store results
TPR_list_LDA = []
FPR_list_LDA = []
FNR_list_LDA = []
P_error_list_LDA = []

for tau in tau_values:
    D_LDA = (y > tau).astype(int)
    
    TP_LDA = np.sum((D_LDA == 1) & (L == 1))
    FP_LDA = np.sum((D_LDA == 1) & (L == 0))
    FN_LDA = np.sum((D_LDA == 0) & (L == 1))
    TN_LDA = np.sum((D_LDA == 0) & (L == 0))
    
    TPR_LDA = TP_LDA / (TP_LDA + FN_LDA)
    FPR_LDA = FP_LDA / (FP_LDA + TN_LDA)
    FNR_LDA = FN_LDA / (TP_LDA + FN_LDA)
    
    P_error_LDA = FPR_LDA * P_L0 + FNR_LDA * P_L1
    
    TPR_list_LDA.append(TPR_LDA)
    FPR_list_LDA.append(FPR_LDA)
    FNR_list_LDA.append(FNR_LDA)
    P_error_list_LDA.append(P_error_LDA)

# Convert to numpy arrays
P_error_list_LDA = np.array(P_error_list_LDA)
tau_values = np.array(tau_values)
TPR_list_LDA = np.array(TPR_list_LDA)
FPR_list_LDA = np.array(FPR_list_LDA)

# Find minimum error probability
min_error_index_LDA = np.argmin(P_error_list_LDA)
P_min_error_LDA = P_error_list_LDA[min_error_index_LDA]
tau_min_LDA = tau_values[min_error_index_LDA]
TPR_min_LDA = TPR_list_LDA[min_error_index_LDA]
FPR_min_LDA = FPR_list_LDA[min_error_index_LDA]

# -----------------------------
# 5. Plot and compare ROC curves for all classifiers
# -----------------------------

plt.figure(figsize=(10,8))
# Original model classifier
plt.plot(FPR_list, TPR_list, label='Original ROC')
# Naive Bayes classifier
plt.plot(FPR_list_NB, TPR_list_NB, label='NB ROC', linestyle='--')
# LDA classifier
plt.plot(FPR_list_LDA, TPR_list_LDA, label='LDA ROC', linestyle='-.')

# Mark minimum error probability points for each classifier
plt.scatter(FPR_min, TPR_min, color='blue', marker='o', label='Min(original)')
plt.scatter(FPR_min_NB, TPR_min_NB, color='red', marker='x', s=100, label='Min(NB)')
plt.scatter(FPR_min_LDA, TPR_min_LDA, color='purple', marker='^', s=100, label='Min(LDA)')
# Mark theoretical optimal threshold point
plt.scatter(FPR_theoretical, TPR_theoretical, color='green', marker='s', s=100, label='Theoratical')

plt.xlabel('FPR (P(D=1|L=0))')
plt.ylabel('TPR (P(D=1|L=1))')
plt.title('ROC')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# 6. Calculate and compare minimum error probability for each classifier
# -----------------------------

print("===== Original Model =====")
print(f"γ_min: {gamma_min}")
print(f"P_min_error: {P_min_error}")
print(f"TPR: {TPR_min}")
print(f"FPR: {FPR_min}")

print("\n===== Theoratical =====")
print(f"γ_theoretical: {gamma_theoretical}")
print(f"TPR: {TPR_theoretical}")
print(f"FPR: {FPR_theoretical}")

print("\n===== NB Classifier =====")
print(f"P_min_error_NB: {P_min_error_NB}")
print(f"γ_min_NB: {gamma_min_NB}")
print(f"TPR: {TPR_min_NB}")
print(f"FPR: {FPR_min_NB}")

print("\n===== LDA Classifier =====")
print(f"τ_min_LDA: {tau_min_LDA}")
print(f"P_min_error_LDA: {P_min_error_LDA}")
print(f"LDA TPR: {TPR_min_LDA}")
print(f"LDA FPR: {FPR_min_LDA}")

# -----------------------------
# 7. Performance analysis and result discussion
# -----------------------------

# Compare minimum error probabilities of each classifier
print("\n===== Min error probability =====")
print(f"Original: {P_min_error}")
print(f"NB: {P_min_error_NB}")
print(f"LDA: {P_min_error_LDA}")

# Plot error probability vs Gamma
plt.figure(figsize=(10, 6))
plt.plot(tau_values, P_error_list_LDA, label='Errors')
plt.scatter(tau_min_LDA, P_error_list_LDA[min_error_index_LDA], color='red', marker='o', label='Minimum Error')
plt.text(tau_min_LDA, P_error_list_LDA[min_error_index_LDA], f'({tau_min_LDA:.4f}, {P_error_list_LDA[min_error_index_LDA]:.4f})', fontsize=10, ha='right', color='red')
plt.xscale('symlog')
plt.xlabel('Gamma')
plt.ylabel('Proportion of Errors')
plt.title('LDA Probability of Error vs. Gamma')
plt.legend()
plt.grid(True)
plt.show()