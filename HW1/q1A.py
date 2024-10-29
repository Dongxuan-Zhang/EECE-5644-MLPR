import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve, auc
from scipy.optimize import minimize
import pandas as pd

# Part 1: Data Generation

def generate_data(mu, sigma, alpha, n_samples):
    """
    Generate Gaussian Mixture Model Data
    """
    n_components = len(alpha)
    data = []
    labels = []
    for i in range(n_samples):
        component = np.random.choice(n_components, p=alpha)
        sample = np.random.multivariate_normal(mu[component], sigma[component])
        data.append(sample)
        labels.append(component)
    return np.array(data).T, np.array(labels)

dimension = 2
p = [0.6, 0.4]  # class priors

# Class 0 parameters
mu0 = np.array([[-0.9, -1.1], [0.8, 0.75]])
Sigma0 = np.array([
    [[0.75, 0], [0, 1.25]],
    [[0.75, 0], [0, 1.25]]
])
alpha0 = [0.5, 0.5]

# Class 1 parameters
mu1 = np.array([[-1.1, 0.9], [0.9, -0.75]])
Sigma1 = np.array([
    [[0.75, 0], [0, 1.25]],
    [[0.75, 0], [0, 1.25]]
])
alpha1 = [0.5, 0.5]

# Generate data for different training sizes
train_sizes = [20, 200, 2000, 10000]
train_data = []
train_labels = []

for n_samples in train_sizes:
    x0, labels0 = generate_data(mu0, Sigma0, alpha0, int(n_samples * p[0]))
    x1, labels1 = generate_data(mu1, Sigma1, alpha1, int(n_samples * p[1]))
    data = np.hstack((x0, x1))
    labels = np.hstack((labels0, labels1 + len(alpha0)))
    train_data.append(data)
    train_labels.append(labels)

# Plot training data for different sample sizes in a single figure with subplots
plt.figure(figsize=(16, 12))
for i, n_samples in enumerate(train_sizes):
    ax = plt.subplot(2, 2, i + 1)
    ax.scatter(train_data[i][0, train_labels[i] == 0], train_data[i][1, train_labels[i] == 0], c='b', label='Class 0', alpha=0.5)
    ax.scatter(train_data[i][0, train_labels[i] == 1], train_data[i][1, train_labels[i] == 1], c='r', label='Class 1', alpha=0.5)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'{n_samples} Samples From Two Classes')
    if i == 3:
        ax.legend()
plt.suptitle('Training and Validating Data Plot')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Part 2: Bayesian Classification

n_samples = 10000
x0, labels0 = generate_data(mu0, Sigma0, alpha0, int(n_samples * p[0]))
x1, labels1 = generate_data(mu1, Sigma1, alpha1, int(n_samples * p[1]))
data = np.hstack((x0, x1))
labels = np.hstack((labels0, labels1 + len(alpha0)))
labels = (labels > 0).astype(int)  # Convert labels to binary format

def bayesian_classifier(x, mu0, Sigma0, mu1, Sigma1, p0, p1):
    """
    Bayesian classifier for two classes
    """
    likelihood0 = multivariate_normal.pdf(x, mean=mu0, cov=Sigma0)
    likelihood1 = multivariate_normal.pdf(x, mean=mu1, cov=Sigma1)
    posterior0 = likelihood0 * p0
    posterior1 = likelihood1 * p1
    return 0 if posterior0 > posterior1 else 1

# Classify all data points
predictions = []
probabilities = []
for i in range(data.shape[1]):
    likelihood0 = multivariate_normal.pdf(data[:, i], mean=mu0[0], cov=Sigma0[0])
    likelihood1 = multivariate_normal.pdf(data[:, i], mean=mu1[0], cov=Sigma1[0])
    posterior0 = likelihood0 * p[0]
    posterior1 = likelihood1 * p[1]
    probabilities.append(posterior1 / (posterior0 + posterior1))
    predictions.append(0 if posterior0 > posterior1 else 1)

predictions = np.array(predictions)
probabilities = np.array(probabilities)

# Plot classification results with true labels
plt.figure(figsize=(10, 6))
plt.scatter(data[0, (labels == 0) & (predictions == 0)], data[1, (labels == 0) & (predictions == 0)], c='g', label='Correct decisions for data from Class 0', alpha=0.5)
plt.scatter(data[0, (labels == 0) & (predictions == 1)], data[1, (labels == 0) & (predictions == 1)], c='r', marker='o', label='Wrong decisions for data from Class 0', alpha=0.5)
plt.scatter(data[0, (labels == 1) & (predictions == 1)], data[1, (labels == 1) & (predictions == 1)], c='g', marker='+', label='Correct decisions for data from Class 1', alpha=0.5)
plt.scatter(data[0, (labels == 1) & (predictions == 0)], data[1, (labels == 1) & (predictions == 0)], c='r', marker='+', label='Wrong decisions for data from Class 1', alpha=0.5)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Data and their Classifier Decisions Versus True Labels')
plt.legend()
plt.show()

# Part 3: ROC Curve and Performance Evaluation
fpr, tpr, thresholds = roc_curve(labels, probabilities)
roc_auc = auc(fpr, tpr)

# Find the threshold corresponding to the minimum probability of error (min-P(error))
min_p_error_index = np.argmin(np.abs(tpr - (1 - fpr)))
min_p_error_threshold = thresholds[min_p_error_index]

# Plot ROC Curve with min-P(error) point marked
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter(fpr[min_p_error_index], tpr[min_p_error_index], color='red', label='Min-P(error) Point', zorder=5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

print(f'Area Under the Curve (AUC): {roc_auc:.2f}')
print(f'Minimum P(error) Threshold: {min_p_error_threshold:.2f}')

# Part 4: Create Theoretical and Estimated Minimum Errors Table
data = {
    'γ': ['1.50', f'{min_p_error_threshold:.2f}'],
    'Min Probability Error': ['25.24%', f'{(1 - roc_auc) * 100:.2f}%']
}

error_table = pd.DataFrame(data, index=['Theoretical', 'Estimated from Data'])
print("\nTable 1. Theoretical and Estimated Minimum Errors with Known PDFs")
print(error_table)
