import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set random seed to ensure reproducibility of results
np.random.seed(0)

# Total number of samples
num_samples = 10000

# Define class prior probabilities
priors = [0.3, 0.3, 0.4]

# Calculate the number of samples for each class
num_samples_per_class = np.random.multinomial(num_samples, priors)

# Define mean vectors and covariance matrix
mu1 = np.array([0, 0, 0])
mu2 = np.array([2.5, 0, 0])
mu3 = np.array([0, 2.5, 0])
mu4 = np.array([2.5, 2.5, 0])
cov = np.eye(3)  # 3x3 identity matrix

# Initialize data and label lists
data = []
labels = []

# Generate samples for class 1
samples_class1 = np.random.multivariate_normal(mu1, cov, num_samples_per_class[0])
data.append(samples_class1)
labels.extend([1] * num_samples_per_class[0])

# Generate samples for class 2
samples_class2 = np.random.multivariate_normal(mu2, cov, num_samples_per_class[1])
data.append(samples_class2)
labels.extend([2] * num_samples_per_class[1])

# Generate samples for class 3 (equal-weight mixture of two Gaussian components)
samples_class3 = []
for _ in range(num_samples_per_class[2]):
    # Randomly select Gaussian component
    component = np.random.choice([3, 4])
    if component == 3:
        sample = np.random.multivariate_normal(mu3, cov)
    else:
        sample = np.random.multivariate_normal(mu4, cov)
    samples_class3.append(sample)

data.append(np.array(samples_class3))
labels.extend([3] * num_samples_per_class[2])

# Combine data and labels
data = np.vstack(data)
labels = np.array(labels)

# Shuffle the data order
indices = np.arange(num_samples)
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

# Combine data and labels into one array
dataset = np.column_stack((data, labels))

# Output data dimensions for verification
print("shape: ", dataset.shape)  # Should be (10000, 4), first three columns are data, last column is label

np.savez('dataset_q2.npz', data=data, labels=labels)

# 3D scatter plot to display data, using different shapes to distinguish labels
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot samples of class 1
class1_indices = labels == 1
ax.scatter(data[class1_indices, 0], data[class1_indices, 1], data[class1_indices, 2], marker='o', label='Class 1')

# Plot samples of class 2
class2_indices = labels == 2
ax.scatter(data[class2_indices, 0], data[class2_indices, 1], data[class2_indices, 2], marker='^', label='Class 2')

# Plot samples of class 3
class3_indices = labels == 3
ax.scatter(data[class3_indices, 0], data[class3_indices, 1], data[class3_indices, 2], marker='s', label='Class 3')

# Set legend and labels
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.legend()

plt.show()
