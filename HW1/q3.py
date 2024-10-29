import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
wine_data = pd.read_csv("./winequality-white.csv", sep=';')
X_wine = wine_data.iloc[:, :-1]
y_wine = wine_data.iloc[:, -1]

X_har_train = pd.read_csv("./UCI HAR Dataset/train/X_train.txt", sep='\s+', header=None)
y_har_train = pd.read_csv("./UCI HAR Dataset/train/y_train.txt", header=None).squeeze()
X_har_test = pd.read_csv("./UCI HAR Dataset/test/X_test.txt", sep='\s+', header=None)
y_har_test = pd.read_csv("./UCI HAR Dataset/test/y_test.txt", header=None).squeeze()

# Split wine dataset into training and testing sets
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Preprocessing, Normalization, and Variance Threshold
def preprocess_data(X, threshold=0.01):
    # Standardize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Apply Variance Threshold
    selector = VarianceThreshold(threshold=threshold)  # 设置方差阈值，例如0.01
    X_reduced = selector.fit_transform(X_normalized)
    
    return X_normalized, X_reduced

# Apply preprocessing to the datasets
X_wine_train_normalized, X_wine_train_reduced = preprocess_data(X_wine_train)
X_wine_test_normalized, X_wine_test_reduced = preprocess_data(X_wine_test)
X_har_train_normalized, X_har_train_reduced = preprocess_data(X_har_train)
X_har_test_normalized, X_har_test_reduced = preprocess_data(X_har_test)


# Function to estimate class priors and mean/covariance of each class
def estimate_parameters(X, y):
    classes = np.unique(y)
    priors = []
    means = []
    covariances = []
    for c in classes:
        X_c = X[y == c]
        priors.append(len(X_c) / len(X))
        means.append(np.mean(X_c, axis=0))
        cov = np.cov(X_c, rowvar=False)
        lambda_reg = np.trace(cov) / cov.shape[0]
        cov += lambda_reg * np.eye(cov.shape[0])  # Add regularization term to ensure positive definiteness
        covariances.append(cov)
    return priors, means, covariances
# Apply models
wine_priors, wine_means, wine_covs = estimate_parameters(X_wine_train, y_wine_train)
human_priors, human_means, human_covs = estimate_parameters(X_har_train_reduced, y_har_train)

# Return the predicted class labels instead of the indices
def minimum_probability_error(priors, means, covariances, X, classes):
    num_classes = len(priors)
    num_samples = X.shape[0]
    log_probs = np.zeros((num_samples, num_classes))
    for i in range(num_classes):
        mean = means[i]
        cov = covariances[i]
        inv_cov = np.linalg.inv(cov)
        log_det_cov = np.log(np.linalg.det(cov))
        diff = X - mean
        log_prob = -0.5 * np.sum(diff @ inv_cov * diff, axis=1) - 0.5 * log_det_cov + np.log(priors[i])
        log_probs[:, i] = log_prob
    return classes[np.argmin(log_probs, axis=1)]


# Calculate POE for wine and human activity datasets
wine_poe_predictions = minimum_probability_error(wine_priors, wine_means, wine_covs, X_wine_test, np.unique(y_wine_train))
human_poe_predictions = minimum_probability_error(human_priors, human_means, human_covs, X_har_test_reduced, np.unique(y_har_train))


wine_poe_accuracy = accuracy_score(y_wine_test, wine_poe_predictions)
human_poe_accuracy = accuracy_score(y_har_test, human_poe_predictions)

print("Wine POE Accuracy:", wine_poe_accuracy)
print("Human Activity POE Accuracy:", human_poe_accuracy)

# Display confusion matrices and PCA visualizations in separate figures
# Wine dataset visualizations
fig_wine, axes_wine = plt.subplots(1, 2, figsize=(20, 8))

# Wine PCA Visualization
pca_wine = PCA(n_components=2)
wine_pca = pca_wine.fit_transform(X_wine_train)
scatter = axes_wine[0].scatter(wine_pca[:, 0], wine_pca[:, 1], c=y_wine_train, cmap='viridis', s=10)
legend1 = axes_wine[0].legend(*scatter.legend_elements(), title="Labels")
axes_wine[0].add_artist(legend1)
axes_wine[0].set_xlabel('Principal Component 1')
axes_wine[0].set_ylabel('Principal Component 2')
axes_wine[0].set_title('2D PCA Visualization of Wine Data')

# Wine Confusion Matrix Heatmap
wine_conf_matrix = pd.crosstab(y_wine_test, wine_poe_predictions)  # Dummy confusion matrix for visualization
sns.heatmap(wine_conf_matrix, annot=True, cmap='Blues', fmt='g', ax=axes_wine[1])
axes_wine[1].set_xlabel('Predicted Label')
axes_wine[1].set_ylabel('True Label')
axes_wine[1].set_title('Confusion Matrix Heatmap of Wine Data')

plt.tight_layout()
plt.show()

# Human Activity dataset visualizations
fig_human, axes_human = plt.subplots(1, 2, figsize=(20, 8))

# Human Activity PCA Visualization
pca_human = PCA(n_components=2)
human_pca = pca_human.fit_transform(X_har_train)
scatter = axes_human[0].scatter(human_pca[:, 0], human_pca[:, 1], c=y_har_train, cmap='viridis', s=10)
legend1 = axes_human[0].legend(*scatter.legend_elements(), title="Labels")
axes_human[0].add_artist(legend1)
axes_human[0].set_xlabel('Principal Component 1')
axes_human[0].set_ylabel('Principal Component 2')
axes_human[0].set_title('2D PCA Visualization of Human Activity Data')


# Human Activity Confusion Matrix Heatmap
human_conf_matrix = pd.crosstab(y_har_test, human_poe_predictions)  # Dummy confusion matrix for visualization
sns.heatmap(human_conf_matrix, annot=True, cmap='Blues', fmt='g', ax=axes_human[1])
axes_human[1].set_xlabel('Predicted Label')
axes_human[1].set_ylabel('True Label')
axes_human[1].set_title('Confusion Matrix Heatmap of Human Activity Data')

plt.tight_layout()
plt.show()

# Output number of labels for each dataset
print("Number of labels in Wine Dataset:", np.unique(y_wine))
print("Number of labels in Human Activity Dataset:", np.unique(y_har_train))