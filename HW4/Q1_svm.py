import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
def generate_data(n_samples, r, sigma, label):
    """
    Generate data samples of the specified class.

    Parameters:
    - n_samples: Number of samples to generate
    - r: Radius, determines the center location of the data points
    - sigma: Standard deviation of Gaussian noise
    - label: Data class label, -1 or +1

    Returns:
    - X: 2D coordinates of the data points, shape (n_samples, 2)
    - y: Data labels, shape (n_samples,)
    """
    # Sample angle theta from a uniform distribution
    theta = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Calculate coordinates without noise
    x_no_noise = r * np.cos(theta)
    y_no_noise = r * np.sin(theta)
    
    # Add Gaussian noise
    x_noise = np.random.normal(0, sigma, n_samples)
    y_noise = np.random.normal(0, sigma, n_samples)
    
    # Obtain final coordinates
    x = x_no_noise + x_noise
    y = y_no_noise + y_noise
    
    # Combine into data matrix
    X = np.column_stack((x, y))
    
    # Create label vector
    y_labels = np.full(n_samples, label)
    
    return X, y_labels

# Parameter settings
r_minus = 2      # r_{-1}
r_plus = 4       # r_{+1}
sigma = 1        # Standard deviation of noise
n_samples_train = 1000  # Number of training samples per class
n_samples_test = 10000  # Number of test samples per class

# Generate training data
X_train_neg, y_train_neg = generate_data(n_samples_train, r_minus, sigma, -1)
X_train_pos, y_train_pos = generate_data(n_samples_train, r_plus, sigma, +1)

# Combine training data
X_train = np.vstack((X_train_neg, X_train_pos))
y_train = np.hstack((y_train_neg, y_train_pos))

# Generate test data
X_test_neg, y_test_neg = generate_data(n_samples_test, r_minus, sigma, -1)
X_test_pos, y_test_pos = generate_data(n_samples_test, r_plus, sigma, +1)

# Combine test data
X_test = np.vstack((X_test_neg, X_test_pos))
y_test = np.hstack((y_test_neg, y_test_pos))

# Shuffle training data
train_indices = np.arange(len(y_train))
np.random.shuffle(train_indices)
X_train = X_train[train_indices]
y_train = y_train[train_indices]

# Shuffle test data
test_indices = np.arange(len(y_test))
np.random.shuffle(test_indices)
X_test = X_test[test_indices]
y_test = y_test[test_indices]

# Plot training data
plt.figure(figsize=(8, 8))
plt.scatter(X_train[y_train==-1, 0], X_train[y_train==-1, 1], label='Class -1', alpha=0.5)
plt.scatter(X_train[y_train==+1, 0], X_train[y_train==+1, 1], label='Class +1', alpha=0.5)
plt.legend()
plt.title('Training Data Distribution')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.axis('equal')
plt.show()

# Define parameter grid
param_grid = {
    'C': np.logspace(-2, 2, 5),        # C parameter: 0.01, 0.1, 1, 10, 100
    'gamma': np.logspace(-2, 1, 4),    # gamma parameter: 0.01, 0.1, 1, 10
    'kernel': ['rbf']                  # Use RBF kernel
}

# Define SVM model
svc = SVC()

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Define grid search
grid_search = GridSearchCV(estimator=svc,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv,
                           n_jobs=-1,     # Use all available CPU cores
                           verbose=1)

# Run grid search
grid_search.fit(X_train, y_train)

# Output best parameters
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Get cross-validation results
cv_results = grid_search.cv_results_

# Convert results to DataFrame
scores_df = pd.DataFrame(cv_results)

# Extract needed columns
scores_df = scores_df[['param_C', 'param_gamma', 'mean_test_score']]

# Pivot table
scores_pivot = scores_df.pivot(index='param_gamma', columns='param_C', values='mean_test_score')

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(scores_pivot, annot=True, fmt=".4f", cmap='viridis')
plt.title('Validation Accuracy')
plt.xlabel('C')
plt.ylabel('gamma')
plt.show()

# Train final SVM model with best parameters
best_params = grid_search.best_params_

print("Training final SVM model with best parameters:")
print("C =", best_params['C'])
print("gamma =", best_params['gamma'])

best_svc = SVC(C=best_params['C'], gamma=best_params['gamma'], kernel='rbf')
best_svc.fit(X_train, y_train)

# Evaluate model performance on test set
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_test_pred = best_svc.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_error_rate = 1 - test_accuracy

print("Test set accuracy: {:.4f}".format(test_accuracy))
print("Test set error rate: {:.4f}".format(test_error_rate))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


conf_mat = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=[-1, 1], yticklabels=[-1, 1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix on Test Set')
plt.show()


# Assuming the best SVM model has already been trained

# Plot decision boundary
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict classification result of grid points
Z = best_svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 10))

# Plot decision regions
plt.contourf(xx, yy, Z, alpha=0.2, levels=np.linspace(Z.min(), Z.max(), 3), cmap='coolwarm')

# Plot decision boundary
contour = plt.contour(xx, yy, Z, levels=[0], colors='black', linewidths=3, linestyles='--')

# Plot test data points
plt.scatter(X_test[y_test==-1, 0], X_test[y_test==-1, 1], label='Class -1', alpha=0.7, edgecolors='k', s=10)
plt.scatter(X_test[y_test==+1, 0], X_test[y_test==+1, 1], label='Class +1', alpha=0.7, edgecolors='k', s=10)

plt.legend()
plt.title('SVM Decision Boundary and Test Data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.show()
 