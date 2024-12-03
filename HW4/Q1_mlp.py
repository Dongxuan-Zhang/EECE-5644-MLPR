import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# ==============================
# 1. Data Generation
# ==============================

def generate_data(n_samples, r, sigma, label):
    """
    Generate data samples for a specified class.

    Parameters:
    - n_samples: Number of samples to generate
    - r: Radius, determines the center position of the data points
    - sigma: Standard deviation of Gaussian noise
    - label: Class label, either -1 or +1

    Returns:
    - X: 2D coordinates of the data points, shape (n_samples, 2)
    - y: Data labels, shape (n_samples,)
    """
    # Sample angle theta from a uniform distribution
    theta = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Calculate noise-free coordinates
    x_no_noise = r * np.cos(theta)
    y_no_noise = r * np.sin(theta)
    
    # Add Gaussian noise
    x_noise = np.random.normal(0, sigma, n_samples)
    y_noise = np.random.normal(0, sigma, n_samples)
    
    # Get the final coordinates
    x = x_no_noise + x_noise
    y = y_no_noise + y_noise
    
    # Combine into data matrix
    X = np.column_stack((x, y))
    
    # Create label vector
    y_labels = np.full(n_samples, label)
    
    return X, y_labels

# Parameters
r_minus = 2      # r_{-1}
r_plus = 4       # r_{+1}
sigma = 1        # Noise standard deviation
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

# Output data shapes
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# Visualize training data
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

# ==============================
# 2. Hyperparameter Tuning for MLP Model
# ==============================

# Define parameter grid
param_grid = {
    'hidden_layer_sizes': [(n,) for n in [5, 10, 20, 50, 100]],  # Single hidden layer with different numbers of neurons
    'activation': ['relu', 'tanh', 'logistic'],  # Activation functions
    'learning_rate_init': [0.001, 0.01, 0.1],  # Learning rate
}

# Define MLP model
mlp = MLPClassifier(max_iter=1000, random_state=42)  # Increased max_iter to allow for better convergence

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', mlp)
])

param_grid_pipeline = {
    'mlp__hidden_layer_sizes': param_grid['hidden_layer_sizes'],
    'mlp__activation': param_grid['activation'],
    'mlp__learning_rate_init': param_grid['learning_rate_init'],
}

# Define grid search
grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid_pipeline,
                           scoring='accuracy',
                           cv=cv,
                           n_jobs=-1,
                           verbose=1)

# Run grid search
grid_search.fit(X_train, y_train)

# Output best parameters
best_params = grid_search.best_params_

print("Best Parameters:")
print("Number of neurons in hidden layer:", best_params['mlp__hidden_layer_sizes'])
print("Activation function:", best_params['mlp__activation'])
print("Learning rate:", best_params['mlp__learning_rate_init'])
print("Best cross-validation accuracy:", grid_search.best_score_)

# ==============================
# 3. Train Final Model with Best Parameters
# ==============================

best_mlp = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(hidden_layer_sizes=best_params['mlp__hidden_layer_sizes'],
                          activation=best_params['mlp__activation'],
                          learning_rate_init=best_params['mlp__learning_rate_init'],
                          max_iter=1000,  # Increased max_iter to allow for better convergence
                          random_state=42))
])

best_mlp.fit(X_train, y_train)

# ==============================
# 4. Evaluate Model Performance on Test Set
# ==============================

# Make predictions on the test set
y_test_pred = best_mlp.predict(X_test)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
test_error_rate = 1 - test_accuracy

print("Test set accuracy: {:.4f}".format(test_accuracy))
print("Test set error rate: {:.4f}".format(test_error_rate))

# Output classification report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Plot confusion matrix
conf_mat = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=[-1, 1], yticklabels=[-1, 1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Test Set Confusion Matrix')
plt.show()

# ==============================
# 5. Plot Decision Boundary
# ==============================

# Define grid range
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict classification for grid points
Z = best_mlp.predict(np.c_[xx.ravel(), yy.ravel()])
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
plt.title('MLP Decision Boundary and Test Data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.show()
