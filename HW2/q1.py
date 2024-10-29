import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

##################Data generation##################
# Class priors
P0 = 0.6  # Prior probability for Class 0
P1 = 0.4  # Prior probability for Class 1

# Gaussian parameters for Class 0
m01 = np.array([-0.9, -1.1])
C01 = np.array([[0.75, 0], [0, 1.25]])

m02 = np.array([0.8, 0.75])
C02 = np.array([[0.75, 0], [0, 1.25]])

# Gaussian parameters for Class 1
m11 = np.array([-1.1, 0.9])
C11 = np.array([[0.75, 0], [0, 1.25]])

m12 = np.array([0.9, -0.75])
C12 = np.array([[0.75, 0], [0, 1.25]])

def generate_samples(N):
    labels = np.random.choice([0, 1], size=N, p=[P0, P1])
    X = []
    Y = labels
    for label in labels:
        if label == 0:
            component = np.random.choice([1, 2])
            if component == 1:
                sample = np.random.multivariate_normal(m01, C01)
            else:
                sample = np.random.multivariate_normal(m02, C02)
        else:
            component = np.random.choice([1, 2])
            if component == 1:
                sample = np.random.multivariate_normal(m11, C11)
            else:
                sample = np.random.multivariate_normal(m12, C12)
        X.append(sample)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

# Training datasets
X_train_20, Y_train_20 = generate_samples(20)
X_train_200, Y_train_200 = generate_samples(200)
X_train_2000, Y_train_2000 = generate_samples(2000)

# Validation dataset
X_validate_10K, Y_validate_10K = generate_samples(10000)

# create 2x2 subplots   
plt.figure(figsize=(15, 12))

# training set 20 samples
plt.subplot(2, 2, 1)
plt.scatter(X_train_20[Y_train_20 == 0][:, 0], X_train_20[Y_train_20 == 0][:, 1], 
           label='Class 0', alpha=0.6, color='blue')
plt.scatter(X_train_20[Y_train_20 == 1][:, 0], X_train_20[Y_train_20 == 1][:, 1], 
           label='Class 1', alpha=0.6, color='red')
plt.title('Training Set (N=20)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# training set 200 samples
plt.subplot(2, 2, 2)
plt.scatter(X_train_200[Y_train_200 == 0][:, 0], X_train_200[Y_train_200 == 0][:, 1], 
           label='Class 0', alpha=0.6, color='blue')
plt.scatter(X_train_200[Y_train_200 == 1][:, 0], X_train_200[Y_train_200 == 1][:, 1], 
           label='Class 1', alpha=0.6, color='red')
plt.title('Training Set (N=200)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# training set 2000 samples
plt.subplot(2, 2, 3)
plt.scatter(X_train_2000[Y_train_2000 == 0][:, 0], X_train_2000[Y_train_2000 == 0][:, 1], 
           label='Class 0', alpha=0.6, color='blue')
plt.scatter(X_train_2000[Y_train_2000 == 1][:, 0], X_train_2000[Y_train_2000 == 1][:, 1], 
           label='Class 1', alpha=0.6, color='red')
plt.title('Training Set (N=2000)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# validation set 10000 samples
plt.subplot(2, 2, 4)
plt.scatter(X_validate_10K[Y_validate_10K == 0][:, 0], X_validate_10K[Y_validate_10K == 0][:, 1], 
           label='Class 0', alpha=0.6, color='blue')
plt.scatter(X_validate_10K[Y_validate_10K == 1][:, 0], X_validate_10K[Y_validate_10K == 1][:, 1], 
           label='Class 1', alpha=0.6, color='red')
plt.title('Validation Set (N=10000)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

plt.tight_layout()  # automatically adjust subplot layout
plt.show()

##################Part 1##################
def class_conditional_pdf_0(X):
    pdf1 = multivariate_normal.pdf(X, mean=m01, cov=C01)
    pdf2 = multivariate_normal.pdf(X, mean=m02, cov=C02)
    return 0.5 * pdf1 + 0.5 * pdf2

def class_conditional_pdf_1(X):
    pdf1 = multivariate_normal.pdf(X, mean=m11, cov=C11)
    pdf2 = multivariate_normal.pdf(X, mean=m12, cov=C12)
    return 0.5 * pdf1 + 0.5 * pdf2

def compute_posterior(X):
    p_X_given_0 = class_conditional_pdf_0(X)
    p_X_given_1 = class_conditional_pdf_1(X)
    numerator = p_X_given_1 * P1
    denominator = p_X_given_0 * P0 + p_X_given_1 * P1
    posterior_L1 = numerator / denominator
    return posterior_L1

posterior_L1 = compute_posterior(X_validate_10K)

# Make predictions based on the threshold of 0.5
predictions = (posterior_L1 > 0.5).astype(int)

# Calculate the minimum probability of error
error_rate = 1 - accuracy_score(Y_validate_10K, predictions)
print(f"Minimum Probability of Error (min-P(error)): {error_rate:.4f}")

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(Y_validate_10K, posterior_L1)
roc_auc = auc(fpr, tpr)

# Find the point corresponding to threshold=0.5
optimal_idx = np.argmin(np.abs(thresholds - 0.5))
optimal_threshold = thresholds[optimal_idx]
optimal_fpr = fpr[optimal_idx]
optimal_tpr = tpr[optimal_idx]

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
plt.scatter(optimal_fpr, optimal_tpr, color='red', label='Min P(error) point')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Theoretically Optimal Classifier')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Create a mesh grid
x_min, x_max = X_validate_10K[:, 0].min() - 1, X_validate_10K[:, 0].max() + 1
y_min, y_max = X_validate_10K[:, 1].min() - 1, X_validate_10K[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
posterior_grid = compute_posterior(grid)
Z = posterior_grid.reshape(xx.shape)

# Plot the contour and training examples
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z > 0.5, alpha=0.5, levels=[0, 0.5, 1], colors=['Darkblue', 'Darkred'])

# correct classification
correct_class_0 = (Y_validate_10K == 0) & (predictions == 0)
correct_class_1 = (Y_validate_10K == 1) & (predictions == 1)
plt.scatter(X_validate_10K[correct_class_0][:, 0], X_validate_10K[correct_class_0][:, 1], 
            label='Correct Class 0', alpha=0.5, marker='o', color='blue')
plt.scatter(X_validate_10K[correct_class_1][:, 0], X_validate_10K[correct_class_1][:, 1], 
            label='Correct Class 1', alpha=0.5, marker='o', color='red')

# incorrect classification
incorrect_class_0 = (Y_validate_10K == 0) & (predictions == 1)
incorrect_class_1 = (Y_validate_10K == 1) & (predictions == 0)
plt.scatter(X_validate_10K[incorrect_class_0][:, 0], X_validate_10K[incorrect_class_0][:, 1], 
            label='Incorrect Class 0', alpha=0.5, marker='x', color='cyan')
plt.scatter(X_validate_10K[incorrect_class_1][:, 0], X_validate_10K[incorrect_class_1][:, 1], 
            label='Incorrect Class 1', alpha=0.5, marker='x', color='magenta')

plt.title('Decision Boundary of Theoretically Optimal Classifier')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

##################Part 2##################
def transform_quadratic(X):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_quad = poly.fit_transform(X)
    return X_quad

# Logistic-linear classifier with 20 samples
clf_linear_20 = LogisticRegression(solver='lbfgs')
clf_linear_20.fit(X_train_20, Y_train_20)

# Predict on validation set
Y_pred_linear_20 = clf_linear_20.predict(X_validate_10K)
error_rate_linear_20 = 1 - accuracy_score(Y_validate_10K, Y_pred_linear_20)
print(f"Logistic-Linear Classifier with 20 samples - Error Rate: {error_rate_linear_20:.4f}")

# Compute probabilities for ROC curve
Y_scores_linear_20 = clf_linear_20.predict_proba(X_validate_10K)[:, 1]

# Logistic-linear classifier with 200 samples
clf_linear_200 = LogisticRegression(solver='lbfgs')
clf_linear_200.fit(X_train_200, Y_train_200)
Y_pred_linear_200 = clf_linear_200.predict(X_validate_10K)
error_rate_linear_200 = 1 - accuracy_score(Y_validate_10K, Y_pred_linear_200)
print(f"Logistic-Linear Classifier with 200 samples - Error Rate: {error_rate_linear_200:.4f}")
Y_scores_linear_200 = clf_linear_200.predict_proba(X_validate_10K)[:, 1]

# Logistic-linear classifier with 2000 samples
clf_linear_2000 = LogisticRegression(solver='lbfgs')
clf_linear_2000.fit(X_train_2000, Y_train_2000)
Y_pred_linear_2000 = clf_linear_2000.predict(X_validate_10K)
error_rate_linear_2000 = 1 - accuracy_score(Y_validate_10K, Y_pred_linear_2000)
print(f"Logistic-Linear Classifier with 2000 samples - Error Rate: {error_rate_linear_2000:.4f}")
Y_scores_linear_2000 = clf_linear_2000.predict_proba(X_validate_10K)[:, 1]

# Transform datasets
X_train_20_quad = transform_quadratic(X_train_20)
X_train_200_quad = transform_quadratic(X_train_200)
X_train_2000_quad = transform_quadratic(X_train_2000)
X_validate_10K_quad = transform_quadratic(X_validate_10K)

# Logistic-quadratic classifier with 20 samples
clf_quad_20 = LogisticRegression(solver='lbfgs', max_iter=1000)
clf_quad_20.fit(X_train_20_quad, Y_train_20)
Y_pred_quad_20 = clf_quad_20.predict(X_validate_10K_quad)
error_rate_quad_20 = 1 - accuracy_score(Y_validate_10K, Y_pred_quad_20)
print(f"Logistic-Quadratic Classifier with 20 samples - Error Rate: {error_rate_quad_20:.4f}")
Y_scores_quad_20 = clf_quad_20.predict_proba(X_validate_10K_quad)[:, 1]

# Logistic-quadratic classifier with 200 samples
clf_quad_200 = LogisticRegression(solver='lbfgs', max_iter=1000)
clf_quad_200.fit(X_train_200_quad, Y_train_200)
Y_pred_quad_200 = clf_quad_200.predict(X_validate_10K_quad)
error_rate_quad_200 = 1 - accuracy_score(Y_validate_10K, Y_pred_quad_200)
print(f"Logistic-Quadratic Classifier with 200 samples - Error Rate: {error_rate_quad_200:.4f}")
Y_scores_quad_200 = clf_quad_200.predict_proba(X_validate_10K_quad)[:, 1]

# Logistic-quadratic classifier with 2000 samples
clf_quad_2000 = LogisticRegression(solver='lbfgs', max_iter=1000)
clf_quad_2000.fit(X_train_2000_quad, Y_train_2000)
Y_pred_quad_2000 = clf_quad_2000.predict(X_validate_10K_quad)
error_rate_quad_2000 = 1 - accuracy_score(Y_validate_10K, Y_pred_quad_2000)
print(f"Logistic-Quadratic Classifier with 2000 samples - Error Rate: {error_rate_quad_2000:.4f}")
Y_scores_quad_2000 = clf_quad_2000.predict_proba(X_validate_10K_quad)[:, 1]

# Plot ROC curves for logistic-linear classifiers
plt.figure()
for scores, label in zip(
    [Y_scores_linear_20, Y_scores_linear_200, Y_scores_linear_2000],
    ['20 samples', '200 samples', '2000 samples']):
    fpr, tpr, _ = roc_curve(Y_validate_10K, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Linear ({label}), AUC={roc_auc:.2f}')
plt.title('ROC Curves for Logistic-Linear Classifiers')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Plot ROC curves for logistic-quadratic classifiers
plt.figure()
for scores, label in zip(
    [Y_scores_quad_20, Y_scores_quad_200, Y_scores_quad_2000],
    ['20 samples', '200 samples', '2000 samples']):
    fpr, tpr, _ = roc_curve(Y_validate_10K, scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Quadratic ({label}), AUC={roc_auc:.2f}')
plt.title('ROC Curves for Logistic-Quadratic Classifiers')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

def plot_decision_boundary(clf, X, Y, title, quadratic=False):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]
    if quadratic:
        grid = transform_quadratic(grid)
    if quadratic:
        X = transform_quadratic(X)
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.5, levels=np.linspace(0, 1, 3), colors=['Darkblue', 'Darkred'])

    # predictions result
    predictions = clf.predict(X)

    # correct classification
    correct_class_0 = (Y == 0) & (predictions == 0)
    correct_class_1 = (Y == 1) & (predictions == 1)
    plt.scatter(X[correct_class_0][:, 0], X[correct_class_0][:, 1], c='blue', marker='o', label='Correct Class 0', edgecolors='k')
    plt.scatter(X[correct_class_1][:, 0], X[correct_class_1][:, 1], c='red', marker='o', label='Correct Class 1', edgecolors='k')

    # incorrect classification
    incorrect_class_0 = (Y == 0) & (predictions == 1)
    incorrect_class_1 = (Y == 1) & (predictions == 0)
    plt.scatter(X[incorrect_class_0][:, 0], X[incorrect_class_0][:, 1], c='cyan', marker='x', label='Incorrect Class 0', edgecolors='k')
    plt.scatter(X[incorrect_class_1][:, 0], X[incorrect_class_1][:, 1], c='magenta', marker='x', label='Incorrect Class 1', edgecolors='k')

    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# Plot for logistic-linear classifier trained on 2000 samples and 10k samples
plot_decision_boundary(clf_linear_2000, X_train_2000, Y_train_2000, 'Logistic-Linear Decision Boundary (2000 samples)')
plot_decision_boundary(clf_linear_2000, X_validate_10K, Y_validate_10K, 'Logistic-Linear Decision Boundary (10k samples)')


# Plot for logistic-quadratic classifier trained on 2000 samples and 10k samples
plot_decision_boundary(clf_quad_2000, X_train_2000, Y_train_2000, 'Logistic-Quadratic Decision Boundary (2000 samples)', quadratic=True)
plot_decision_boundary(clf_quad_2000, X_validate_10K, Y_validate_10K, 'Logistic-Quadratic Decision Boundary (10k samples)', quadratic=True)

