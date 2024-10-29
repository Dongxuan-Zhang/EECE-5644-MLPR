import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
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

# Reduce HAR dataset features to 20 using PCA for classification only
pca_har = PCA(n_components=20)
X_har_train_pca = pca_har.fit_transform(X_har_train_reduced)
X_har_test_pca = pca_har.transform(X_har_test_reduced)

# Initialize and train Gaussian Naive Bayes classifier for Wine dataset
gnb_wine = GaussianNB()
gnb_wine.fit(X_wine_train_normalized, y_wine_train)

# Predict and evaluate on Wine dataset
y_wine_pred = gnb_wine.predict(X_wine_test_normalized)
accuracy_wine = accuracy_score(y_wine_test, y_wine_pred)
print(f"Wine Dataset Accuracy: {accuracy_wine:.2f}")

# Confusion Matrix for Wine dataset
cm_wine = confusion_matrix(y_wine_test, y_wine_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_wine, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix for Wine Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Initialize and train Gaussian Naive Bayes classifier for HAR dataset
gnb_har = GaussianNB()
gnb_har.fit(X_har_train_pca, y_har_train)

# Predict and evaluate on HAR dataset
y_har_pred = gnb_har.predict(X_har_test_pca)
accuracy_har = accuracy_score(y_har_test, y_har_pred)
print(f"HAR Dataset Accuracy: {accuracy_har:.2f}")

# Confusion Matrix for HAR dataset
cm_har = confusion_matrix(y_har_test, y_har_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_har, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix for HAR Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
