import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from tqdm import tqdm  # Optional, for displaying progress bars

# -----------------------------
# 1. Preprocessing Steps
# -----------------------------

# Load the image (please replace with the actual path of your selected image from the BSDS300 dataset)
image_path = 'BSDS300/images/test/16077.jpg'  # Replace with your image path
image = Image.open(image_path)
image = image.convert('RGB')  # Ensure the image is in RGB format
image_np = np.array(image)

# Get the dimensions of the image
rows, cols, channels = image_np.shape

# Optionally, downsample the image if needed to reduce computation
# Downsampling factor (adjust as needed)
downsample_factor = 1  # Set to 1 if no downsampling is required
if downsample_factor > 1:
    image_np = image_np[::downsample_factor, ::downsample_factor, :]
    rows, cols, channels = image_np.shape

# Generate a meshgrid of row and column indices
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

# Flatten and combine the row indices, column indices, and RGB values into feature vectors
feature_vectors = np.column_stack((
    Y.ravel(),                    # Row indices
    X.ravel(),                    # Column indices
    image_np[:, :, 0].ravel(),    # Red channel values
    image_np[:, :, 1].ravel(),    # Green channel values
    image_np[:, :, 2].ravel()     # Blue channel values
))

# Independently normalize each feature to the [0, 1] range
normalized_features = np.zeros_like(feature_vectors, dtype=np.float64)

for i in range(feature_vectors.shape[1]):
    min_val = feature_vectors[:, i].min()
    max_val = feature_vectors[:, i].max()
    range_val = max_val - min_val
    # Avoid division by zero
    if range_val == 0:
        normalized_features[:, i] = 0.0
    else:
        normalized_features[:, i] = (feature_vectors[:, i] - min_val) / range_val

# -----------------------------
# 2. GMM Model Training and Selection
# -----------------------------

print('Performing GMM model training and selection...')

# Optionally, select a subset of data for model selection to speed up computation
subset_size = 10000  # Adjust as needed
if normalized_features.shape[0] > subset_size:
    np.random.seed(42)
    indices = np.random.choice(normalized_features.shape[0], subset_size, replace=False)
    features_subset = normalized_features[indices]
else:
    features_subset = normalized_features

# Set the range of component numbers to try
component_range = range(2, 21)  # From 2 to 20

# Record the average validation log-likelihood for each number of components
avg_validation_scores = []

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for n_components in component_range:
    validation_log_likelihood = []
    for train_index, test_index in kf.split(features_subset):
        X_train, X_test = features_subset[train_index], features_subset[test_index]
        # Create the GMM model
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        # Train the model
        gmm.fit(X_train)
        # Compute the log-likelihood on the validation set
        log_likelihood = gmm.score_samples(X_test)
        validation_log_likelihood.append(np.mean(log_likelihood))
    # Compute the average validation log-likelihood
    avg_score = np.mean(validation_log_likelihood)
    avg_validation_scores.append(avg_score)
    print(f'Number of components: {n_components}, Average validation log-likelihood: {avg_score:.2f}')

# Find the number of components with the highest average validation log-likelihood
best_n_components = component_range[np.argmax(avg_validation_scores)]
print(f'The best number of components is: {best_n_components}')

# -----------------------------
# 3. MAP Classification using the Best GMM
# -----------------------------

print('Performing MAP classification using the best GMM...')

# Retrain the GMM using all data with the best number of components
best_gmm = GaussianMixture(n_components=best_n_components, covariance_type='full', random_state=42)
best_gmm.fit(normalized_features)

# Compute the posterior probabilities for each pixel
posterior_probs = best_gmm.predict_proba(normalized_features)

# Assign each pixel to the component with the highest posterior probability (MAP classification)
labels = np.argmax(posterior_probs, axis=1)

# Generate random colors for each component label
np.random.seed(42)  # For reproducibility
colors = np.random.randint(0, 255, size=(best_n_components, 3))

# Map the labels to colors
label_colors = colors[labels]

# Reshape the colors to the original image shape
segmented_image = label_colors.reshape((rows, cols, 3)).astype(np.uint8)

# -----------------------------
# 4. Display Results
# -----------------------------

# Create a figure to display the original image and the segmented image side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Display the original image
axes[0].imshow(image_np)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display the color segmented image
axes[1].imshow(segmented_image)
axes[1].set_title('GMM Segmentation (Color)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
