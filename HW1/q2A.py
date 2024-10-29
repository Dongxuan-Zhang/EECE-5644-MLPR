import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
dataset = np.load('dataset_Q2.npz')
data = dataset['data']
labels = dataset['labels']

# Total number of samples
num_samples = len(data)

# Define mean vectors
mu1 = np.array([0, 0, 0])
mu2 = np.array([2.5, 0, 0])
mu3 = np.array([0, 2.5, 0])
mu4 = np.array([2.5, 2.5, 0])

# Define prior probabilities
P_w1 = 0.3
P_w2 = 0.4
P_w3 = 0.3

# Minimum error probability Bayesian decision rule (using 0-1 loss)
def decision_rule_bayes(x):
    # Calculate conditional probability density for each class
    p1 = np.exp(-0.5 * np.sum((x - mu1) ** 2)) * P_w1
    p2 = np.exp(-0.5 * np.sum((x - mu2) ** 2)) * P_w2
    p3 = 0.5 * (np.exp(-0.5 * np.sum((x - mu3) ** 2)) + np.exp(-0.5 * np.sum((x - mu4) ** 2))) * P_w3
    # Make decision based on maximizing posterior probability
    return np.argmax([p1, p2, p3]) + 1

# Classify 10K samples and count each decision-label pair
confusion_matrix = np.zeros((3, 3))
predictions = []
for i in range(num_samples):
    x = data[i]
    true_label = labels[i]
    predicted_label = decision_rule_bayes(x)
    predictions.append(predicted_label)
    confusion_matrix[predicted_label - 1, true_label - 1] += 1

print("Confusion matrix:")
print(confusion_matrix / num_samples)

# 3D scatter plot to display data, distinguishing correct and incorrect classifications with different shapes and colors
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create legend handles
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Class 1 (Correct)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Class 1 (Incorrect)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='g', markersize=10, label='Class 2 (Correct)'),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='r', markersize=10, label='Class 2 (Incorrect)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='g', markersize=10, label='Class 3 (Correct)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='r', markersize=10, label='Class 3 (Incorrect)')
]

# Plot samples
for i in range(num_samples):
    x = data[i]
    true_label = labels[i]
    predicted_label = predictions[i]
    if true_label == predicted_label:
        color = 'g'  # Correct classification in green
    else:
        color = 'r'  # Incorrect classification in red

    if true_label == 1:
        marker = 'o'
    elif true_label == 2:
        marker = '^'
    else:
        marker = 's'

    ax.scatter(x[0], x[1], x[2], c=color, marker=marker)

# Set legend and labels
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('X3')
ax.legend(handles=legend_elements, loc='upper right')
ax.set_title('3D Scatter Plot of Data Classification Results')
plt.show()
