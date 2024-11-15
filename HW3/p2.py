import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# define true parameters
true_means = [
    [0, 0],
    [3, 3],    
    [-3, -3],
    [3, 0]   
]

true_covariances = [
    [[1, 0], [0, 1]],
    [[1, 0.8], [0.8, 1]],
    [[1, -0.6], [-0.6, 1]],
    [[1, 0], [0, 1]]
]

true_weights = [0.2, 0.3, 0.25, 0.25]

# set random seed to ensure reproducibility
np.random.seed(42)

def generate_data(n_samples):
    """Generate data according to the true GMM and return samples with their labels."""
    samples = []
    labels = []
    for _ in range(n_samples):
        component = np.random.choice(len(true_weights), p=true_weights)
        mean = true_means[component]
        cov = true_covariances[component]
        sample = np.random.multivariate_normal(mean, cov)
        samples.append(sample)
        labels.append(component)
    return np.array(samples), np.array(labels)

def select_best_k(X, max_k=10, n_splits=10):
    """Select the best GMM order using cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True)
    avg_log_likelihoods = []

    for k in range(1, max_k + 1):
        log_likelihoods = []
        valid_fold = False
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            # Check if training samples are sufficient
            if len(X_train) < k:
                continue  # Skip this fold if training samples are insufficient
            # Estimate GMM parameters
            gmm = GaussianMixture(n_components=k, covariance_type='full', max_iter=200, init_params='kmeans')
            gmm.fit(X_train)
            # Calculate log-likelihood on validation set
            log_likelihood = gmm.score(X_val)  # Average log-likelihood
            log_likelihoods.append(log_likelihood)
            valid_fold = True
        if valid_fold and log_likelihoods:
            avg_log_likelihood = np.mean(log_likelihoods)
        else:
            # Set to negative infinity if no valid folds
            avg_log_likelihood = -np.inf
        avg_log_likelihoods.append(avg_log_likelihood)
    # Select the model order with maximum average log-likelihood
    best_k = np.argmax(avg_log_likelihoods) + 1  # k starts from 1
    return best_k, avg_log_likelihoods

def experiment(n_samples, n_repeats=100, max_k=10):
    """Conduct experiments for specified dataset size, count frequency of best model order selection."""
    k_selection_counts = []
    # Adjust number of cross-validation folds
    if n_samples < 10:
        n_splits = n_samples  # Leave-one-out cross-validation
    else:
        n_splits = min(10, n_samples)

    for repeat in range(n_repeats):
        # Generate data
        X, _ = generate_data(n_samples)
        
        # Select best k
        best_k, avg_log_likelihoods = select_best_k(X, max_k=max_k, n_splits=n_splits)
        
        # Only plot log-likelihood values on first repeat
        if repeat == 0:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, max_k + 1), avg_log_likelihoods, 'b-o')
            plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
            plt.xlabel('Model Order (K)')
            plt.ylabel('Average Log-likelihood')
            plt.title(f'Dataset Size: {n_samples} - Average Log-likelihood vs Model Order')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'figures/log_likelihood_vs_k_{n_samples}.png', bbox_inches='tight', dpi=300)
            plt.close()
            
        k_selection_counts.append(best_k)
    
    # Count frequency of each k selection
    counts = Counter(k_selection_counts)
    # Calculate proportions
    total = sum(counts.values())
    proportions = {k: counts[k] / total for k in range(1, max_k+1)}
    return counts, proportions

def main():
    n_samples_list = [10, 100, 1000]
    n_repeats = 100
    max_k = 10

    results_counts = {}
    results_proportions = {}

    for n_samples in n_samples_list:
        print(f"Processing dataset size: {n_samples}")
        counts, proportions = experiment(n_samples, n_repeats=n_repeats, max_k=max_k)
        results_counts[n_samples] = counts
        results_proportions[n_samples] = proportions

    # Output results
    print("\nResults (Number of selections):")
    for n_samples in n_samples_list:
        counts = results_counts[n_samples]
        print(f"\nDataset size: {n_samples}")
        for k in range(1, max_k+1):
            print(f"Model order {k}: Selected {counts.get(k, 0)} times")
    
    # Output results in table format
    print("\nResults (Proportions):")
    print(f"{'Model order':<10} {'10 samples':<15} {'100 samples':<15} {'1000 samples':<15}")
    for k in range(1, max_k+1):
        proportions = [results_proportions[n_samples].get(k, 0) for n_samples in n_samples_list]
        print(f"{k:<10} {proportions[0]:<15.2f} {proportions[1]:<15.2f} {proportions[2]:<15.2f}")

    # Plot results
    for n_samples in n_samples_list:
        counts = results_counts[n_samples]
        ks = list(range(1, max_k+1))
        frequencies = [counts.get(k, 0) for k in ks]
        plt.figure()
        plt.bar(ks, frequencies)
        plt.xlabel('Model order K')
        plt.ylabel('Number of selections')
        plt.title(f'Dataset size: {n_samples}')
        plt.xticks(ks)
        plt.savefig(f'figures/dataset_size_{n_samples}_selections.png', bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
