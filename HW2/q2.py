import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# Function to generate synthetic data from a Gaussian Mixture Model (GMM)
def generate_data(N):
    gmm_parameters = {
        'priors': [.3, .4, .3],  # priors should be a row vector
        'mean_vectors': np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]]),
        'cov_matrices': np.zeros((3, 3, 3))
    }
    # Defining covariance matrices
    gmm_parameters['cov_matrices'][:, :, 0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmm_parameters['cov_matrices'][:, :, 1] = np.array([[8, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    gmm_parameters['cov_matrices'][:, :, 2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    
    # Ensure covariance matrices are positive definite
    for i in range(3):
        gmm_parameters['cov_matrices'][:, :, i] += np.eye(3) * 1e-6
    
    x, labels = generate_data_from_gmm(N, gmm_parameters)
    return x

# Function to generate data samples from a GMM
def generate_data_from_gmm(N, gmm_parameters):
    priors = gmm_parameters['priors']  # priors should be a row vector
    mean_vectors = gmm_parameters['mean_vectors']
    cov_matrices = gmm_parameters['cov_matrices']
    n = mean_vectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    x = np.zeros((n, N))
    labels = np.zeros(N)
    
    # Decide randomly which samples will come from each component
    u = np.random.random(N)
    thresholds = np.cumsum(priors)
    thresholds = np.insert(thresholds, 0, 0)
    
    for l in range(C):
        indl = np.where((u > thresholds[l]) & (u <= thresholds[l + 1]))[0]
        Nl = len(indl)
        labels[indl] = l
        x[:, indl] = np.transpose(np.random.multivariate_normal(mean_vectors[:, l], cov_matrices[:, :, l], Nl))

    return x, labels

# Function to plot 3D data
def plot3(a, b, c, mark="o", col="b", title='Training Dataset'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.set_title(title)
    plt.show()

# Function to transform input features for linear regression
def z(x):
    return np.asarray([np.ones(x.shape[1]), x[0], x[1], x[0] ** 2, x[0] * x[1], x[1] ** 2, x[0] ** 3, (x[0] ** 2) * x[1],
                       x[0] * (x[1] ** 2), x[1] ** 3])

# Function to define the MLE objective function
def func_mle(x, y):
    zx = z(x)
    
    def f(theta):
        w = theta[0:10]
        sigma = theta[10]
        if sigma <= 0:
            return np.inf  # To ensure sigma is positive
        return np.log(sigma) + np.mean((y - w @ zx) ** 2) / (2 * sigma ** 2)
    
    return f

# Function to define the MAP objective function
def func_map(x, y, gamma):
    zx = z(x)
    mu = np.zeros(zx.shape[0])
    sigma_w = np.eye(mu.size) * gamma
    
    def f(theta):
        w = theta[0:10]
        sigma_v = theta[10]
        if sigma_v <= 0:
            return np.inf  # To ensure sigma_v is positive
        prior_prob = multivariate_normal.pdf(w, mu, sigma_w)
        if prior_prob == 0:
            return np.inf  # Avoid log(0)
        return np.log(sigma_v) + np.mean((y - w @ zx) ** 2) / (2 * sigma_v ** 2) - np.log(prior_prob)
    
    return f

# Function to calculate the mean squared error
def loss(w, x, y):
    return np.mean((y - w @ z(x)) ** 2)

# Main function
def main():
    # Generate synthetic data for training and validation
    N_train = 100
    data_train = generate_data(N_train)
    plot3(data_train[0, :], data_train[1, :], data_train[2, :], title='Training Dataset')
    x_train = data_train[0:2, :]
    y_train = data_train[2, :]

    N_validate = 1000
    data_validate = generate_data(N_validate)
    plot3(data_validate[0, :], data_validate[1, :], data_validate[2, :], title='Validation Dataset')
    x_validate = data_validate[0:2, :]
    y_validate = data_validate[2, :]

    # MLE optimization
    res_mle = minimize(func_mle(x_train, y_train), np.random.random(11),
                       method='Nelder-Mead',
                       options={'maxiter': 10000})
    
    # Calculate MLE training and validation losses
    loss_mle_train = np.log(loss(res_mle.x[:-1], x_train, y_train) + 1e-6)  # Adding small value to avoid log(0)
    loss_mle_valid = np.log(loss(res_mle.x[:-1], x_validate, y_validate) + 1e-6)
    
    # MAP optimization for different values of gamma
    loss_map_train = []
    loss_map_valid = []
    m, n = -10, 10
    
    for i in range(m, n + 1):
        print('gamma = 10^', i)
        gamma = 10 ** i
        res_map = minimize(func_map(x_train, y_train, gamma),
                           np.random.random(11),
                           method='Nelder-Mead',
                           options={'maxiter': 2000})
        loss_map_train.append(np.log(loss(res_map.x[:-1], x_train, y_train) + 1e-6))
        loss_map_valid.append(np.log(loss(res_map.x[:-1], x_validate, y_validate) + 1e-6))
    
    # Plot the results
    plt.figure()
    plt.plot(range(m, n + 1), loss_map_train, label='P_TRAIN')
    plt.plot(range(m, n + 1), loss_map_valid, label='P_VALIDATION')
    plt.plot([-15, 15], [loss_mle_train, loss_mle_train], label='E_TRAIN')
    plt.plot([-15, 15], [loss_mle_valid, loss_mle_valid], label='E_VALIDATION')
    plt.xticks(range(m, n + 1))
    plt.ylabel('$\ln(loss)$')
    plt.xlabel('$\log_{10}\gamma$')
    plt.legend()
    plt.savefig('MAPandMLE.png')
    plt.show()

if __name__ == "__main__":
    main()
