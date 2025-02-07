import numpy as np
import ot  # POT library
from sklearn.decomposition import PCA  # For principal component analysis
import matplotlib.pyplot as plt  # For plotting

# Function for gradient ascent on given data
def gradient_ascent_on_transport(y, n=10, eta_x=0.01, eta_mu=0.00001, eta_lambda_1=0.01, 
                                 eta_lambda_2=0.01, eta_lambda_3=0.01,  eta_lambda_1i=0.01,
                                 L=4.0, max_iter=30000, epsilon=1e-4, delta=1e-8):
    m = y.shape[0]  # Number of discrete points for nu
    nu = np.ones(m) / m  # Evenly distributed nu weights

    # Step 1: Principal Component Analysis (PCA) to find the principal direction of nu
    pca = PCA(n_components=1)  # We only need the first principal component
    pca.fit(y)
    principal_direction = pca.components_[0]  # Get the first principal component

    # Step 2: Initialize x evenly spaced along the principal direction of nu
    t = np.linspace(-1, 1, n)  # n evenly spaced values between -1 and 1
    x = t[:, None] * principal_direction  # Scale the principal direction by t

    # Step 3: Initialize mu with evenly distributed values
    mu = np.ones(n) / n  # Each mu_i is evenly distributed, so mu_i = 1/n

    # Initialize Lagrange multipliers
    lambda_1, lambda_2, lambda_3 = 0, 0, 0
    lambda_1i = np.zeros(n)  # Initialize lambda_{1,i} for non-negativity constraint

    # Helper function definitions for gradient ascent
    def pairwise_distances(x, y):
        return np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)

    def wasserstein_distance(mu, nu, x, y):
        cost_matrix = pairwise_distances(x, y)  # Pairwise squared distance
        W2 = ot.emd2(mu, nu, cost_matrix)  # Use POT's emd2 function to get the W2 distance
        pi = ot.emd(mu, nu, cost_matrix)  # Optimal transport plan
        return W2, cost_matrix, pi


    def grad_length_constraint(x):
        grad = np.zeros_like(x)
        # For i = 1, move x_1 toward x_2
        grad[0] = (x[1] - x[0]) / np.linalg.norm(x[1] - x[0])
        # For i = n, move x_n toward x_{n-1}
        grad[-1] = -(x[-1] - x[-2]) / np.linalg.norm(x[-1] - x[-2])
        # For i = 2 to n-1, adjust x_i relative to x_{i+1} and x_{i-1}
        for i in range(1, len(x) - 1):
            grad[i] = (x[i+1] - x[i]) / np.linalg.norm(x[i+1] - x[i]) - (x[i] - x[i-1]) / np.linalg.norm(x[i] - x[i-1])
        return grad

    # Initialize objective function value for comparison
    previous_obj = None

    # Gradient ascent loop
    for iter in range(max_iter):  
        # Compute Wasserstein distance and transport plan
        W2, cost_matrix, pi = wasserstein_distance(mu, nu, x, y)

        # Compute the length constraint: sum of distances between x_i and x_{i+1}
        length = np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1))

        # Compute gradients of the Lagrangian
        grad_x = (2 * x * mu[:, None] 
                  - 2 * np.sum(x * mu[:, None], axis=0) * mu[:, None] 
                  - 2 * lambda_3 * x * mu[:, None] 
                  - 2 * lambda_3 * np.sum(pi[:, :, None] * (x[:, None, :] - y[None, :, :]), axis=1) 
                  + lambda_2 * grad_length_constraint(x))
        grad_mu = (np.linalg.norm(x, axis=1)**2 
                   - 2 * np.sum(x * mu[:, None], axis=0).dot(x.T) 
                   + lambda_1 
                   - lambda_1i
                   - lambda_3 * (np.linalg.norm(x, axis=1)**2 + np.sum(pi * cost_matrix, axis=1)))

        # Objective function (variance maximization)
        obj_value = np.sum(np.linalg.norm(x, axis=1)**2 * mu)

        # Stopping criterion 1: Check if the change in objective function is small
        if previous_obj is not None and np.abs(obj_value - previous_obj) < delta:
            print(f"Stopping at iteration {iter}: Objective function converged")
            break

        # Stopping criterion 2: Check if the norm of the gradient (for x) is small
        grad_norm = np.linalg.norm(grad_x)
        if grad_norm < epsilon:
            print(f"Stopping at iteration {iter}: Gradient magnitude converged")
            break

        # Monitor progress every 500 iterations
        if iter % 5000 == 0:
            # Print constraint values
            rhs_lambda_3 = np.sum(np.linalg.norm(y, axis=1)**2 * nu) - np.sum(np.linalg.norm(x, axis=1)**2 * mu)
            print(f"Iteration {iter}: W2 = {W2}, RHS (lambda_3) = {rhs_lambda_3}")
            print(f"Length = {length}, L = {L}")

        # Update x and mu using gradient ascent
        x += eta_x * grad_x  # Gradient ascent for x
        mu += eta_mu * grad_mu  # Gradient ascent for mu

        # Project mu back onto the simplex (ensure sum(mu) = 1 and mu >= 0)
        mu = np.maximum(mu, 0)
        mu /= np.sum(mu)

        # Update Lagrange multipliers
        lambda_1 += eta_lambda_1 * (np.sum(mu) - 1)  # Probability constraint
        lambda_2 = max(0, lambda_2 + eta_lambda_2 * (length - L))  # Length constraint
        lambda_3 = max(0, lambda_3 + eta_lambda_3 * (W2 - (np.sum(np.linalg.norm(y, axis=1)**2 * nu) - np.sum(np.linalg.norm(x, axis=1)**2 * mu))))  # Kantorovich order constraint
        # Update Lagrange multipliers for non-negativity constraint
        lambda_1i = np.maximum(0, lambda_1i - eta_lambda_1i * mu)

        # Update the previous objective value for comparison in the next iteration
        previous_obj = obj_value

    # Output final results
    return x, mu, lambda_1, lambda_2, lambda_3, lambda_1i, W2


# Example of how to call the gradient_ascent_on_transport function with data
if __name__ == "__main__":
    # Generate data
    np.random.seed(45)  # For reproducibility
    m = 100
    sigma = 0.1
    z = np.linspace(-1, 1, m)
    noise = np.random.normal(0, sigma, size=(m, 2))
    # y = np.column_stack((z, z * np.abs(z))) + noise  # Modified data generation
    y = np.column_stack((z, z**2)) + noise 
    
    # Call the gradient ascent function
    x, mu, lambda_1, lambda_2, lambda_3, lambda_1i, W2 = gradient_ascent_on_transport(y)

    # Plot final data mu with nu
    plt.figure()
    plt.scatter(y[:, 0], y[:, 1], c='blue', label='Final data nu', alpha=0.5)
    plt.scatter(x[:, 0], x[:, 1], c='red', label='Final data mu', marker='x')
    plt.title('Final Data: mu and nu')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()
