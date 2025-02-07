import numpy as np
import ot
import gurobipy as gp
import matplotlib.pyplot as plt
from gurobipy import GRB
import math

# Import the external function for Kantorovich order computation
import bounded_length_PC_kantarovich_order_for_call

# Objective function 
def c(x, y):
    return np.sum((x - y) ** 2, axis=-1)

def find_location(y, ga, no_clusters, L):
    no_data = y.shape[0]
    LL = L * L

    model = gp.Model("curve_fitting")

    # Decision variables
    x = model.addVars(no_clusters, 2, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="x")

    # Objective function
    obj = gp.quicksum([(x[i, 0] - y[j, 0]) * (x[i, 0] - y[j, 0]) * ga[i, j] for i in range(no_clusters) for j in range(no_data)]) + \
          gp.quicksum([(x[i, 1] - y[j, 1]) * (x[i, 1] - y[j, 1]) * ga[i, j] for i in range(no_clusters) for j in range(no_data)])
    
    # Minimize the objective function
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints: Bounded length
    length_expr = gp.QuadExpr()
    for i in range(no_clusters - 1):
        length_expr += (x[i + 1, 0] - x[i, 0]) * (x[i + 1, 0] - x[i, 0]) + (x[i + 1, 1] - x[i, 1]) * (x[i + 1, 1] - x[i, 1])

    model.addConstr(length_expr <= LL, "length_constraint")
    
    # Optimize model
    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
            print("New incumbent's objective:", obj)
    model.optimize(mycallback)

    # print out the optimal value
    print(f'Optimal objective value: {model.objVal}')

    if model.status == GRB.OPTIMAL:
        x_solution = np.array([[x[i, 0].X, x[i, 1].X] for i in range(no_clusters)])
        return model.objVal, x_solution
    else:
        print("No optimal solution found")
        return None

def find_coupling(x, y, nu, la, no_clusters, no_data):
    n, k = no_clusters, no_data
    X = x[:, np.newaxis, :] 
    Y = y[np.newaxis, :, :]
    C = c(X, Y)

    # Pre-calculate the squared distances
    dist_squared = np.sum((x[:, np.newaxis] - y) ** 2, axis=2)

    # Create a new model
    m = gp.Model("quadratic")

    # Create variables
    ga = m.addVars(n, k, vtype=GRB.CONTINUOUS, name="ga")

    # Set objective
    obj = gp.quicksum(dist_squared[i, j] * ga[i, j] for i in range(n) for j in range(k)) + \
        la * gp.quicksum(sum((sum(y[j][d] * ga[i, j] for j in range(k)) - x[i][d]) ** 2 for d in range(2)) for i in range(n))
    m.setObjective(obj, GRB.MINIMIZE)

    # Add constraints
    for j in range(k):
        m.addConstr(sum(ga[i, j] for i in range(n)) == nu[j])

    # Optimize model
    m.optimize()

    # Calculate the optimal solution
    ga_opt = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            ga_opt[i, j] = ga[i, j].X

    total_value = np.sum(ga_opt * C)

    m.setParam('OutputFlag', 0)

    return ga_opt, total_value

def recover_original_data(scaled_data, min_vals, max_vals, new_min=-1.5, new_max=1.5):
    # Reverse the scaling and shifting
    norm_data = (scaled_data - new_min) / (new_max - new_min)
    
    # Recover the original data
    original_data = norm_data * (max_vals - min_vals) + min_vals
    
    return original_data

def closest_point_and_distance(x, y):
    """
    Find the closest point or line segment in x for each point in y.
    """
    
    n = x.shape[0]
    m = y.shape[0]

    closest_points = np.zeros((m, 2))
    square_distances = np.zeros(m)
    project_set = np.zeros(m, dtype=int)
    
    for i, point_y in enumerate(y):
        # Calculate distances from point y to each point in x
        sq_distances_to_segments = np.zeros(n-1)
        closest_points_on_segments = np.zeros((n-1, 2))
        sq_distances_to_points = np.sum((x - point_y)**2, axis=1)

        # Calculate distances from point y to each line segment formed by consecutive points in x
        for j in range(len(x)-1):
            sq_distances_to_segments[j], closest_points_on_segments[j] = point_line_distance_and_closest_point(x[j], x[j+1], point_y)

        # Find the minimum distance
        min_sq_distance_to_point = sq_distances_to_points.min()
        min_sq_distance_to_segment = sq_distances_to_segments.min()
        
        # Determine which is closer, point or segment
        if min_sq_distance_to_point <= min_sq_distance_to_segment:
            closest_point_index = sq_distances_to_points.argmin()
            closest_point = x[closest_point_index]
            closest_distance = min_sq_distance_to_point
            project_index = closest_point_index
        else:
            closest_segment_index = sq_distances_to_segments.argmin()
            closest_point = closest_points_on_segments[closest_segment_index]
            closest_distance = min_sq_distance_to_segment
            project_index = n + closest_segment_index

        closest_points[i] = closest_point
        square_distances[i] = closest_distance
        project_set[i] = project_index

    return closest_points, square_distances, project_set

def initial_curve_with_pca(X, shift_X, no_data, no_clusters):
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    # Project the data onto the first principal component
    X_pca = np.dot(X, Vt[0])

    min_val = np.min(X_pca)
    max_val = np.max(X_pca)

    # Define t based on the range of the projected data
    t = np.linspace(min_val, max_val, no_clusters)
    cluster_width = (max_val - min_val) / no_clusters
    groups = np.zeros((no_data, no_clusters))
    groups_int = np.zeros((no_data, no_clusters), dtype=int)
    groups_no = np.zeros(no_clusters)

    for i in range(no_data):
        j = int((X_pca[i] - min_val) / cluster_width)
        # Ensure that the maximum value is assigned to the last group
        if j == no_clusters:
            j = no_clusters - 1
        groups[i, j] = 1 / no_data
        groups_int[i, j] = 1
        groups_no[j] += 1

    x = np.zeros((no_clusters,2))
    pca_mean = np.zeros(no_clusters)
    mu = np.zeros(no_clusters)
    for j in range(no_clusters):
        # find the barycenter of each cluster
        mu[j] = np.sum(groups[:, j])
        pca_mean[j] = np.sum(X_pca * groups_int[:, j]) / groups_no[j]

    x = np.outer(pca_mean, Vt[0])
    x = x + shift_X

    # calculate the total square distance from all the y to the corresponding baricenter
    total_length = 0
    for i in range(no_data):
        for j in range(no_clusters):
            total_length += groups[i, j] * ((y[i][0] - x[j][0]) ** 2 + (y[i][1] - x[j][1]) ** 2)

    return x, groups.T, total_length

def rescale_data(data, new_min=-1.5, new_max=1.5):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # Normalize to 0-1
    norm_data = (data - min_vals) / (max_vals - min_vals)
    
    # Scale to new range and shift
    scaled_data = norm_data * (new_max - new_min) + new_min
    
    # Return scaled data along with the scaling parameters
    return scaled_data, min_vals, max_vals

def calculate_wasserstein_distance(mu, nu, X1, X2):
    # Calculate the cost matrix
    M = ot.dist(X1, X2)

    # Calculate the Wasserstein distance
    wasserstein_distance = ot.emd2(mu, nu, M)

    return wasserstein_distance

def point_line_distance_and_closest_point(p1, p2, p3):
    """
    Calculate the perpendicular distance between each point in p3_array and the line formed by points p1 and p2.
    Also find the closest point on the line to each point in p3_array.
    If the closest point is not between p1 and p2, return infinity.
    """

    # Vector representing the line segment from p1 to p2
    line_vector = p2 - p1

    # Vector representing the line segment from p1 to p3
    point_vector = p3 - p1

    # Calculate the scalar projection of point_vector onto line_vector
    scalar_projection = np.dot(point_vector, line_vector) / np.linalg.norm(line_vector)

    # Calculate the projection vector
    projection_vector = scalar_projection * line_vector / np.linalg.norm(line_vector)

    # Calculate the perpendicular distance
    # Calculate the closest point on the line to p3
    # If the scalar projection is not between 0 and the length of the line segment, return distance to two end
    if scalar_projection <= 0: 
        # perpendicular_distance = np.linalg.norm(p3 - p1)
        perp_sq_dist = np.sum((p3 - p1)**2)
        closest_point = p1
    elif scalar_projection >= np.linalg.norm(line_vector):
        # perpendicular_distance = np.linalg.norm(p2 - p1)
        perp_sq_dist = np.sum((p2 - p1)**2)
        closest_point = p2
    else:
        # perpendicular_distance = np.linalg.norm(point_vector - projection_vector)
        perp_sq_dist = np.sum((point_vector - projection_vector)**2)
        closest_point = p1 + projection_vector

    return perp_sq_dist, closest_point

# --- PROCESS FLOW ---

# 0. Set parameters
objs = []
itr = 0
no_data = 200
no_clusters = 10
eta = 0.6
la = 0.006

np.random.seed(40)  # for reproducibility

# 1. Data Creation and preprocessing
sigma = 0.1
z = np.linspace(-1, 1, no_data)
z = np.column_stack((z, z**2))
noise = np.random.normal(0, sigma, size=(no_data, 2))
y = z + noise 

# transform the data y so that the mean of y is 0
z = z - y.mean(axis=0)
y = y - y.mean(axis=0)

y, y_rescale_min, y_rescale_max = rescale_data(y)
nu = np.ones(no_data) / no_data
mu_original = np.ones(no_data) / no_data

# 2. Run Kantorovich Order
x_kan_2, mu_kan_2, lambda_1_kan_2, lambda_2_kan_2, lambda_3_kan_2, lambda_1i_kan_2, W2_kan_2 = bounded_length_PC_kantarovich_order_for_call.gradient_ascent_on_transport(y, L=2)
closest_point_len_kan_2, _, _ = closest_point_and_distance(x_kan_2, y)
x_kan_plot_2 = recover_original_data(x_kan_2, y_rescale_min, y_rescale_max)
print(f'The Wassertein distance between Kantorovich order closest point with length 2 and orginal mu is {calculate_wasserstein_distance(mu_original, mu_original, closest_point_len_kan_2, z)}')
print(f'The total square distance between Kantorovich order with length 2 and orginal mu is {np.sqrt(np.sum((z - closest_point_len_kan_2) ** 2))}')

x_kan_3, mu_kan_3, lambda_1_kan_3, lambda_2_kan_3, lambda_3_kan_3, lambda_1i_kan_3, W2_kan_3 = bounded_length_PC_kantarovich_order_for_call.gradient_ascent_on_transport(y, L=3)
closest_point_len_kan_3, _, _ = closest_point_and_distance(x_kan_3, y)
x_kan_plot_3 = recover_original_data(x_kan_3, y_rescale_min, y_rescale_max)
print(f'The Wassertein distance between Kantorovich order closest point with length 3 and orginal mu is {calculate_wasserstein_distance(mu_original, mu_original, closest_point_len_kan_3, z)}')
print(f'The total square distance between Kantorovich order with length 3 and orginal mu is {np.sqrt(np.sum((z - closest_point_len_kan_3) ** 2))}')

x_kan_4, mu_kan_4, lambda_1_kan_4, lambda_2_kan_4, lambda_3_kan_4, lambda_1i_kan_4, W2_kan_4 = bounded_length_PC_kantarovich_order_for_call.gradient_ascent_on_transport(y, L=4)
closest_point_len_kan_4, _, _ = closest_point_and_distance(x_kan_4, y)
x_kan_plot_4 = recover_original_data(x_kan_4, y_rescale_min, y_rescale_max)
print(f'The Wassertein distance between Kantorovich order closest point with length 4 and orginal mu is {calculate_wasserstein_distance(mu_original, mu_original, closest_point_len_kan_4, z)}')
print(f'The total square distance between Kantorovich order with length 4 and orginal mu is {np.sqrt(np.sum((z - closest_point_len_kan_4) ** 2))}')

# 3. Run Fifth Method
y_shift = y - y.mean(axis=0)
x_convex, ga, initialL = initial_curve_with_pca(y_shift, y.mean(axis=0), no_data, no_clusters)
prev_L = np.sqrt(initialL)


while itr <= 100:
    print(f"Convex order Iteration: {itr}")

    obj_convex, x_convex = find_location(y, ga, no_clusters, 2)
    new_ga, total_value = find_coupling(x_convex, y, nu, la, no_clusters, no_data)

    objs.append([math.sqrt(obj_convex), math.sqrt(total_value)])


    print(f'The diiferent between prev_L and total_value is {prev_L - math.sqrt(total_value)}')
    if np.abs(prev_L - math.sqrt(total_value)) < 1e-4:
        print(itr)
        break
    else:
        ga = new_ga
        prev_L = math.sqrt(total_value)
        itr = itr + 1
closest_point_len, _, _ = closest_point_and_distance(x_convex, y)

x_con_plot = recover_original_data(x_convex, y_rescale_min, y_rescale_max)
print(f'The Wassertein distance between Length constrained closest point and orginal mu is {calculate_wasserstein_distance(mu_original, mu_original, closest_point_len, z)}')
print(f'The total square distance between Length constrained and orginal mu is {np.sqrt(np.sum((z - closest_point_len) ** 2))}')


# 4. Plot results

color_scale = ['#0000ff', '#ff0000', '#00ff00']

curve_div_no = no_data
y_plot = recover_original_data(y, y_rescale_min, y_rescale_max)

# Plot Kantorovich Dominance vs Convex Order
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(figsize=[7, 5])

# Plot the data points
ax.scatter(x=y_plot[:,0], y=y_plot[:,1], s=25, marker='s', color='black', alpha=.2, label='Observed Data Points')
ax.scatter(x=z[:,0], y=z[:,1], s=25, marker='s', color='#E7D046', alpha=.7, label='Un-polluted Data Points')

# Plot the Kantorovich Dominance curve
ax.plot(x_kan_plot_4[:,0], x_kan_plot_4[:,1], color=color_scale[0], linewidth=5, alpha=.85)
ax.scatter(x=x_kan_plot_4[:,0], y=x_kan_plot_4[:,1], color=color_scale[0], s=20, alpha=.75, label='Kantorovich dominance')
  
# Plot Convex order curve
ax.plot(x_con_plot[:,0], x_con_plot[:,1], color=color_scale[1], linewidth=4, alpha=.55  )
ax.scatter(x=x_con_plot[:,0], y=x_con_plot[:,1], color=color_scale[1], s=40, alpha  =.75, label='Convex order')

plt.legend(prop={'size': 20}, markerscale=2.5)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()



# Plot Kantorovich Dominance comparison
plt.rcParams['text.usetex'] = True
fig, ax = plt.subplots(figsize=[7, 5])

# Plot the data points
ax.scatter(x=y_plot[:,0], y=y_plot[:,1], s=25, marker='s', color='black', alpha=.2, label='Observed Data Points')
ax.scatter(x=z[:,0], y=z[:,1], s=25, marker='s', color='#E7D046', alpha=.7, label='Un-polluted Data Points')

# Plot the Kantorovich Dominance curve
ax.plot(x_kan_plot_2[:,0], x_kan_plot_2[:,1], color=color_scale[0], linewidth=5, alpha=.85)
ax.scatter(x=x_kan_plot_2[:,0], y=x_kan_plot_2[:,1], color=color_scale[0], s=20, alpha=.75, label='Length = 2')

ax.plot(x_kan_plot_3[:,0], x_kan_plot_3[:,1], color=color_scale[1], linewidth=5, alpha=.85)
ax.scatter(x=x_kan_plot_3[:,0], y=x_kan_plot_3[:,1], color=color_scale[1], s=20, alpha=.75, label='Length = 3')

ax.plot(x_kan_plot_4[:,0], x_kan_plot_4[:,1], color=color_scale[2], linewidth=5, alpha=.85)
ax.scatter(x=x_kan_plot_4[:,0], y=x_kan_plot_4[:,1], color=color_scale[2], s=20, alpha=.75, label='Length = 4')

plt.legend(prop={'size': 20}, markerscale=2.5)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.show()

