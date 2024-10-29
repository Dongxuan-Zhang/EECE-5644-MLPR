import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

np.random.seed(42)  # Set a random seed for reproducibility

SIGMA_X = 0.25
SIGMA_Y = 0.25
SIGMA_I = 0.3

CONTOUR_LEVELS = np.geomspace(0.0001, 250, 100)

def random_unit_circle_coords():
    r = np.sqrt(np.random.uniform(0, 1))
    theta = np.random.uniform(0, 2) * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])

def get_range_measurements(K, xy_true):
    return [generate_measurement(landmark_pos(i, K), xy_true) for i in range(K)]

def landmark_pos(i, K):
    angle = 2 * np.pi / K * i
    x = np.cos(angle)
    y = np.sin(angle)
    return np.array([x, y])

def generate_measurement(xy_landmark, xy_true):
    dTi = np.linalg.norm(xy_true - xy_landmark)
    for _ in range(100):  # Limit attempts to avoid infinite loop
        noise = np.random.normal(0, SIGMA_I)
        measurement = dTi + noise
        if measurement >= 0:
            return measurement
    return dTi  # Fallback to deterministic value if noise fails

def plot_equilevels(range_measurements, xy_true):
    gridpoints = np.meshgrid(np.linspace(-2, 2, 128), np.linspace(-2, 2, 128))
    contour_values = MAP_objective(gridpoints, range_measurements)

    plt.style.use('default')
    ax = plt.gca()

    # 画单位圆
    unit_circle = plt.Circle((0, 0), 1, color='#888888', fill=False, label='Unit Circle')
    ax.add_artist(unit_circle)

    # 画等高线
    contour = plt.contour(gridpoints[0], gridpoints[1], contour_values, 
                         cmap='plasma_r', levels=CONTOUR_LEVELS)

    # 画地标和测量圆
    landmarks = []
    range_circles = []
    for (i, r_i) in enumerate(range_measurements):
        (x, y) = landmark_pos(i, len(range_measurements))
        landmark = plt.plot((x), (y), 'o', color='g', markerfacecolor='none', 
                          label='Landmark' if i == 0 else "")[0]
        range_circle = plt.Circle((x, y), r_i, color='#0000bb', alpha=0.2, 
                                fill=False, label='Range Circle' if i == 0 else "")
        ax.add_artist(range_circle)
        landmarks.append(landmark)
        range_circles.append(range_circle)

    # 画真实位置
    true_pos = ax.plot([xy_true[0]], [xy_true[1]], '+', color='r', 
                      label='True Position')[0]

    ax.set_xlabel("x coordinate")
    ax.set_ylabel("y coordinate")
    ax.set_title("MAP estimation objective contours, K = " + str(len(range_measurements)))

    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    
    # 添加图例
    plt.legend(loc='upper right')
    
    # 添加颜色条
    plt.colorbar(label='Objective Value')
    
    plt.show()

def MAP_objective(xy, range_measurements):
    xy = np.expand_dims(np.transpose(xy, axes=(1, 2, 0)), axis=len(np.shape(xy))-1)
    prior = np.matmul(xy, np.linalg.inv(np.array([[SIGMA_X**2, 0], [0, SIGMA_Y**2]])))
    prior = np.matmul(prior, np.swapaxes(xy, 2, 3))
    prior = np.squeeze(prior)
    range_sum = 0

    for (i, r_i) in enumerate(range_measurements):
        xy_i = landmark_pos(i, len(range_measurements))
        d_i = np.linalg.norm(xy - xy_i[None, None, None, :], axis=3)
        range_sum += np.squeeze((r_i - d_i)**2 / SIGMA_I**2)

    return prior + range_sum

xy_true = random_unit_circle_coords()

for K in [1, 2, 3, 4]:
    range_measurements = get_range_measurements(K, xy_true)
    plot_equilevels(range_measurements, xy_true)
