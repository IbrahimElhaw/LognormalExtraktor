# this file hat help functions, which are not directly related to the algorithm
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


# this function generates a log normal y coordinations for given parameters and x coordinations
def generate_lognormal_curve(D, std_dev, mean, x_0, start, end, number_of_points):
    time = np.linspace(start, end, number_of_points, dtype="float64")
    curve = np.zeros_like(time, dtype="float64")
    if std_dev == 0:
        return curve
    # Calculate the curve only for values greater than or equal to x_0
    condition = time > x_0
    curve[condition] = (D / ((time[condition] - x_0) * std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(time[condition] - x_0) - mean) ** 2) / (2 * std_dev ** 2))
    # condition = curve < 0.005 * np.max(curve)
    # curve[condition] = 0
    return curve


# calculates MSE
def calculate_MSE(real_y_values, forged_yvalues):
    condition = ~np.isnan(real_y_values) & ~np.isnan(forged_yvalues)
    return np.mean((real_y_values[condition] - forged_yvalues[condition]) ** 2)


# this function is only used for testing reason. it generates a sigma lognormal curve, which consists of 4 profiles
def generate_4_lognormal_curves(timestamps):
    D_1 = 24  # Amplitude range(5 -> 70)
    std_dev_1 = 0.3  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_1 = -1.6  # Mean (meu) range(-2.2 -> -1.6)
    x_01 = 0.1  # shifted range(0 -> 1)

    D_2 = 15  # Amplitude range(5 -> 70)
    std_dev_2 = 0.3  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_2 = -1.6  # Mean (meu) range(-2.2 -> -1.6)
    x_02 = 0.3  # shifted range(0 -> 1)

    D_3 = 26  # Amplitude range(5 -> 70)
    std_dev_3 = 0.1  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_3 = -2  # Mean (meu) range(-2.2 -> -1.6)
    x_03 = 0.5  # shifted range(0 -> 1)

    D_4 = 45  # Amplitude range(5 -> 70)
    std_dev_4 = 0.2  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_4 = -1.4  # Mean (meu) range(-2.2 -> -1.6)
    x_04 = 0.5  # shifted range(0 -> 1)

    velocity1 = generate_lognormal_curve(D_1, std_dev_1, mean_1, x_01, timestamps[0], timestamps[-1], len(timestamps))
    velocity2 = generate_lognormal_curve(D_2, std_dev_2, mean_2, x_02, timestamps[0], timestamps[-1], len(timestamps))
    velocity3 = generate_lognormal_curve(D_3, std_dev_3, mean_3, x_03, timestamps[0], timestamps[-1], len(timestamps))
    velocity4 = generate_lognormal_curve(D_4, std_dev_4, mean_4, x_04, timestamps[0], timestamps[-1], len(timestamps))
    velocity_profile = velocity1 + velocity2 + velocity3 + velocity4

    return velocity_profile


# retunrs all indexes of local max of a give curve
def get_local_max(curve):
    local_maxs = np.array([i for i in range(1, len(curve) - 1) if
                           curve[i] > curve[i - 1] and curve[i] >
                           curve[i + 1]])
    return local_maxs


# retunrs all indexes of local min of a give curve
def get_local_min(curve):
    local_maxs = np.array([i for i in range(1, len(curve) - 1) if
                           curve[i] < curve[i - 1] and curve[i] <
                           curve[i + 1]])
    return local_maxs


def get_extrems(curve):
    return get_local_max(curve), get_local_min(curve)


# filters the local max and local min of the points under a given threshold, as the shall not represent any stroke.
# it removes the local max and local min under this threshold while making sure that between every 2 max
# there is a local min and between every 2 min there is a local max.
# moreover, it makes sure that number of local max = the number of local min
# for that the last point may be added as a local min
def correct_local_extrems(local_min, local_max,x_values, y_values, threshold = 0.01):
    summit = np.max(y_values)
    local_min_copy = local_min.copy()
    local_max_copy = local_max.copy()
    if len(local_min_copy)>0:
        y_values_local_min = y_values[local_min_copy]
    else:
        y_values_local_min = np.insert(np.array([]), 0, y_values[-1])
        local_min_copy = np.insert(local_min_copy, len(local_min_copy), len(y_values) - 1)
        local_min_copy = np.array([int(local_min_copy)])
    y_values_local_max = y_values[local_max_copy]
    if len(local_min_copy) == 0 or len(local_min_copy) < len(local_max_copy):
        y_values_local_min = np.insert(y_values[local_min_copy], len(local_min_copy) , y_values[-1]) # add the last point of y values as a local min
        local_min_copy = np.insert(local_min_copy, len(local_min_copy), len(y_values) - 1)
    for min, max, index_min, index_max in zip(y_values_local_min, y_values_local_max, local_min_copy, local_max_copy):
        min_to_remove = local_min_copy[np.where(y_values[local_min_copy] == min)[0]]
        max_to_remove = local_max_copy[np.where(y_values[local_max_copy] == max)[0]]
        condition_1 = max < (threshold * summit)
        condition_2 = (y_values[index_max] - y_values[index_min]) < (0.1 * np.max(y_values)) # 0.1
        if condition_1 or condition_2:
            local_min_copy = local_min_copy[local_min_copy != min_to_remove]
            local_max_copy = local_max_copy[local_max_copy != max_to_remove]
    return local_min_copy, local_max_copy


# returns the indecies of the charcteristic_points
def find_char_points_lognormal(x_values, sigma, meu, x_0):
    v1 = x_0+np.exp(meu-3*sigma)
    p1 = find_nearest_index(x_values, v1)
    v2 = x_0+np.exp(meu-(1.5*sigma**2+sigma*np.sqrt(0.25*sigma**2+1)))
    p2 = find_nearest_index(x_values, v2)
    v3 = x_0+np.exp(meu-sigma**2)
    p3 = find_nearest_index(x_values, v3)
    v4 = x_0+np.exp(meu-(1.5*sigma**2-sigma*np.sqrt(0.25*sigma**2+1)))
    p4 = find_nearest_index(x_values, v4)
    v5 = x_0+np.exp(meu+3*sigma)
    p5 = find_nearest_index(x_values, v5)
    return np.array([p1, p2, p3, p4, p5])


# returns the  index of the nearst value in the array to a given value.
def find_nearest_index(arr, value):
    absolute_diff = np.abs(np.array(arr) - value)
    return np.argmin(absolute_diff)



if __name__ == '__main__':
    x = np.linspace(0, 0.4, 3000)
    y = generate_lognormal_curve(60, 0.308, -2.2, 0.1, x[0], x[-1], len(x))
    charpoints = find_char_points_lognormal(x, 0.308, -2.2, 0.1)
    print(x[charpoints])
    plt.plot(x, y)
    plt.scatter(x[charpoints], y[charpoints])
    plt.show()