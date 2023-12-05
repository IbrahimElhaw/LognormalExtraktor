# this file hat help functions, which are not directly related to the algorithm
import numpy as np


# this function generates a log normal y coordinations for given parameters and x coordinations
def generate_lognormal_curve(D, std_dev, mean, x_0, start, end, number_of_points):
    time = np.linspace(start, end, number_of_points)
    curve = np.zeros_like(time)
    if std_dev == 0:
        return curve
    # Calculate the curve only for values greater than or equal to x_0
    condition = time > x_0
    curve[condition] = (D / ((time[condition] - x_0) * std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(time[condition] - x_0) - mean) ** 2) / (2 * std_dev ** 2))
    # condition = curve < 0.005 * np.max(curve)
    # curve[condition] = 0
    return curve


# calculates MSE, and returns the square root of it
def calculate_MSE(real_y_values, forged_yvalues):
    return np.sqrt(np.mean((real_y_values - forged_yvalues) ** 2))


# this function is only used for testing reason. it generates a sigma lognormal curve, which consists of 4 profiles
def generate_4_lognormal_curves(timestamps):
    D_1 = 18  # Amplitude range(5 -> 70)
    std_dev_1 = 0.3  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_1 = -1.8  # Mean (meu) range(-2.2 -> -1.6)
    x_01 = 0.1  # shifted range(0 -> 1)

    D_2 = 15  # Amplitude range(5 -> 70)
    std_dev_2 = 0.3  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_2 = -1.6  # Mean (meu) range(-2.2 -> -1.6)
    x_02 = 0.25  # shifted range(0 -> 1)

    D_3 = 26  # Amplitude range(5 -> 70)
    std_dev_3 = 0.1  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_3 = -1.7  # Mean (meu) range(-2.2 -> -1.6)
    x_03 = 0.5  # shifted range(0 -> 1)

    D_4 = 40  # Amplitude range(5 -> 70)
    std_dev_4 = 0.2  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_4 = -2.1  # Mean (meu) range(-2.2 -> -1.6)
    x_04 = 0.7  # shifted range(0 -> 1)

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
        if condition_1:
            local_min_copy = local_min_copy[local_min_copy != min_to_remove]
            local_max_copy = local_max_copy[local_max_copy != max_to_remove]
    return local_min_copy, local_max_copy


# returns the indexes inflection points of a graph. it assumes that the graph only 2 inflection points has.
# this works for perfect lognormal profiles
def inflection_points(x, y):
    first_dervitive = np.gradient(y, x)
    index1 = np.argmax(first_dervitive)
    index2 = np.argmin(first_dervitive)
    return index1, index2

# returns the indexes edge points of a lognormal graph (t_1, t_5).
def edge_points(x, y):
    t_3 = np.argmax(y)
    try:
        t_1 = x[(y >= 0.01 * y[t_3]) & (x < x[t_3])][0]
    except IndexError:
        t_1 = x[0]
    try:
        t_5 = x[(y >= 0.01 * y[t_3]) & (x > x[t_3])][-1]
    except IndexError:
        t_5 = x[-1]
    t_1_id, t_5_id = np.where(np.isin(x, [t_1, t_5]))[0]
    return t_1_id, t_5_id