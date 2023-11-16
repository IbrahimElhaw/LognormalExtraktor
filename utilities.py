import numpy as np


# Parameters for the sigma log-normal distribution
# D_1 = 25  # Amplitude range(5 -> 70)
# std_dev_1 = 0.3  # Standard deviation (sigma) range(0.1 -> 0.45)
# mean_1 = -2  # Mean (meu) range(-2.2 -> -1.6)
# x_01 = 0.5  # shifted range(0 -> 1)
# velocity_profile = generate_lognormal_curve(D_1, std_dev_1, mean_1, x_01, seconds)



def generate_lognormal_curve(D, std_dev, mean, x_0, seconds):
    time = np.linspace(0.01, seconds, int(200 * seconds))
    curve = np.zeros_like(time)
    if std_dev == 0:
        return curve
    # Calculate the curve only for values greater than or equal to x_0
    condition = time > x_0
    curve[condition] = (D / ((time[condition] - x_0) * std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(time[condition] - x_0) - mean) ** 2) / (2 * std_dev ** 2))
    return curve

# 1 43 - 69
# 2 60 - 75
def corresponding_x_values(x_values, velocity_profile, v_3, x_3, sigma_accuracy = 200):
    condition = (velocity_profile < (0.43 * v_3)) & (x_values < x_3)
    corresponding_x_value1 = x_values[condition][-1]
    condition = (velocity_profile < (0.60 * v_3)) & (x_values < x_3)
    corresponding_x_value2 = x_values[condition][-1]
    condition = (velocity_profile < (0.60 * v_3)) & (x_values > x_3)
    corresponding_x_value3 = x_values[condition][0]
    condition = (velocity_profile < (0.75 * v_3)) & (x_values > x_3)
    corresponding_x_value4 = x_values[condition][0]
    x_values_v2_inf1 = np.linspace(corresponding_x_value1, corresponding_x_value2, sigma_accuracy)
    x_values_v4_inf2 = np.linspace(corresponding_x_value3, corresponding_x_value4, sigma_accuracy)
    return x_values_v2_inf1, x_values_v4_inf2


def calculate_MSE(real_y_values, forged_yvalues):
    return np.mean((real_y_values - forged_yvalues) ** 2)


def generate_4_lognormal_curves(seconds):
    D_1 = 15  # Amplitude range(5 -> 70)
    std_dev_1 = 0.1  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_1 = -1.8  # Mean (meu) range(-2.2 -> -1.6)
    x_01 = 0.1  # shifted range(0 -> 1)

    D_2 = 40  # Amplitude range(5 -> 70)
    std_dev_2 = 0.2  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_2 = -1.9  # Mean (meu) range(-2.2 -> -1.6)
    x_02 = 0.2  # shifted range(0 -> 1)

    D_3 = 40  # Amplitude range(5 -> 70)
    std_dev_3 = 0.3  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_3 = -2  # Mean (meu) range(-2.2 -> -1.6)
    x_03 = 0.4  # shifted range(0 -> 1)

    D_4 = 20  # Amplitude range(5 -> 70)
    std_dev_4 = 0.2  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_4 = -1.8  # Mean (meu) range(-2.2 -> -1.6)
    x_04 = 0.5  # shifted range(0 -> 1)

    velocity1 = generate_lognormal_curve(D_1, std_dev_1, mean_1, x_01,seconds)
    velocity2 = generate_lognormal_curve(D_2, std_dev_2, mean_2, x_02, seconds)
    velocity3 = generate_lognormal_curve(D_3, std_dev_3, mean_3, x_03, seconds)
    velocity4 = generate_lognormal_curve(D_4, std_dev_4, mean_4, x_04, seconds)
    velocity_profile = velocity1 + velocity2 + velocity3 + velocity4
    return velocity_profile
