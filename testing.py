import warnings

import matplotlib.pyplot as plt
import numpy as np

import openfiles
import utilities
from angles import draw_stroke
from openfiles import calculate_velocity, open_file_children, extra_smooth
from ranged_infelction_points import represent_curve, generate_curve_from_parameters


def calculate_snr_dB(y):
    signal_power = np.mean(y) ** 2
    noise_power = np.var(y)
    snr_dB = 10 * np.log10(signal_power / noise_power)
    return snr_dB


def angle(index, x, y, num_points=1):
    start_index = index
    end_index = min(index + num_points + 1, len(x)-1)
    dy = y[end_index] - y[start_index]
    dx = x[end_index] - x[start_index]
    angle_rad = np.arctan(dy/dx)

    # plt.plot(x_values, y_values, color="green")
    # plt.plot(x_values[start_index: end_index], y_values[start_index: end_index], color="orange")
    # plt.show()

    print(np.degrees(angle_rad))
    return angle_rad

def traveled_distance(D_par, sigma_par, i):
    if i == 1:
        return 0
    elif i == 5:
        return D_par
    elif i == 2 or i == 3 or i == 4:
        a = [np.nan, np.nan]
        a.append((3 / 2) * sigma_par ** 2 + sigma_par * np.sqrt(((sigma_par ** 2) / 4) + 1))
        a.append(sigma_par ** 2)
        a.append((3 / 2) * sigma_par ** 2 - sigma_par * np.sqrt(((sigma_par ** 2) / 4) + 1))
        return (D_par / 2) * (1 + np.math.erf(-a[i] / (sigma_par * (2 ** 0.5))))
    return None


def traveled_distance_2(t, t_0, D_par, sigma_par, meu_par):
    return (D_par / 2) * (1 + np.math.erf((np.log(t - t_0) - meu_par) / (sigma_par * (2 ** 0.5))))


def find_characteristic_points(x, y):
    t_3 = x[np.argmax(y)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        first_derivitive = np.gradient(x, y)
        second_derivitive = np.gradient(x, first_derivitive)
    inflection_points_x = x[np.where(np.diff(np.sign(second_derivitive)))[0] + 1]
    x_coordinate_max_value = x[np.argmax(y)]
    try:
        t_2, t_4 = (inflection_points_x[inflection_points_x < x_coordinate_max_value][-1],
                    inflection_points_x[inflection_points_x > x_coordinate_max_value][0])
    except IndexError:
        raise "small time, problem in data, no inlection points"
    try:
        t_1 = x[(y <= 0.01 * y[x == t_3]) & (x < t_3)][-1]
    except IndexError:
        t_1 = x[0]
    try:
        t_5 = x[(y <= 0.01 * y[x == t_3]) & (x > t_3)][0]
    except IndexError:
        t_5 = x[-1]
    characteristic_x = np.array([t_1, t_2, t_3, t_4, t_5])
    return np.where(np.isin(x, characteristic_x))[0]


def delt_angle(characteristic_times, x_values_par, y_values_par, D_par, sigma_par):
    characteristic_times = np.insert(characteristic_times, 0, -1000)
    term1_1 = angle(characteristic_times[2], x_values_par, y_values_par)
    term1_2 = angle(characteristic_times[4], x_values_par, y_values_par)
    term2_1 = traveled_distance(D_par, sigma_par, 2)
    term2_2 = traveled_distance(D_par, sigma_par, 4)
    return (term1_1 - term1_2) / (term2_1 - term2_2)


def estimate_angles(delta_phi_par, characteristic_points, x, y, D_par, sigma_par):
    term1 = angle(characteristic_points[2], x, y)
    term2 = traveled_distance(D, sigma_par, 3) - traveled_distance(D_par, sigma_par, 1)
    theta_start = term1 - delta_phi_par * term2
    term2 = traveled_distance(D, sigma_par, 5) - traveled_distance(D_par, sigma_par, 3)
    theta_end = term1 - delta_phi_par * term2
    return theta_start, theta_end


if __name__ == '__main__':
    # x_values, y_values, timestamps_arr = open_file_children(
    #     'DB\\3\\Dataset\\tablet\\4\\singletouch-draganddrop.xml', 3)
    x_values, y_values, timestamps_arr = openfiles.open_file_unistroke("DB\\1 unistrokes\\s01 (pilot)\\fast\\check02.xml")
    plt.plot(x_values, y_values, marker="o")
    velocity = calculate_velocity(x_values, y_values, timestamps_arr)
    smoothed_velocity = extra_smooth(velocity, int(0.3*len(velocity)), 2)
    plt.figure(5)
    plt.plot(timestamps_arr, smoothed_velocity)
    plt.show()
    strokes = represent_curve(timestamps_arr, smoothed_velocity.copy())
    regenerated_curve = generate_curve_from_parameters(strokes, timestamps_arr)
    plt.plot(timestamps_arr, velocity, label="original", color="cyan")
    plt.plot(timestamps_arr, regenerated_curve, label="regenerated")
    plt.plot(timestamps_arr, smoothed_velocity, label="smoothed", color="purple")
    print(utilities.calculate_MSE(regenerated_curve, smoothed_velocity))
    plt.legend()
    plt.figure(2)
    plt.plot(x_values, y_values, marker="o")
    plt.show()

    angles = []
    time = np.linspace(timestamps_arr[0], timestamps_arr[-1], len(timestamps_arr))
    X = np.zeros_like(time)
    Y = np.zeros_like(time)

    for stroke_order in range(len(strokes)):
        D = strokes[stroke_order][0]
        sigma = strokes[stroke_order][1]
        meu = strokes[stroke_order][2]
        x_0 = strokes[stroke_order][3]

        print("D: ", D)
        print("sigma: ", sigma)
        print("meu: ", meu)
        print("x_0: ", x_0)

        angular_curve = utilities.generate_lognormal_curve(D, sigma, meu, x_0, time[0], time[-1], len(time))
        characteristic_points_x = find_characteristic_points(time, angular_curve)
        # plt.plot(time, angular_curve)
        # plt.scatter(time[characteristic_points_x], angular_curve[characteristic_points_x])
        # plt.show()
        delta_phi = delt_angle(characteristic_points_x, x_values, y_values, D, sigma)
        theta_s, theta_e = estimate_angles(delta_phi, characteristic_points_x, x_values, y_values, D, sigma)
        angles.append((theta_s, theta_e))

    for stroke, angele in zip(strokes, angles):
        D = stroke[0]
        sigma = stroke[1]
        meu = stroke[2]
        x_0 = stroke[3]
        theta_s = angele[0]
        theta_e = angele[1]
        new_X, new_Y = draw_stroke(D, theta_s, theta_e , time, x_0, meu, sigma)
        if theta_s<0:
            theta_s+=np.pi
        if theta_e < 0:
            theta_e += np.pi
        print(f"D: {D}\t sigma: {sigma}\t meu: {meu}\t X_0:{x_0}\t theta_s: {np.degrees(theta_s)}\t theta_e: {np.degrees(theta_e)}")
        plt.plot(x_values, y_values)
        X[~np.isnan(new_X)] += new_X[~np.isnan(new_X)]
        Y[~np.isnan(new_Y)] += new_Y[~np.isnan(new_Y)]
        X -= np.min(X)
        Y -= np.min(Y)
        plt.plot(X, Y)
        plt.show()
