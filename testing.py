# this file is the main file. The functions in this file aim to estimate the start and end angles, the main method
# is used to test the actual advance in the implementing. The file uses all other files.

import matplotlib.pyplot as plt
import numpy as np

import openfiles
import utilities
from draw_profile import draw_stroke
from openfiles import calculate_velocity, open_file_children, extra_smooth
from ranged_infelction_points import represent_curve, generate_curve_from_parameters


# not used yet.
def calculate_snr_dB(y):
    signal_power = np.mean(y) ** 2
    noise_power = np.var(y)
    snr_dB = 10 * np.log10(signal_power / noise_power)
    return snr_dB


# gets tha angle of at given index from the graph x and y, num_points is the number of points
# from which the average is used to get the angle. this is because some noise can effect the
# angle significantly
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


# the distance travelled at some characteristic points
# ref https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# section 3, function (34), refered as l(t_i)
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


# TODO: try it instead of traveled_distance
# the distance travelled at some characteristic points
# ref https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# section 3, function (33), refered as l(t-t_0, D, meu, sigma), not in use yet
def traveled_distance_2(t, D, sigma, mu, t0):
    t = t - t0
    t[t<=0] = 0
    dist = D / 2. * (1 + np.math.erf((np.log(t) - mu) / sigma * 0.707107))
    return dist


# returns the 5 characteristic_points of lognormal profile in following order:
# begin, first inflection point, maxima, second inflection point, end
# as defined in
# https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
def find_characteristic_points(x, y):
    t_3 = np.argmax(y)
    t_2, t_4 = utilities.inflection_points(x, y)
    t_1, t_5 = utilities.edge_points(x, y)
    characteristic_x = np.array([t_1, t_2, t_3, t_4, t_5])
    return characteristic_x


# returns the change in angle as defined in
# ref https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# section 3, function (45)
def delt_angle(characteristic_times, x_values_par, y_values_par, D_par, sigma_par):
    plt.plot(x_values_par, y_values_par, marker="o", zorder=1, label="actual drawing")
    plt.scatter(x_values_par[characteristic_times],y_values_par[characteristic_times], color="red", zorder=2,
                label="the points in the\n drawing which correspond the \ncharacteristic points in lognormal profile")
    plt.title("characteristic points in actual drawing")
    plt.xlabel("x-co")
    plt.ylabel("y-co")
    plt.legend()
    plt.show()
    characteristic_times = np.insert(characteristic_times, 0, -1000)
    term1_1 = angle(characteristic_times[2], x_values_par, y_values_par)
    a = np.degrees(term1_1)
    term1_2 = angle(characteristic_times[4], x_values_par, y_values_par)
    b = np.degrees(term1_2)
    term2_1 = traveled_distance(D_par, sigma_par, 2)
    term2_2 = traveled_distance(D_par, sigma_par, 4)
    return (term1_1 - term1_2) / (term2_1 - term2_2)


# returns start and end angle as defined in
# ref https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# section 3, functions (36a) and (36b)
def estimate_angles(delta_phi_par, characteristic_points_par, x, y, D_par, sigma_par):
    term1 = angle(characteristic_points_par[2], x, y)
    term2 = traveled_distance(D, sigma_par, 3) - traveled_distance(D_par, sigma_par, 1)
    theta_start = term1 - delta_phi_par * term2
    term2 = traveled_distance(D, sigma_par, 5) - traveled_distance(D_par, sigma_par, 3)
    theta_end = term1 - delta_phi_par * term2
    return theta_start, theta_end


if __name__ == '__main__':
    # x_values, y_values, timestamps_arr = open_file_children(
    #     'DB\\3\\Dataset\\tablet\\4\\singletouch-draganddrop.xml', 3)
    x_values, y_values, timestamps_arr, smoothed_velocity, velocity = openfiles.open_file_unistroke("DB\\1 unistrokes\\s01 (pilot)\\fast\\right_sq_bracket05.xml")
    plt.figure(5)
    plt.plot(timestamps_arr, smoothed_velocity, label="velocity")
    plt.plot(timestamps_arr, velocity, label="veolcity before smoothing")
    plt.title("smoothed velocity")
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.legend()
    plt.show()
    strokes = represent_curve(timestamps_arr, smoothed_velocity.copy())
    regenerated_curve = generate_curve_from_parameters(strokes, timestamps_arr)
    plt.plot(timestamps_arr, velocity, label="original", color="cyan")
    plt.plot(timestamps_arr, regenerated_curve, label="regenerated")
    plt.plot(timestamps_arr, smoothed_velocity, label="smoothed", color="purple")
    plt.legend()
    plt.figure(2)
    plt.plot(x_values, y_values, marker="o")
    plt.show()
    print(utilities.calculate_MSE(regenerated_curve, smoothed_velocity))

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
        plt.plot(time, angular_curve)
        plt.scatter(time[characteristic_points_x], angular_curve[characteristic_points_x], label="characteristic points")
        plt.title("characteristic points of actual curve")
        plt.xlabel("time")
        plt.ylabel("velocity")
        plt.legend()
        plt.show()
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
        print(f"D: {D}\t sigma: {sigma}\t meu: {meu}\t X_0:{x_0}\t theta_s: {np.degrees(theta_s)}\t theta_e: {np.degrees(theta_e)}")
        plt.plot(x_values, y_values, label="real drawing")
        X[~np.isnan(new_X)] += new_X[~np.isnan(new_X)]
        Y[~np.isnan(new_Y)] += new_Y[~np.isnan(new_Y)]
        plt.plot(X, Y, label="generated drawing")
        plt.title("real drawing vs generated")
        plt.xlabel("x-co")
        plt.ylabel("y-co")
        plt.legend()
        plt.show()
