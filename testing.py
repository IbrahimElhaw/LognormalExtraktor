# this file is the main file. The functions in this file aim to estimate the start and end angles, the main method
# is used to test the actual advance in the implementing. The file uses all other files.
import math
# TODO, try scaler loop for the parameter D
# TODO, try velocity calculator AUC

import matplotlib.pyplot as plt
import numpy as np

import input
import openfiles
import utilities
from draw_profile import draw_stroke_updated, draw_stroke_original
from openfiles import calculate_velocity, open_file_children, extra_smooth
from ranged_infelction_points import represent_curve, generate_curve_from_parameters


# ref original on-line signature, func(33), per point
def calculate_distance_pp(t, D, sigma,  meu, t0):
    t = t - t0
    t[t < 0] = 0
    return D/2 * (1+np.math.erf(np.log((t-t0)-meu)/(sigma*2**0.5)))


# ref original on-line signature, func(34), characteristic point
def calculate_distance_cp(D, sigma, point):
    if point==1:
        return 0
    if point==5:
        return D
    if 0 < point < 5:
        point -= 2
        sig_sq = sigma**2
        a = [(3 / 2) * sig_sq + sigma * np.sqrt(((sig_sq) / 4) + 1),
             sig_sq,
             (3 / 2) * sig_sq - sigma * np.sqrt(((sig_sq) / 4) + 1)]

        return D/2 * (1+np.math.erf(-a[point]/(sigma*(2**0.5))))
    return None

# the angle at this point of time, the angular position, ref iDeLog, func (3)
def calculate_phi(t, D, sigma, meu, t0, theta_s, theta_e):
    phi = calculate_distance_pp(t, D, t0, meu, sigma) / D * (theta_e - theta_s) + theta_s
    return phi


def XY_velocity_combination(velocity, t, D, sigma , mu, t0 , theta_s, theta_e):
    if abs(theta_s - theta_e) <0.001:
        Vx= velocity*np.cos(theta_s)
        Vy= velocity*np.sin(theta_s)
    else:
        theta = calculate_phi(t, D, sigma, mu, t0, theta_s, theta_e)
        Vx= velocity*np.cos(theta)
        Vy= velocity*np.sin(theta)
    return Vx, Vy


# TODO: I have to look again at it, ref iDeLog func(4)
def estimateLxy(t, D, sigma, meu, t0, theta_s, theta_e, shift=(0, 0), dt=0.01):
    dtheta= theta_e - theta_s
    if dtheta <0.001:
        distance = calculate_distance_pp(t, D, sigma, meu, t0)
        average_velocity_x = distance * np.cos(theta_s) / dt + shift[0]
        average_velocity_y = distance * np.sin(theta_s) / dt + shift[0]
    else:
        theta = calculate_phi(t, D, sigma, meu, t0, theta_s, theta_e)
        dD = D/dtheta
        average_velocity_x = dD * (np.sin(theta) - np.sin(theta_s)) / dt + shift[0]
        average_velocity_y = dD * (np   .cos(theta_s) - np.cos(theta)) / dt + shift[1]
    return average_velocity_x, average_velocity_y


# characteristic_points are indexes
def estimate_theta_SE(x_values, y_values, D, sigma, characteristic_points):
    _, xinf1, x3, xinf2, _ = characteristic_points

    d1 = calculate_distance_cp(D, sigma, 1)
    d2 = calculate_distance_cp(D, sigma, 2)
    d3 = calculate_distance_cp(D, sigma, 3)
    d4 = calculate_distance_cp(D, sigma, 4)
    d5 = calculate_distance_cp(D, sigma, 5)

    angle_t2 = np.arctan(np.gradient(y_values, x_values)[xinf1])
    angle_t3 = np.arctan(np.gradient(y_values, x_values)[x3])
    angle_t4 = np.arctan(np.gradient(y_values, x_values)[xinf2])

    # degt2 = np.degrees(angle_t2)
    # while degt2<0:
    #     degt2+=360
    # degt3 = np.degrees(angle_t3)
    # while degt3<0:
    #     degt3+=360
    # degt4 = np.degrees(angle_t4)
    # while degt4 < 0:
    #     degt4 += 360
    # plt.plot(x_values, y_values, zorder=0, marker="o")
    # plt.scatter(x_values[[xinf1, x3, xinf2]], y_values[[xinf1, x3, xinf2]], color="red", zorder=1)
    # plt.show()
    dAngle = angle_t4 - angle_t2
    # TODO this line should be tried
    # dAngle = math.copysign(2 * math.pi - abs(dAngle), -dAngle) if abs(dAngle) > 3./2 * math.pi else dAngle
    dAngle = dAngle / (d4 - d2)

    theta_s = angle_t3 - dAngle * (d3 - d1)
    theta_e = angle_t3 + dAngle * (d5 - d3)
    return theta_s, theta_e


# param are the indexes of the 5 char points, the middle 3 are more reliable.
def left_before_right(x_values, param):
    point1 = x_values[param[1]]
    point2 = x_values[param[2]]
    point3 = x_values[param[3]]

    condition1 = point2 > point1
    condition2 = point3 > point2
    condition3 = point3 > point1

    n_con_satisfied = sum([condition1, condition2, condition3])
    print("sum of left before right =", n_con_satisfied, " and the result is ", n_con_satisfied >= 2)
    return n_con_satisfied >= 2



if __name__ == '__main__':
    # x_values, y_values, timestamps_arr, smoothed_velocity, velocity = openfiles.open_file_unistroke("DB\\1 unistrokes\\s01 (pilot)\\fast\\left_sq_bracket03.xml")  #v02
    # x_values, y_values, timestamps_arr, smoothed_velocity, velocity = openfiles.open_file_signature("DB\\Task1\\U1S1.TXT")
    x_values, y_values, timestamps_arr, smoothed_velocity, velocity = input.get_preprocessed_input()

    plt.plot(timestamps_arr, smoothed_velocity, label="velocity")
    plt.plot(timestamps_arr, velocity, label="veolcity before smoothing")
    plt.title("smoothed velocity")
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.legend()
    plt.figure(2)
    plt.plot(x_values, y_values, marker="o")
    plt.show()

    strokes = represent_curve(timestamps_arr, smoothed_velocity.copy())  # strokes in form (D, sigma, meu, t0)^n
    regenerated_curve = generate_curve_from_parameters(strokes, timestamps_arr)
    plt.title("the whole regenerated curve")
    plt.plot(timestamps_arr, velocity, label="original", color="cyan")
    plt.plot(timestamps_arr, regenerated_curve, label="regenerated", marker="o")
    plt.plot(timestamps_arr, smoothed_velocity, label="smoothed", color="purple")
    plt.legend()
    plt.show()

    print(strokes)
    angles = []
    for stroke in strokes:
        D, sigma, meu, t0 = stroke
        actual_curve = utilities.generate_lognormal_curve(D, sigma, meu, t0, timestamps_arr[0], timestamps_arr[-1], len(timestamps_arr))
        characteristic_points = utilities.find_char_points_lognormal(timestamps_arr, sigma, meu, t0)
        # plt.plot(timestamps_arr, actual_curve, zorder=0)
        # plt.scatter(timestamps_arr[characteristic_points], actual_curve[characteristic_points], color="red", zorder=1)
        # plt.figure(2)
        theta_s, theta_e = estimate_theta_SE(x_values, y_values, D, sigma, characteristic_points)
        angles.append((theta_s, theta_e))
        print(np.degrees(theta_s), np.degrees(theta_e))
        # plt.plot(x_values, y_values)
        # plt.scatter(x_values[characteristic_points], y_values[characteristic_points], color="red")
        # plt.show()

    acX = np.zeros_like(timestamps_arr)
    acY = np.zeros_like(timestamps_arr)

    local_max, local_min = utilities.get_extrems(smoothed_velocity)
    local_min, local_max = utilities.correct_local_extrems(local_min, local_max, timestamps_arr, smoothed_velocity)
    local_min = np.insert(local_min, 0, 0)
    plt.plot(timestamps_arr, smoothed_velocity)
    plt.scatter(timestamps_arr[local_min], smoothed_velocity[local_min])
    plt.scatter(timestamps_arr[local_max], smoothed_velocity[local_max])
    plt.show()
    for stroke, angle, i in zip(strokes, angles, range(len(strokes))):
        D1, sigma, meu, t0 = stroke

        vx_selected = timestamps_arr[local_min[i]:local_min[i+1]]
        vy_selected = regenerated_curve[local_min[i]:local_min[i+1]]
        area_under_curve = np.trapz(vy_selected, vx_selected)
        D2 = area_under_curve
        # TODO: choose between D1 and D2 and the mean
        # D = np.mean([D1, D2])
        # D = D2
        D = D1
        # print((D, D2))

        theta_s, theta_e = angle
        l_befor_r = left_before_right(x_values, utilities.find_char_points_lognormal(timestamps_arr, sigma, meu, t0))
        X, Y = draw_stroke_updated(D, theta_s, theta_e, timestamps_arr, t0, meu, sigma, l_befor_r)
        # X, Y = draw_stroke_original(D, theta_s, theta_e, timestamps_arr, t0, meu, sigma)
        # TODO: centering
        # X -= np.min(X)
        # Y -= np.min(Y)

        acX[~np.isnan(X)] += X[~np.isnan(X)]
        acY[~np.isnan(Y)] += Y[~np.isnan(Y)]
        plt.plot(X, Y)
        plt.plot(x_values, y_values)
        plt.plot(acX, acY, color="pink")
        plt.show()

    acX, acY = openfiles.normalize(acX, acY)
    plt.plot(x_values, y_values, color="red", label="original")
    plt.figure(3)
    plt.plot(acX, acY, color="black", label="regeneratd")
    plt.title("final result")
    plt.legend()
    plt.show()

    # # acX_r = np.array([])
    # # acY_r = np.array([])
    # acX_r = np.zeros_like(timestamps_arr)
    # acY_r = np.zeros_like(timestamps_arr)
    # for stroke, angle, i in zip(strokes, angles, range(len(strokes))):
    #     D, sigma, meu, t0 = stroke
    #
    #     vx_selected = timestamps_arr[local_min[i]:local_min[i+1]]
    #     vy_selected = regenerated_curve[local_min[i]:local_min[i+1]]
    #     area_under_curve = np.trapz(vy_selected, vx_selected)
    #     D2 = area_under_curve
    #     D = np.mean([D, D2])
    #     print((D, D2))
    #
    #     theta_s, theta_e = angle
    #     X, Y = draw_stroke(D, theta_s, theta_e, timestamps_arr, t0, meu, sigma)
    #     acX_r[~np.isnan(X)] += X[~np.isnan(X)]
    #     acY_r[~np.isnan(Y)] += Y[~np.isnan(Y)]
    #     plt.plot(X, Y)
    #     plt.scatter(X[~np.isnan(X)][0], Y[~np.isnan(Y)][0])
    #     plt.scatter(X[~np.isnan(X)][-1], Y[~np.isnan(Y)][-1], color="red")
    #     plt.plot(x_values, y_values)
    #     plt.plot(acX_r, acY_r, color="red")
    #     plt.show()
    # acX_r -= np.min(acX_r)
    # acY_r -= np.min(acY_r)
    # plt.plot(x_values, y_values, color="red", label="original")
    # plt.plot(acX_r, acY_r, color="black", label="regeneratd")
    # plt.title("final result")
    # plt.legend()
    # plt.show()



