# this file is the main file. The functions in this file aim to estimate the start and end angles, the main method
# is used to test the actual advance in the implementing. The file uses all other files.
import math
# TODO, try scaler loop for the parameter D
# TODO, try velocity calculator AUC

import matplotlib.pyplot as plt
import numpy as np

import draw_movment
import openfiles
import utilities
from redraw_profile import draw_stroke_updated
from ranged_infelction_points import represent_curve, generate_curve_from_parameters


# ref original on-line signature, func(33), per point
def calculate_distance_pp(t, D, sigma, meu, t0):
    t = t - t0
    t[t < 0] = 0.00001
    numerator = np.log(t) - meu
    denominator = sigma * (2 ** 0.5)
    return D / 2 * (1 + np.math.erf(numerator / denominator))


# ref original on-line signature, func(34), characteristic point
def calculate_distance_cp(D, sigma, point):
    if point == 1:
        return 0
    if point == 5:
        return D
    if 0 < point < 5:
        point -= 2
        sig_sq = sigma ** 2
        a = [(3 / 2) * sig_sq + sigma * np.sqrt(((sig_sq) / 4) + 1),
             sig_sq,
             (3 / 2) * sig_sq - sigma * np.sqrt(((sig_sq) / 4) + 1)]

        return D / 2 * (1 + np.math.erf(-a[point] / (sigma * (2 ** 0.5))))
    return None


# characteristic_points are indexes
def correct_direction(angle_t2, angle_t3, angle_t4, dAngle):
    while angle_t2<0:
        angle_t2+=np.pi
    while angle_t3<0:
        angle_t3+=np.pi
    while angle_t4 < 0:
        angle_t4 += np.pi

    if angle_t4 >= angle_t3:
        return dAngle if dAngle >= 0 else -dAngle
    else:
        return dAngle if dAngle <= 0 else -dAngle


def estimate_theta_SE(x_values, y_values, D, sigma, meu, t0, characteristic_points, i=1):
    _, xinf1, x3, xinf2, _ = characteristic_points

    d1 = calculate_distance_cp(D, sigma, 1)
    d2 = calculate_distance_cp(D, sigma, 2)
    d3 = calculate_distance_cp(D, sigma, 3)
    d4 = calculate_distance_cp(D, sigma, 4)
    d5 = calculate_distance_cp(D, sigma, 5)

    current_curve = utilities.generate_lognormal_curve(D, sigma, meu, t0, timestamps_arr[0], timestamps_arr[-1],
                                                       len(timestamps_arr))
    plt.plot(timestamps_arr, current_curve, marker="s", zorder=0)
    plt.scatter(timestamps_arr[characteristic_points], current_curve[characteristic_points], zorder=1, color="red")
    plt.figure(2)
    plt.plot(x_values, y_values)
    plt.scatter(x_values[characteristic_points][1], y_values[characteristic_points][1], color="green")
    plt.scatter(x_values[characteristic_points][2], y_values[characteristic_points][2], color="yellow")
    plt.scatter(x_values[characteristic_points][3], y_values[characteristic_points][3], color="red")
    plt.show()

    angle_t2 = np.arctan(np.gradient(y_values, x_values)[xinf1])
    angle_t3 = np.arctan(np.gradient(y_values, x_values)[x3])
    angle_t4 = np.arctan(np.gradient(y_values, x_values)[xinf2])


    angle2_g = np.degrees(angle_t2)
    angle3_g = np.degrees(angle_t3)
    angle4_g = np.degrees(angle_t4)

    # TODO: Iimportant: find a condition
    dAngle = angle_t4 - angle_t2
    dAngle_g = np.degrees(dAngle)

    if abs(dAngle) > np.pi/2.2 :
        if angle_t4<0:
            angle_t4 = np.pi+angle_t4
        else:
            angle_t4 = -(np.pi-angle_t4)
        angle4_g = np.degrees(angle_t4)
        dAngle = angle_t4 - angle_t2
        print("flipped")
        dAngle = correct_direction(angle_t2, angle_t3, angle_t4, dAngle)

    dDistanz = d4 - d2
    dAngle_g = np.degrees(dAngle)

    # TODO this line should be tried
    # dAngle = math.copysign(2 * math.pi - abs(dAngle), -dAngle) if abs(dAngle) > 3./2 * math.pi else dAngle
    dA_dD = dAngle / dDistanz
    dA_dD_G = np.degrees(dA_dD)

    theta_s = angle_t3 - dA_dD * (d3 - d1)
    theta_e = angle_t3 + dA_dD * (d5 - d3)
    return theta_s, theta_e


# param are the indexes of the 5 char points, the middle 3 are more reliable.
def left_before_right(x_values_p, param):
    point1 = x_values_p[param[1]]
    point2 = x_values_p[param[2]]
    point3 = x_values_p[param[3]]

    condition1 = point2 > point1
    condition2 = point3 > point2
    condition3 = point3 > point1

    n_con_satisfied = sum([condition1, condition2, condition3])
    # TODO: important, find a better condition
    print("sum of left before right =", n_con_satisfied, " and the result is ", n_con_satisfied >= 3)
    return n_con_satisfied >= 2


def save_input(x_values, y_values, timestamps_arr, smoothed_velocity, velocity, filename="data.npz"):
    np.savez(filename, x_values=x_values, y_values=y_values, timestamps_arr=timestamps_arr,
             smoothed_velocity=smoothed_velocity, velocity=velocity)


def load_input(filename="data.npz"):
    data = np.load(filename)
    return data['x_values'], data['y_values'], data['timestamps_arr'], data['smoothed_velocity'], data['velocity']


if __name__ == '__main__':
    # x_values, y_values, timestamps_arr, smoothed_velocity, velocity = openfiles.open_file_unistroke("DB\\1 unistrokes\\s01 (pilot)\\fast\\left_sq_bracket03.xml")  #v02
    # x_values, y_values, timestamps_arr, smoothed_velocity, velocity = openfiles.open_file_signature("DB\\Task1\\U1S1.TXT")
    x_values, y_values, timestamps_arr, smoothed_velocity, velocity = load_input("solved_bug, angle_t4 = np.pi+angle_t4.npz")
    # x_values, y_values, timestamps_arr, smoothed_velocity, velocity = draw_movment.get_preprocessed_input()
    # save_input(x_values, y_values, timestamps_arr, smoothed_velocity, velocity, filename="new_try.npz")

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
    for stroke, i in zip(strokes, range(len(strokes))):
        D, sigma, meu, t0 = stroke
        characteristic_points = utilities.find_char_points_lognormal(timestamps_arr, sigma, meu, t0)
        current_curve = utilities.generate_lognormal_curve(D, sigma, meu, t0, timestamps_arr[0], timestamps_arr[-1], len(timestamps_arr))
        theta_s, theta_e = estimate_theta_SE(x_values, y_values, D, sigma, meu, t0, characteristic_points, i)
        angles.append((theta_s, theta_e))
        print(np.degrees(theta_s), np.degrees(theta_e))

    acX = np.zeros_like(timestamps_arr)
    acY = np.zeros_like(timestamps_arr)

    local_max, local_min = utilities.get_extrems(smoothed_velocity)
    local_min, local_max = utilities.correct_local_extrems(local_min, local_max, timestamps_arr, smoothed_velocity)
    local_min = np.insert(local_min, 0, 0)

    for stroke, angle, i in zip(strokes, angles, range(len(strokes))):
        D1, sigma, meu, t0 = stroke

        vx_selected = timestamps_arr[local_min[i]:local_min[i + 1]]
        vy_selected = regenerated_curve[local_min[i]:local_min[i + 1]]
        area_under_curve = np.trapz(vy_selected, vx_selected)
        D2 = area_under_curve
        # TODO: choose between D1 and D2 and the mean
        D = np.mean([D1, D2])
        # D = D2
        # D = D1
        # print((D, D2))

        theta_s, theta_e = angle
        # if i == 1:
        #     theta_s+=np.pi/2
        #     theta_e-=np.pi/2
        #     print("neuer Winkel S: ", np.degrees(theta_s))
        #     print("neuer Winkel E:", np.degrees(theta_e))

        l_before_r = left_before_right(x_values, utilities.find_char_points_lognormal(timestamps_arr, sigma, meu, t0))
        X, Y = draw_stroke_updated(D, theta_s, theta_e, timestamps_arr, t0, meu, sigma, l_before_r)

        condition = ~np.isnan(X) & ~np.isnan(Y)
        acX[condition] += X[condition]
        acY[condition] += Y[condition]
        plt.plot(X, Y)
        try:
            plt.scatter(X[~np.isnan(X)][0], Y[~np.isnan(Y)][0])
        except IndexError:
            pass
        plt.plot(x_values, y_values)
        plt.plot(acX, acY, color="pink")
        plt.show()

    acX, acY = openfiles.normalize(acX, acY)
    plt.plot(x_values, y_values, color="red", label="original")
    plt.figure(2)
    plt.plot(acX, acY, color="black", label="regeneratd")
    plt.title("final result")
    plt.figure(3)
    plt.plot(acX, acY, color="black", label="regeneratd")
    plt.plot(x_values, y_values, color="red", label="original")
    plt.legend()
    plt.show()
