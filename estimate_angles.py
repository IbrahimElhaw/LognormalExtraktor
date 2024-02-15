# this file is the main file. The functions in this file aim to estimate the start and end angles, the main method
# is used to test the actual advance in the implementing. The file uses all other files.
import math

import matplotlib.pyplot as plt
import numpy as np

import draw_movment
import redraw_profile
import utilities
from ranged_infelction_points import extract_parameters_first_mode, generate_curve_from_parameters,\
    extract_parameters_second_mode


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


def estimate_theta_SE(x_values, y_values, D, sigma, characteristic_points, i=-1):
    _, xinf1, x3, xinf2, _ = characteristic_points
    d1, d2, d3, d4, d5 = [calculate_distance_cp(D, sigma, i) for i in range(1,6)]

    # plt.scatter(x_values[characteristic_points], y_values[characteristic_points], label=f"char. points: {i}")

    dy = np.gradient(y_values)
    dx = np.gradient(x_values)
    anlges_list = np.arctan2(dy, dx)
    angle_t2 = anlges_list[xinf1]
    angle_t3 = anlges_list[x3]
    angle_t4 = anlges_list[xinf2]

    dAngle = angle_t4 - angle_t2

    # dieser Linie ist von dem Projekt SynSig2Vec inspiriert
    dAngle = math.copysign(2 * math.pi - abs(dAngle), -dAngle) if abs(dAngle) > 3./2 * math.pi else dAngle

    # dAngle2 = angle_t3 - angle_t2
    # dAngle2 = math.copysign(2 * math.pi - abs(dAngle2), -dAngle2) if abs(dAngle2) > 3./2 * math.pi else dAngle2
    #
    # dAngle3 = angle_t4 - angle_t3
    # dAngle3 = math.copysign(2 * math.pi - abs(dAngle3), -dAngle3) if abs(dAngle3) > 3./2 * math.pi else dAngle3

    dDistanz =  d4 - d2
    # dDistanz2 = d3 - d2
    # dDistanz3 = d4 - d3

    dA_dD =  dAngle / dDistanz
    # dA_dD2 = dAngle2 /dDistanz2
    # dA_dD3 = dAngle3 /dDistanz3

    # abs1 = abs(dA_dD-dA_dD3)
    # abs2 = abs(dA_dD-dA_dD2)
    # abs3 = abs(dA_dD2-dA_dD3)
    # if np.min(([abs1, abs2, abs3])) == abs1:
    #     firstdA_dD, seconddA_dD = dA_dD, dA_dD3
    # elif np.min(([abs1, abs2, abs3])) == abs2:
    #     firstdA_dD, seconddA_dD = dA_dD, dA_dD2
    # else:
    #     firstdA_dD, seconddA_dD = dA_dD2, dA_dD3

    # dA_dD = np.mean([firstdA_dD, seconddA_dD])
    theta_s = angle_t3 - dA_dD * (d3 - d1)
    theta_e = angle_t3 + dA_dD * (d5 - d3)

    return theta_s, theta_e


# param are the indexes of the 5 char points, the middle 3 are more reliable.
def left_before_right(x_values_p, param, i=-1):

    point1 = x_values_p[param[1]]  # xinf1
    point2 = x_values_p[param[2]]  # xm
    point3 = x_values_p[param[3]]  # xinf2

    condition1 = point2 > point1
    condition2 = point3 > point2
    condition3 = point3 > point1

    n_con_satisfied = sum([condition1, condition2, condition3])
    if n_con_satisfied == 3:
        print(True)
        return True
    if n_con_satisfied == 0:
        print(False)
        return False
    return None


def estimate_angles(X_values, Y_values, strokes_list, time):
    angles=[]
    # plt.plot(x_values, y_values, color="black", label="Bewegung")
    for i, stroke in enumerate(strokes_list):
        D, sigma, meu, t0 = stroke
        characteristic_points = utilities.find_char_points_lognormal(time, sigma, meu, t0)
        theta_s, theta_e = estimate_theta_SE(X_values, Y_values, D, sigma, characteristic_points, i)
        angles.append((theta_s, theta_e))
    # plt.legend()
    # plt.show()
    return angles


def redraw(smoothed_v, time, strokes_list, angles_list, regenerated_curve):
    acX = np.zeros_like(time)
    acY = np.zeros_like(time)

    local_max, local_min = utilities.get_extrems(smoothed_v)
    local_min, local_max = utilities.correct_local_extrems(local_min, local_max, time, smoothed_v)
    local_min = np.insert(local_min, 0, 0)

    for stroke, angle, i in zip(strokes_list, angles_list, range(len(strokes_list))):
        D1, sigma, meu, t0 = stroke
        theta_s, theta_e = angle

        vx_selected = time[local_min[i]:local_min[i + 1]]
        vy_selected = regenerated_curve[local_min[i]:local_min[i + 1]]
        area_under_curve = np.trapz(vy_selected, vx_selected)
        D2 = area_under_curve
        # TODO: choose between D1 and D2 and the mean
        D = np.mean([D1, D2])
        # D = D2
        # D = D1
        # print((D, D2))

        X, Y = redraw_profile.draw_stroke_original(D, theta_s, theta_e, time, t0, meu, sigma)
        condition = ~np.isnan(X) & ~np.isnan(Y)
        acX[condition] += X[condition]
        acY[condition] += Y[condition]

    acX, acY = utilities.normalize(acX, acY)
    return acX, acY


if __name__ == '__main__':
    # x_values, y_values, timestamps_arr, smoothed_velocity, velocity = utilities.load_input("last_run.npz")
    x_values, y_values, timestamps_arr, smoothed_velocity, velocity = draw_movment.get_preprocessed_input()
    # utilities.save_input(x_values, y_values, timestamps_arr, smoothed_velocity, velocity, filename="last_run.npz")
    strokes1 = extract_parameters_first_mode(timestamps_arr, smoothed_velocity.copy())  # strokes in form (D, sigma, meu, t0)^n
    regenerated_log_curve = generate_curve_from_parameters(strokes1, timestamps_arr)
    plt.plot(timestamps_arr, regenerated_log_curve)
    plt.plot(timestamps_arr, smoothed_velocity)
    plt.show()
    angles1 = estimate_angles(x_values, y_values, strokes1, timestamps_arr)
    acX, acY = redraw(smoothed_velocity, timestamps_arr, strokes1, angles1, regenerated_log_curve)

    plt.plot(x_values, y_values, color="red", label="original")
    plt.figure(2)
    plt.plot(acX, acY, color="black", label="regeneratd")
    plt.title("final result")
    plt.figure(3)
    plt.plot(acX, acY, color="black", label="regeneratd")
    plt.plot(x_values, y_values, color="red", label="original")
    plt.legend()
    plt.show()

    # difference = smoothed_velocity - regenerated_log_curve
    # strokes2 = extract_parameters_second_mode(timestamps_arr, difference)
    # regenerated_log_curve2 = generate_curve_from_parameters(strokes2, timestamps_arr)
    # plt.plot(timestamps_arr, regenerated_log_curve, color="cyan", label="first one")
    # plt.plot(timestamps_arr, regenerated_log_curve+regenerated_log_curve2, label="addition",  color="yellow")
    # plt.plot(timestamps_arr, smoothed_velocity, color="black", label="original")
    # plt.legend()
    # plt.show()
    # angles2 = estimate_angles(x_values, y_values, strokes2, timestamps_arr)
    # plt.plot(x_values, y_values, color="black", label="original")
    # plt.figure(2)
    # plt.plot(acX, acY, color="red", label="before")
    # MSE1 = utilities.calculate_MSE(x_values, acX)+ utilities.calculate_MSE(y_values, acY)
    # for stroke, angles in zip(strokes2, angles2):
    #     d, s, m, t = stroke
    #     theta_s, theta_e = angles
    #     X, Y = redraw_profile.draw_stroke_original(d, theta_s, theta_e, timestamps_arr, t, m, s)
    #     acX+=X
    #     acY+=Y
    # plt.figure(3)
    # plt.plot(acX, acY, color="red", label="after")
    # plt.legend()
    # plt.show()
    # MSE2 = utilities.calculate_MSE(x_values, acX)+ utilities.calculate_MSE(y_values, acY)
    # print(MSE1)
    # print(MSE2)
