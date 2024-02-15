# this file is used to estimate all parameters of sigma lognormal profils. the main method
# is used to test the file on a perfect profile after changes. it should show 2 identical graphs
# if the changes do not negatively affect the algorithm
import math

import matplotlib.pyplot as plt
import numpy as np

import draw_movment
from utilities import load_input, interpolate
from utilities import generate_lognormal_curve, calculate_MSE, get_local_max, \
    get_local_min, correct_local_extrems, generate_4_lognormal_curves

# parameter ref: https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# table 2

sigma_accuracy = 30  # how many points are used in the estimated range of characteristic points
# the more accuracy the more calculations. O(n^2). 25 seems to be optimal value


# the next 4 values are the proportion of the  start or end point of the range where the inflection points are likely
# be. first range is for the first inflection point. the second is for the second
# should be 0.43, 0.60, 0.60, 0.75 and sigma_accuracy=25, but I expanded the range to get more confident result
first_range_start = 0.45
first_range_end = 0.63
second_range_start = 0.60
second_range_end = 0.75


def calculate_meu(t1, t2, a1, a2):
    devider = np.exp(-a1) - np.exp(-a2)
    if devider==0:
        devider+=0.01
    devider = math.copysign(devider, (t1-t2))
    mu = np.log((t1 - t2) / devider)
    return mu


def calculate_t_0(t, a, meu):
    t0 = t - np.exp(meu - a)
    return t0


def calculate_D(a, meu, sigma, v1):
    if sigma==0:
        sigma +=0.01
    D = np.sqrt(np.pi*2) * v1 * sigma * np.exp(meu-a+a**2/(2*sigma**2))
    return D


# after determining the range of inflection points in the y coordination, the range is determined in x coordination.
# the function returns 2 array, each for one inflection point. the array tells, in which x coor range the inflection
# point can be
def corresponding_x_values(x_values, velocity_profile, v_3, x_3, sigma_accuracy):
    condition = (velocity_profile < (first_range_start * v_3)) & (x_values < x_3)
    corresponding_x_value1 = x_values[condition][-1]
    condition = (velocity_profile < (first_range_end * v_3)) & (x_values < x_3)
    corresponding_x_value2 = x_values[condition][-1]
    condition = (velocity_profile < (second_range_start * v_3)) & (x_values > x_3)
    if np.any(condition):
        corresponding_x_value3 = x_values[condition][0]
    else:
        print("in else getreten")
        points_after_x3 = x_values[x_values > x_3]
        corresponding_x_value3 = points_after_x3[int(len(points_after_x3)/2)]
    condition = (velocity_profile < (second_range_end * v_3)) & (x_values > x_3)
    if np.any(condition):
        corresponding_x_value4 = x_values[condition][0]
    else:
        print("in else getreten")
        points_after_x4 = x_values[x_values > x_3]
        corresponding_x_value4 = points_after_x4[int(len(points_after_x4)*4 / 5)]
    x_values_v2_inf1 = np.linspace(corresponding_x_value1, corresponding_x_value2, sigma_accuracy)
    x_values_v4_inf2 = np.linspace(corresponding_x_value3, corresponding_x_value4, sigma_accuracy)
    return x_values_v2_inf1, x_values_v4_inf2


# calulate sigma parameter
# ref: https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# table 2
def calculate_sigmas(first_inflection, local_max, second_inflection):
    sigmas = []
    # 1
    beta_23 = first_inflection / local_max
    sigma_quadratic = -2 - 2 * np.log(beta_23) - (1 / (2 * np.log(beta_23)))
    sigma = np.sqrt(sigma_quadratic)
    sigmas.append(sigma)

    # 2
    if second_inflection == first_inflection:
        first_inflection += 0.01
    beta_42 = second_inflection / first_inflection
    sigma_quadratic = -2 + 2 * np.sqrt(1 + (np.log(beta_42)) ** 2)
    sigma = np.sqrt(sigma_quadratic)
    sigmas.append(sigma)

    # 3
    beta_43 = second_inflection / local_max
    sigma_quadratic = -2 - 2 * np.log(beta_43) - (1 / (2 * np.log(beta_43)))
    sigma = np.sqrt(sigma_quadratic)
    sigmas.append(sigma)

    return np.array(sigmas)


# after calculating D, the small errors can be corrected, for that the maxima is used to increase, or decrease D,
# until the regenrated maxima equals the orgiginal maxima
def correct_D(bestfit, x_values, velocity):
    while np.max(velocity) < np.max(
            generate_lognormal_curve(bestfit[0], bestfit[1], bestfit[2], bestfit[3], x_values[0], x_values[-1],
                                     len(x_values))):
        bestfit = (bestfit[0] - 0.1, bestfit[1], bestfit[2], bestfit[3])
    # while np.max(velocity) > np.max(
    #         generate_lognormal_curve(bestfit[0], bestfit[1], bestfit[2], bestfit[3], x_values[0], x_values[-1],
    #                                  len(x_values))):
    #     bestfit = (bestfit[0] + 0.1, bestfit[1], bestfit[2], bestfit[3])
    return bestfit




# returns array of strokes parameter in form of (D, sigma, meu, t_0)^n
# for that every local maximum represent on stroke
# the outer loop is used to estimate the parameters of one stroke then it itirates to the next.

def correct_t0(bestfit, x_values, trimmed_velocity):
    D, sigma, meu, t0 = bestfit
    new_t0 = t0
    gen = generate_lognormal_curve(D, sigma, meu, t0, x_values[0], x_values[-1], len(x_values))
    MSE = calculate_MSE(trimmed_velocity, gen)
    gen_pos = generate_lognormal_curve(D, sigma, meu, new_t0+0.01, x_values[0], x_values[-1], len(x_values))
    MSE_pos = calculate_MSE(trimmed_velocity, gen_pos)
    sign = 1 if MSE_pos < MSE else -1
    while True:
        gen2 = generate_lognormal_curve(D, sigma, meu, new_t0+ sign * 0.01, x_values[0], x_values[-1], len(x_values))
        gen3 = generate_lognormal_curve(D, sigma, meu, new_t0+ sign * 0.02, x_values[0], x_values[-1], len(x_values))
        MSE2 = calculate_MSE(trimmed_velocity, gen2)
        MSE3 = calculate_MSE(trimmed_velocity, gen3)
        condition = (MSE2 < MSE) or (MSE3 < MSE)
        if ~condition:
            break
        new_t0 += (0.01*sign)
        MSE = MSE2
    bestfit2 = [D, sigma, meu, new_t0]
    return bestfit2


def extract_parameters_first_mode(x_values, y_values):
    strokes = []
    local_maximum_indices = get_local_max(y_values)
    local_minimum_indices = get_local_min(y_values)
    local_minimum_indices, local_maximum_indices = correct_local_extrems(local_minimum_indices,
                                                                                   local_maximum_indices,
                                                                                   x_values, y_values)
    # plt.plot(x_values,y_values, label="Geschwindigkeit", color="black")
    # plt.scatter(x_values[local_maximum_indices],y_values[local_maximum_indices], label="local maxima", color="red")
    # plt.legend()
    # plt.show()

    for i in range(len(local_maximum_indices)):
        trimmed_velocity = y_values.copy()

        used_local_max_index = local_maximum_indices[i]
        try:
            used_local_min_index = local_minimum_indices[local_minimum_indices > used_local_max_index][0]
        except IndexError:
            used_local_min_index = -1

        # the values after used local min are trimmed to not calcculate the irrelevant values in the MSE (here one
        # stroke is researched).
        # plt.plot(x_values, y_values, label="Geschwindigkeit", color="black")
        # plt.scatter(x_values[local_maximum_indices], y_values[[local_maximum_indices]], label="locales Maximum", color= "purple")
        #
        trimmed_velocity[x_values > x_values[used_local_min_index]] = 0
        # plt.plot(x_values, trimmed_velocity)
        # plt.show()
        v_3 = y_values[used_local_max_index]
        t_3 = x_values[used_local_max_index]

        # the range of y coordinations of inflection pints is calculated, n the range of x coordinations
        v2_inf1_range = np.linspace(first_range_start * v_3, first_range_end * v_3, sigma_accuracy)
        v4_inf2_range = np.linspace(second_range_start * v_3, second_range_end * v_3, sigma_accuracy)
        x_values_v2_inf1, x_values_v4_inf2 = corresponding_x_values(x_values, y_values, v_3, t_3,
                                                                    sigma_accuracy)
        # plt.plot(x_values, y_values, color="black", label="Geschwindigkeit")
        # plt.scatter(x_values[used_local_max_index], y_values[used_local_max_index], color="purple", label="lokales Maximum")
        # plt.plot(x_values_v2_inf1, v2_inf1_range, label="die Bereiche der Wendepunkts", color="red")
        # plt.plot(x_values_v4_inf2, v4_inf2_range, color="red")
        # plt.legend()
        # plt.show()


        params = []

        # choose every pair of points in the estimated range to calculate sigma
        for v2_inf1, x2_inf1 in zip(v2_inf1_range, x_values_v2_inf1):
            for v4_inf2, x4_inf2 in zip(v4_inf2_range, x_values_v4_inf2):
                three_char_time = [x2_inf1, t_3, x4_inf2]
                three_char_velocity = [v2_inf1, v_3, v4_inf2]
                sigmas = calculate_sigmas(v2_inf1, v_3, v4_inf2)
                # sigmas = calculate_sigmas(x2_inf1, t_3, x4_inf2)
                for sigma in sigmas:
                    sig_sq = sigma ** 2
                    a = [(3 / 2) * sig_sq + sigma * np.sqrt(((sig_sq) / 4) + 1),
                         sig_sq,
                         (3 / 2) * sig_sq - sigma * np.sqrt(((sig_sq) / 4) + 1)]
                    for alpha, beta in [[0, 1], [1, 2], [0, 2]]:
                        meu = calculate_meu(three_char_time[alpha], three_char_time[beta], a[alpha], a[beta])
                        t_0 = calculate_t_0(three_char_time[alpha], a[alpha], meu)
                        D = calculate_D(a[alpha], meu, sigma, three_char_velocity[alpha])
                        params.append([D, sigma, meu, t_0])
        bestMSE = float("inf")
        bestfit = None
        best_generate = np.zeros_like(y_values)
        for param in params:
            D, sigma, meu, t_0,  = param
            if sigma==0:
                continue
            generated_profile = generate_lognormal_curve(D, sigma, meu, t_0, x_values[0], x_values[-1], len(x_values))
            MSE = calculate_MSE(generated_profile[generated_profile>0.01*np.max(generated_profile)], trimmed_velocity[generated_profile>0.01*np.max(generated_profile)])
            if MSE < bestMSE:
                best_generate = generated_profile
                bestMSE = MSE
                bestfit = (D, sigma, meu, t_0)
        if bestfit is None:
            bestfit=(0.1, 0.1, 0.1, 0.1)
        # bestfit = correct_D(bestfit, x_values, trimmed_velocity)
        bestfit = correct_t0(bestfit, x_values, trimmed_velocity)
        strokes.append(bestfit)
        y_values -= best_generate
        y_values[x_values<x_values[used_local_min_index]] = 0
    return np.array(strokes)

# after estimation of parameters of all strokes, this function regenrate th whole sigma lognoraml profile.
# timestamps represent the used x coord values
def generate_curve_from_parameters(parameter_matrix, timestampes):
    generated_velocity = np.zeros_like(timestampes)
    for stroke in parameter_matrix: # (D, sigma, meu, t_0)
        generated_velocity += generate_lognormal_curve(stroke[0],
                                                       stroke[1],
                                                       stroke[2],
                                                       stroke[3],
                                                       timestampes[0],
                                                       timestampes[-1],
                                                       len(timestampes))
    return generated_velocity

# this method tests the estimation of all parameters of sigmal lognormal, in the best case
def test_perfect_curve():
    sec = 1
    time = np.linspace(0.01, sec, int(200 * sec))
    velocity_profile = generate_4_lognormal_curves(time)
    strokes = extract_parameters_first_mode(time, velocity_profile.copy())
    print(strokes)
    generated_velocity = generate_curve_from_parameters(strokes, time)

    print(calculate_MSE(velocity_profile, generated_velocity))

    plt.plot(time, velocity_profile, color="black", label="base velocity")
    plt.plot(time, generated_velocity, color="red", label="generated veolicity")
    plt.title("best fit")
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.legend()
    plt.show()


def extract_parameter_1stroke(xvalues, yvalues):
    v_3 = yvalues[np.argmax(yvalues)]
    t_3 = xvalues[np.argmax(yvalues)]

    # the range of y coordinations of inflection pints is calculated, n the range of x coordinations
    v2_inf1_range = np.linspace(first_range_start * v_3, first_range_end * v_3, sigma_accuracy)
    v4_inf2_range = np.linspace(second_range_start * v_3, second_range_end * v_3, sigma_accuracy)
    x_values_v2_inf1, x_values_v4_inf2 = corresponding_x_values(xvalues, yvalues, v_3, t_3,
                                                                sigma_accuracy)
    params = []
    bestgenerated = []
    for v2_inf1, x2_inf1 in zip(v2_inf1_range, x_values_v2_inf1):
        for v4_inf2, x4_inf2 in zip(v4_inf2_range, x_values_v4_inf2):
            three_char_time = [x2_inf1, t_3, x4_inf2]
            three_char_velocity = [v2_inf1, v_3, v4_inf2]
            sigmas = calculate_sigmas(x2_inf1, t_3, x4_inf2)
            # sigmas = calculate_sigmas(v2_inf1, v_3, v4_inf2)
            for sigma in sigmas:
                if sigma==0:
                    continue
                sig_sq = sigma ** 2
                a = [(3 / 2) * sig_sq + sigma * np.sqrt(((sig_sq) / 4) + 1),
                     sig_sq,
                     (3 / 2) * sig_sq - sigma * np.sqrt(((sig_sq) / 4) + 1)]
                for alpha, beta in [[0, 1], [1, 2], [0, 2]]:
                    meu = calculate_meu(three_char_time[alpha], three_char_time[beta], a[alpha], a[beta])
                    t_0 = calculate_t_0(three_char_time[alpha], a[alpha], meu)
                    D = calculate_D(a[alpha], meu, sigma, three_char_velocity[alpha])
                    params.append([D, sigma, meu, t_0])
    bestMSE = float("inf")
    bestfit = None
    for param in params:
        D, sigma, meu, t_0, = param
        generated_profile = generate_lognormal_curve(D, sigma, meu, t_0, xvalues[0], xvalues[-1], len(xvalues))
        MSE = calculate_MSE(generated_profile[generated_profile > 0.01 * np.max(generated_profile)],
                            yvalues[generated_profile > 0.01 * np.max(generated_profile)])
        if MSE < bestMSE:
            bestMSE = MSE
            bestfit = (D, sigma, meu, t_0)
    return bestfit


def get_currecnt_stroke(x_values, y_values, crossing_indices):
    biggest_auc = -float("inf")
    returned_begin_index = -1
    sign = 1
    for i in range(len(crossing_indices)-1):
        begin = crossing_indices[i]
        end = crossing_indices[i+1]
        auc = np.trapz(y_values[begin:end], x_values[begin:end])
        positive_auc = abs(auc)
        if positive_auc > biggest_auc:
            sign = np.sign(auc)
            returned_begin_index = i
            biggest_auc = positive_auc
    return crossing_indices[returned_begin_index], crossing_indices[returned_begin_index+1], sign

def extract_parameters_second_mode(time, vel_difference):
    vel_difference = interpolate(vel_difference, n_points=500)
    time = interpolate(time, n_points=500)
    crossing_indices = np.where(np.diff(np.sign(vel_difference)))[0]
    strokes=[]
    try:
        i=1
        while True:
            print(f"step {i}")
            begin, end, area_sign = get_currecnt_stroke(time, vel_difference, crossing_indices)

            x_segment = time[begin:end + 1]
            y_segment = vel_difference[begin:end + 1] * area_sign

            stroke = extract_parameter_1stroke(x_segment, y_segment)
            d, s, m, t = stroke
            d = d * area_sign
            strokes.append((d, s, m, t))
            curve = generate_lognormal_curve(d, s, m, t, x_segment[0], x_segment[-1], len(x_segment))
            vel_difference[begin:end + 1] -= curve
            i+=1
    except IndexError:
        print("no more strokes to extract")
    return strokes



if __name__ == '__main__':
    # test_perfect_curve()
    # _, _, timestamps_arr, smoothed_velocity, velocity = load_input("exmaples/keep3.npz")
    _, _, timestamps_arr, smoothed_velocity, velocity = draw_movment.get_preprocessed_input()
    plt.plot(timestamps_arr, smoothed_velocity, label="velocity")
    plt.plot(timestamps_arr, velocity, label="veolcity before smoothing")
    plt.title("smoothed velocity")
    plt.xlabel("time")
    plt.ylabel("velocity")
    plt.legend()
    plt.show()

    strokes1 = extract_parameters_first_mode(timestamps_arr, smoothed_velocity.copy())  # strokes in form (D, sigma, meu, t0)^n
    regenerated_curve = generate_curve_from_parameters(strokes1, timestamps_arr)
    plt.plot(timestamps_arr, regenerated_curve, label="regenerated", color="black")
    plt.plot(timestamps_arr, smoothed_velocity, label="original", color="red")
    plt.legend()
    plt.show()
    difference = smoothed_velocity - regenerated_curve
    strokes2 = extract_parameters_second_mode(timestamps_arr, difference)
    regenerated_curve2 = regenerated_curve + generate_curve_from_parameters(strokes2, timestamps_arr)
    plt.plot(timestamps_arr, smoothed_velocity, label="original", color="black")
    plt.plot(timestamps_arr, regenerated_curve, label="regenerated after first mode", color="red")
    plt.plot(timestamps_arr, regenerated_curve2, label="regenrated after second mode", color="green")
    plt.legend()
    plt.show()





