# this file is used to estimate all parameters of sigma lognormal profils. the main method
# is used to test the file on a perfect profile after changes. it should show 2 identical graphs
# if the changes do not negatively affect the algorithm
import warnings
import matplotlib.pyplot as plt
import numpy as np
from utilities import generate_lognormal_curve, calculate_MSE, generate_4_lognormal_curves, get_local_max, \
    get_local_min, correct_local_extrems


alpha_beta = [(2, 3), (3, 4), (4, 2)]  # alpha and beta define which characteristic points are used to estimate the
# parameter ref: https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# table 2

sigma_accuracy = 40  # how many points are used in the estimated range of characteristic points
# the more accuracy the more calculations. O(n^2). 25 seems to be optimal value


# the next 4 values are the proportion of the  start or end point of the range where the inflection points are likely
# be. first range is for the first inflection point. the second is for the second
# should be 0.43, 0.60, 0.60, 0.75 and sigma_accuracy=25, but I expanded the range to get more confident result
first_range_start = 0.35
first_range_end = 0.68
second_range_start = 0.50
second_range_end = 0.80

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
        points_after_x3 = x_values[x_values > x_3]
        corresponding_x_value3 = points_after_x3[int(len(points_after_x3)/2)]
    condition = (velocity_profile < (second_range_end * v_3)) & (x_values > x_3)
    if np.any(condition):
        corresponding_x_value4 = x_values[condition][0]
    else:
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
    sigma_quadrad = -2 - 2 * np.log(beta_23) - (1 / (2 * np.log(beta_23)))
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    # 2
    beta_42 = second_inflection / first_inflection
    sigma_quadrad = -2 + 2 * np.sqrt(1 + (np.log(beta_42)) ** 2)
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    # 3
    beta_43 = second_inflection / local_max
    sigma_quadrad = -2 - 2 * np.log(beta_43) - (1 / (2 * np.log(beta_43)))
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    return np.array(sigmas)

# calulate meu parameter
# ref: https://publications.polymtl.ca/3165/1/EPM-RT-2008-04_Djioua.pdf
# function (18)
def calculate_meus(x_values, y_values, x2_inf1, x4_inf2, sigma):
    meus = []
    a = [np.nan, np.nan]
    a.append((3 / 2) * sigma[0] ** 2 + sigma[0] * np.sqrt(((sigma[0] ** 2) / 4) + 1))
    a.append(sigma[1] ** 2)
    a.append((3 / 2) * sigma[2] ** 2 - sigma[2] * np.sqrt(((sigma[2] ** 2) / 4) + 1))

    t_3 = x_values[np.argmax(y_values)]
    t2_inf1, t4_inf2 = x2_inf1, x4_inf2
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]
    for alpha, beta in alpha_beta:
        first_term = (carachterstic_times[alpha] - carachterstic_times[beta])
        second_term = (np.exp(-a[alpha]) - np.exp(-a[beta]))
        if np.sign(first_term) != np.sign(second_term):
            second_term *= -1
        meus.append(np.log(first_term / second_term))
    return np.array(meus)

# calulate meu parameter
# ref: https://publications.polymtl.ca/3165/1/EPM-RT-2008-04_Djioua.pdf
# function (19)
def calculate_t_0(x_values, y_values, x2_inf1, x4_inf2, sigmas, meus):
    modified_meus = meus.copy()
    t_0_liste = []
    a = [np.nan, np.nan]
    a.append((3 / 2) * sigmas[0] ** 2 + sigmas[0] * np.sqrt(((sigmas[0] ** 2 / 4) + 1)))
    a.append(sigmas[1] ** 2)
    a.append((3 / 2) * sigmas[2] ** 2 - sigmas[2] * np.sqrt(((sigmas[2] ** 2 / 4) + 1)))

    while len((modified_meus)) != len(a):
        modified_meus = np.insert(modified_meus, 0, np.nan)

    t_3 = x_values[np.argmax(y_values)]
    t2_inf1, t4_inf2 = x2_inf1, x4_inf2
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]
    for alpha, beta in alpha_beta:
        t_0_liste.append(carachterstic_times[alpha] - np.exp(modified_meus[alpha] - a[alpha]))
    return np.array(t_0_liste)


# calulate meu parameter
# ref: https://publications.polymtl.ca/3165/1/EPM-RT-2008-04_Djioua.pdf
# function (22)
def calculate_Ds(x_values, y_values, x2_inf1, x4_inf2, sigmas, meus):
    modified_sigmas = sigmas.copy()
    modified_meus = meus.copy()
    D = []
    a = [np.nan, np.nan]
    a.append((3 / 2) * modified_sigmas[0] ** 2 + modified_sigmas[0] * np.sqrt(((modified_sigmas[0] ** 2 / 4) + 1)))
    a.append(modified_sigmas[1] ** 2)
    a.append((3 / 2) * modified_sigmas[2] ** 2 - modified_sigmas[2] * np.sqrt(((modified_sigmas[2] ** 2 / 4) + 1)))

    while len((modified_sigmas)) != len(a):
        modified_sigmas = np.insert(modified_sigmas, 0, np.nan)
    while len((modified_meus)) != len(a):
        modified_meus = np.insert(modified_meus, 0, np.nan)

    t_3 = x_values[np.argmax(y_values)]  # the biggest summit in the remaining curve
    t2_inf1, t4_inf2 = x2_inf1, x4_inf2
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for alpha, beta in alpha_beta:
            v_0 = y_values[np.argmin(np.abs(x_values - carachterstic_times[alpha]))]
            D.append(v_0 * modified_sigmas[alpha] * np.sqrt(2 * np.pi) * np.exp(
                modified_meus[alpha] + ((a[alpha] ** 2) / (2 * modified_sigmas[alpha] ** 2)) - a[alpha]))
    return np.array(D)


# after calculating D, the small errors can be corrected, for that the maxima is used to increase, or decrease D,
# until the regenrated maxima equals the orgiginal maxima
def correct_D(bestfit, x_values, velocity):
    while np.max(velocity) < np.max(
            generate_lognormal_curve(bestfit[0], bestfit[1], bestfit[2], bestfit[3], x_values[0], x_values[-1],
                                     len(x_values))):
        bestfit = (bestfit[0] - 1, bestfit[1], bestfit[2], bestfit[3])
    while np.max(velocity) > np.max(
            generate_lognormal_curve(bestfit[0], bestfit[1], bestfit[2], bestfit[3], x_values[0], x_values[-1],
                                     len(x_values))):
        bestfit = (bestfit[0] + 1, bestfit[1], bestfit[2], bestfit[3])
    print("corrected")
    return bestfit

# returns array of strokes parameter in form of (D, sigma, meu, t_0)^n
# for that every local maximum represent on stroke
# the outer loop is used to estimate the parameters of one stroke then it itirates to the next.
def represent_curve(x_values, y_values):
    strokes = []
    local_maximum_indices = get_local_max(y_values)
    local_minimum_indices = get_local_min(y_values)
    local_minimum_indices, local_maximum_indices = correct_local_extrems(local_minimum_indices,
                                                                                   local_maximum_indices,
                                                                                   x_values, y_values)
    plt.scatter(x_values[local_maximum_indices], y_values[local_maximum_indices], color="purple", label="maxima")
    plt.scatter(x_values[local_minimum_indices], y_values[local_minimum_indices], color="red", label="minima")

    for i in range(len(local_maximum_indices)):
        trimmed_velocity = y_values.copy()

        used_local_max_index = local_maximum_indices[i]
        used_local_min_index = local_minimum_indices[local_minimum_indices > used_local_max_index][0]

        # the values after used local min are trimmed to not calcculate the irrelevant values in the MSE (here one
        # stroke is researched).
        trimmed_velocity[x_values > x_values[used_local_min_index]] = 0

        v_3 = y_values[used_local_max_index]
        t_3 = x_values[used_local_max_index]

        # the range of y coordinations of inflection pints is calculated, n the range of x coordinations
        v2_inf1_range = np.linspace(first_range_start * v_3, first_range_end * v_3, sigma_accuracy)
        v4_inf2_range = np.linspace(second_range_start * v_3, second_range_end * v_3, sigma_accuracy)
        x_values_v2_inf1, x_values_v4_inf2 = corresponding_x_values(x_values, y_values, v_3, t_3,
                                                                    sigma_accuracy)

        # plt.plot(x_values, y_values, color="red", label= "velocity")
        # plt.plot(x_values_v2_inf1, v2_inf1_range, color="black", label="range of inflection points")
        # plt.plot(x_values_v4_inf2, v4_inf2_range, color="black", label="range of inflection points")
        # plt.title("range og infelction points")
        # plt.xlabel("time")
        # plt.ylabel("velocity")
        # plt.legend()
        # plt.show()
        #
        # plt.plot(x_values, trimmed_velocity, color="cyan", label="actual stroke")
        # plt.plot(x_values, y_values, label="velocity")
        # plt.title("actual stroke")
        # plt.xlabel("time")
        # plt.ylabel("velocity")
        # plt.legend()
        # plt.show()
        params = []

        # choose every pair of points in the estimated range to calculate sigma
        for v2_inf1, x2_inf1 in zip(v2_inf1_range, x_values_v2_inf1):
            for v4_inf2, x4_inf2 in zip(v4_inf2_range, x_values_v4_inf2):
                sigmas = calculate_sigmas(v2_inf1, v_3, v4_inf2)
                meus = calculate_meus(x_values, trimmed_velocity, x2_inf1, x4_inf2, sigmas)
                t_0s = calculate_t_0(x_values, trimmed_velocity, x2_inf1, x4_inf2, sigmas, meus)
                Ds = calculate_Ds(x_values, trimmed_velocity, x2_inf1, x4_inf2, sigmas, meus)
                for D, sigma, meu, t_0 in zip(Ds, sigmas, meus, t_0s):
                    params.append([D, sigma, meu, t_0])

        bestMSE = float("inf")
        bestfit = None
        best_generate = np.zeros_like(y_values)
        for param in params:
            D, sigma, meu, t_0,  = param
            generated_profile = generate_lognormal_curve(D, sigma, meu, t_0, x_values[0], x_values[-1], len(x_values))
            MSE = calculate_MSE(generated_profile, trimmed_velocity)
            if MSE < bestMSE:
                best_generate = generated_profile
                bestMSE = MSE
                bestfit = (D, sigma, meu, t_0)
                        # print(count)
        # plt.scatter(x_values, best_generate, color="red", label="best generated stroke")
        # plt.plot(x_values, y_values, color="blue", label= "velocity")
        # plt.title("best generate of one stroke")
        # plt.xlabel("time")
        # plt.ylabel("velocity")
        # plt.legend()
        # plt.show()
        bestfit = correct_D(bestfit, x_values, trimmed_velocity)
        strokes.append(bestfit)
        y_values -= best_generate
        y_values[x_values<x_values[np.argmax(best_generate)]] = 0
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
if __name__ == '__main__':
    sec = 1
    time = np.linspace(0.01, sec, int(200 * sec))
    velocity_profile = generate_4_lognormal_curves(time)
    strokes = represent_curve(time, velocity_profile.copy())
    print(strokes)
    generated_velocity = generate_curve_from_parameters(strokes, time)

    print(calculate_MSE(velocity_profile, generated_velocity))

    # plt.plot(time, velocity_profile, color = "black", label="base velocity")
    # plt.plot(time, generated_velocity, color = "red", label="generated veolicity")
    # plt.title("best fit")
    # plt.xlabel("time")
    # plt.ylabel("velocity")
    # plt.legend()
    # plt.show()