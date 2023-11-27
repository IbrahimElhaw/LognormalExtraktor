import warnings

import matplotlib.pyplot as plt
import numpy as np

import utilities
from utilities import corresponding_x_values, generate_lognormal_curve, calculate_MSE, generate_4_lognormal_curves

alpha_beta = [(2, 3), (3, 4), (4, 2)]
# should be 0.43, 0.60, 0.60, 0.75 and sigma_accuracy=25
sigma_accuracy = 40  # the more accuracy the more calculations. O(n^2). 25 seems to be optimal value
first_range_start = 0.35  # bad structure. if changed, do not forget to change the values in utilities.corresponding_x_values
first_range_end = 0.68
second_range_start = 0.50
second_range_end = 0.80

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


def calculate_meus(x_values, y_values, x2_inf1, x4_inf2, sigmas):
    meus = []
    a = [np.nan, np.nan]
    a.append((3 / 2) * sigmas[0] ** 2 + sigmas[0] * np.sqrt(((sigmas[0] ** 2) / 4) + 1))
    a.append(sigmas[1] ** 2)
    a.append((3 / 2) * sigmas[2] ** 2 - sigmas[2] * np.sqrt(((sigmas[2] ** 2) / 4) + 1))

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


# returns array of strokes parameter in form of (D, sigma, meu, t_0, MSE)^n
def represent_curve(x_values, y_values):
    strokes = []
    local_maximum_indices = utilities.get_local_max(y_values)
    local_minimum_indices = utilities.get_local_min(y_values)
    local_minimum_indices, local_maximum_indices = utilities.correct_local_extrems(local_minimum_indices,
                                                                                   local_maximum_indices,
                                                                                   x_values, y_values)
    plt.scatter(x_values[local_maximum_indices], y_values[local_maximum_indices], color="purple")
    plt.scatter(x_values[local_minimum_indices], y_values[local_minimum_indices], color="red")

    for i in range(len(local_maximum_indices)):
        trimmed_velocity = y_values.copy()
        used_local_max_index = local_maximum_indices[i]

        used_local_min_index = local_minimum_indices[local_minimum_indices > used_local_max_index][0]
        trimmed_velocity[x_values > x_values[used_local_min_index]] = 0

        v_3 = y_values[used_local_max_index]
        t_3 = x_values[used_local_max_index]

        v2_inf1_range = np.linspace(first_range_start * v_3, first_range_end * v_3, sigma_accuracy)
        v4_inf2_range = np.linspace(second_range_start * v_3, second_range_end * v_3, sigma_accuracy)
        x_values_v2_inf1, x_values_v4_inf2 = corresponding_x_values(x_values, y_values, v_3, t_3,
                                                                    sigma_accuracy)

        # plt.plot(x_values, y_values, color="red")
        # plt.plot(x_values_v2_inf1, v2_inf1_range, color="black")
        # plt.plot(x_values_v4_inf2, v4_inf2_range, color="black")
        # plt.show()

        count = 0
        bestMSE = float("inf")
        bestfit = None
        best_generate = np.zeros_like(y_values)
        # plt.plot(x_values, trimmed_velocity, color="cyan")
        # plt.plot(x_values, y_values)
        # plt.show()
        for v2_inf1, x2_inf1 in zip(v2_inf1_range, x_values_v2_inf1):
            for v4_inf2, x4_inf2 in zip(v4_inf2_range, x_values_v4_inf2):
                count += 1
                sigmas = calculate_sigmas(v2_inf1, v_3, v4_inf2)
                meus = calculate_meus(x_values, trimmed_velocity, x2_inf1, x4_inf2, sigmas)
                t_0s = calculate_t_0(x_values, trimmed_velocity, x2_inf1, x4_inf2, sigmas, meus)
                Ds = calculate_Ds(x_values, trimmed_velocity, x2_inf1, x4_inf2, sigmas, meus)
                for sigma, meu, t_0, D in zip(sigmas, meus, t_0s, Ds):
                    generated_profile = generate_lognormal_curve(D, sigma, meu, t_0, x_values[0], x_values[-1], len(x_values))
                    MSE = calculate_MSE(generated_profile, trimmed_velocity)
                    if MSE < bestMSE:
                        best_generate = generated_profile
                        bestMSE = MSE
                        bestfit = (D, sigma, meu, t_0)
                        # print(count)
        # plt.scatter(x_values, best_generate, color="red")
        # plt.plot(x_values, y_values, color="blue")
        # plt.show()
        strokes.append(bestfit)
        y_values -= best_generate
        y_values[x_values<x_values[np.argmax(best_generate)]] = 0
    return np.array(strokes)


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


if __name__ == '__main__':
    sec = 1
    time = np.linspace(0.01, sec, int(200 * sec))
    velocity_profile = generate_4_lognormal_curves(time)
    strokes = represent_curve(time, velocity_profile.copy())
    print(strokes)
    generated_velocity = generate_curve_from_parameters(strokes, time)

    print(calculate_MSE(velocity_profile, generated_velocity))

    plt.plot(time, velocity_profile, color = "black")
    plt.plot(time, generated_velocity, color = "red")
    plt.show()