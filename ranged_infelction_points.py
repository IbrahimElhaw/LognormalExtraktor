import matplotlib.pyplot as plt
import numpy as np

import utilities
from utilities import corresponding_x_values, generate_lognormal_curve

alpha_beta = [(2, 3), (3, 4), (4, 2)]
sigma_accuracy = 25  # the more accuracy the more calculations. O(n^2)


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
    a.append((3/2)*sigmas[0]**2+sigmas[0]*np.sqrt(((sigmas[0]**2/4)+1)))
    a.append(sigmas[1]**2)
    a.append((3/2)*sigmas[2]**2-sigmas[2]*np.sqrt(((sigmas[2]**2/4)+1)))

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

    for alpha, beta in alpha_beta:
        v_0 = y_values[np.argmin(np.abs(x_values-carachterstic_times[alpha]))]
        D.append(v_0 * modified_sigmas[alpha] * np.sqrt(2 * np.pi) * np.exp(
            modified_meus[alpha] + ((a[alpha] ** 2) / (2 * modified_sigmas[alpha] ** 2)) - a[alpha]))
    return np.array(D)



# returns array of strokes parameter in form of (D, sigma, meu, t_0, MSE)^n
def represent_curve(x_values, y_values):
    strokes = []
    local_maximum_indices = np.array([i for i in range(1, len(y_values) - 1) if
                                      y_values[i] > y_values[i - 1] and y_values[i] >
                                      y_values[i + 1]])
    local_minimum_indices = np.array([i for i in range(1, len(y_values) - 1) if
                                      y_values[i] < y_values[i - 1] and y_values[i] <
                                      y_values[i + 1]])

    for i in range(len(local_maximum_indices)):
        trimmed_velocity = y_values.copy()
        used_local_max_index = local_maximum_indices[i]
        if i < len(local_maximum_indices) - 1:
            used_local_min_index = local_minimum_indices[local_minimum_indices > used_local_max_index][0]
            trimmed_velocity[x_values > x_values[used_local_min_index]] = 0

        v_3 = y_values[used_local_max_index]
        t_3 = x_values[used_local_max_index]

        v2_inf1_range = np.linspace(0.43 * v_3, 0.60 * v_3, sigma_accuracy)
        v4_inf2_range = np.linspace(0.60 * v_3, 0.75 * v_3, sigma_accuracy)
        x_values_v2_inf1, x_values_v4_inf2 = corresponding_x_values(x_values, y_values, v_3, t_3,
                                                                    sigma_accuracy)

        plt.plot(x_values, y_values, color="red")
        plt.plot(x_values_v2_inf1, v2_inf1_range, color="black")
        plt.plot(x_values_v4_inf2, v4_inf2_range, color="black")
        plt.show()

        count = 0
        bestMSE = float("inf")
        bestfit = None
        best_generate = np.zeros_like(y_values)
        plt.plot(x_values, trimmed_velocity)
        plt.plot(x_values, y_values)
        plt.show()
        for v2_inf1, x2_inf1 in zip(v2_inf1_range, x_values_v2_inf1):
            for v4_inf2, x4_inf2 in zip(v4_inf2_range, x_values_v4_inf2):
                count += 1
                sigmas = calculate_sigmas(v2_inf1, v_3, v4_inf2)
                meus = calculate_meus(x_values, trimmed_velocity, x2_inf1, x4_inf2, sigmas)
                t_0s = calculate_t_0(x_values, trimmed_velocity, x2_inf1, x4_inf2, sigmas, meus)
                Ds = calculate_Ds(x_values, trimmed_velocity, x2_inf1, x4_inf2, sigmas, meus)
                for sigma, meu, t_0, D in zip(sigmas, meus, t_0s, Ds):
                    generated_profile = generate_lognormal_curve(D, sigma, meu, t_0, seconds)
                    MSE = utilities.calculate_MSE(generated_profile, trimmed_velocity)
                    if MSE < bestMSE:
                        best_generate = generated_profile
                        bestMSE = MSE
                        bestfit = (D, sigma, meu, t_0, MSE)
                        print(count)
        plt.scatter(x_values, best_generate, color="red")
        plt.plot(x_values, y_values, color="blue")
        plt.show()
        strokes.append(bestfit)
        y_values -= best_generate
    return strokes


if __name__ == '__main__':
    seconds = 0.8
    time = np.linspace(0.01, seconds, int(200 * seconds))
    velocity_profile = utilities.generate_4_lognormal_curves(seconds)
    strokes = represent_curve(time, velocity_profile)
    for s in strokes:
        print(s)



