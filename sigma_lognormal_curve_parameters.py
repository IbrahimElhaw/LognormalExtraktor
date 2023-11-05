import numpy as np
import matplotlib.pyplot as plt

alpha_beta = [(2, 3), (4, 2), (3, 4)]


def calculate_sigmas(x_values, y_values, local_max_indezies):
    sigmas = []
    v_3 = y_values[local_max_indezies[0]]
    v2_inf1, v4_inf2 = get_inflex_points_y_co(x_values, y_values, local_max_indezies)

    # 1
    beta_23 = v2_inf1/v_3
    sigma_quadrad = -2 - 2 * np.log(beta_23) - (1/(2*np.log(beta_23)))
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    # 2
    beta_42 = v4_inf2 / v2_inf1
    sigma_quadrad = -2 + 2 * np.sqrt(1 + (np.log(beta_42)) ** 2)
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    # 3
    beta_43 = v4_inf2/v_3
    sigma_quadrad = -2 - 2*np.log(beta_43) - (1/(2*np.log(beta_43)))
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    return sigmas


def calculate_meus(x_values, y_values, sigmas, local_max_indezies):
    meus = []
    a = [np.nan, np.nan]
    a.append((3/2)*sigmas[0]**2+sigmas[0]*np.sqrt(((sigmas[0]**2)/4)+1))
    a.append(sigmas[1]**2)
    a.append((3/2)*sigmas[2]**2-sigmas[2]*np.sqrt(((sigmas[2]**2)/4)+1))

    first_derevitive = np.gradient(y_values, x_values)
    second_dervitive = np.gradient(first_derevitive, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_dervitive)))[0] + 1]

    t_3 = x_values[local_max_indezies[0]]  # the biggest summit in the remaining curve
    t2_inf1, t4_inf2 = (inflection_points[inflection_points < t_3][-1], inflection_points[inflection_points > t_3][0])
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]
    for alpha, beta in alpha_beta:
        first_term = (carachterstic_times[alpha] - carachterstic_times[beta])
        second_term = (np.exp(-a[alpha]) - np.exp(-a[beta]))
        if np.sign(first_term)!= np.sign(second_term):
            second_term*=-1
        meus.append(np.log(first_term/second_term))
    return meus


def calculate_t_0(x_values, y_values, sigmas, meus, local_max_indezies):
    modified_meus = meus.copy()
    t_0_liste = []
    a = [np.nan, np.nan]
    a.append((3/2)*sigmas[0]**2+sigmas[0]*np.sqrt(((sigmas[0]**2/4)+1)))
    a.append(sigmas[1]**2)
    a.append((3/2)*sigmas[0]**2-sigmas[0]*np.sqrt(((sigmas[0]**2/4)+1)))

    while len((modified_meus)) != len(a):
        modified_meus.insert(0, np.nan)

    first_derevitive = np.gradient(y_values, x_values)
    second_dervitive = np.gradient(first_derevitive, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_dervitive)))[0] + 1]

    t_3 = x_values[local_max_indezies[0]]  # the biggest summit in the remaining curve
    t2_inf1, t4_inf2 = (inflection_points[inflection_points < t_3][-1], inflection_points[inflection_points > t_3][0])
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]
    for alpha, beta in alpha_beta:
        t_0_liste.append(carachterstic_times[alpha] - np.exp(modified_meus[alpha] - a[alpha]))
    return t_0_liste

def calculate_Ds(x_values, y_values, sigmas, meus, local_max_indezies):
    modified_sigmas = sigmas.copy()
    modified_meus = meus.copy()
    D = []
    a = [np.nan, np.nan]
    a.append((3 / 2) * modified_sigmas[0] ** 2 + modified_sigmas[0] * np.sqrt(((modified_sigmas[0] ** 2 / 4) + 1)))
    a.append(modified_sigmas[1] ** 2)
    a.append((3 / 2) * modified_sigmas[0] ** 2 - modified_sigmas[0] * np.sqrt(((modified_sigmas[0] ** 2 / 4) + 1)))

    while len((modified_sigmas)) != len(a):
        modified_sigmas.insert(0, np.nan)
    while len((modified_meus)) != len(a):
        modified_meus.insert(0, np.nan)

    first_derevitive = np.gradient(y_values, x_values)
    second_dervitive = np.gradient(first_derevitive, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_dervitive)))[0] + 1]

    t_3 = x_values[local_max_indezies[0]]  # the biggest summit in the remaining curve
    t2_inf1, t4_inf2 = (inflection_points[inflection_points < t_3][-1], inflection_points[inflection_points > t_3][0])
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]

    for alpha, beta in alpha_beta:
        v_0 = y_values[x_values==carachterstic_times[alpha]][0]
        D.append(v_0 * modified_sigmas[alpha] * np.sqrt(2 * np.pi) * np.exp(modified_meus[alpha] + ((a[alpha] ** 2) / (2 * modified_sigmas[alpha] ** 2)) - a[alpha]))
    return D


def get_inflex_points_y_co(x_values, y_values, local_max_indezies):
    t_3 = x_values[local_max_indezies[0]]
    first_derivative = np.gradient(y_values, x_values)
    second_derivative = np.gradient(first_derivative, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_derivative)))[0] + 1]

    # the inflection_points around the biggest summit
    t2_inf1, t4_inf2 = (inflection_points[inflection_points < t_3][-1],
                        inflection_points[inflection_points > t_3][0])
    v2_inf1, v4_inf2 = y_values[(x_values == t2_inf1) | (x_values == t4_inf2)]

    x_index1 = np.where(x_values == t2_inf1)[0]
    x_index2 = np.where(x_values == t4_inf2)[0]

    next_element1 = y_values[x_index2 + 1]
    next_element2 = y_values[x_index1 + 1]

    # get weighted average of the 2 points around the real inflex points
    weighted1 = (4 * v2_inf1 + next_element1) / 5
    weighted2 = (4 * v4_inf2 + 2 * next_element2) / 6
    return weighted1[0], weighted2[0]


def generate_lognormal_curve(D, std_dev, mean, x_0):
    curve = np.zeros_like(x_values)
    # Calculate the curve only for values greater than or equal to x_0
    condition = x_values >= x_0
    curve[condition] = (D / ((x_values[condition] - x_0) * std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x_values[condition] - x_0) - mean) ** 2) / (2 * std_dev ** 2))
    return curve


def calculate_MSE(real_y_values, forged_yvalues):
    return np.mean((real_y_values-forged_yvalues)**2)


# returns (D, sigma, meu, t_0, MSE)
def get_bestfit(x_values, y_values, local_max_indezies):
    sigmas = calculate_sigmas(x_values, y_values, local_max_indezies)
    # print("sigmas\t\t", sigmas)
    meus = calculate_meus(x_values, y_values, sigmas, local_max_indezies)
    # print("meus\t\t", meus)
    t_0_list = calculate_t_0(x_values, y_values, sigmas, meus, local_max_indezies)
    # print("t_0 list\t", t_0_list)
    Ds = calculate_Ds(x_values, y_values, sigmas, meus, local_max_indezies)
    # print("D    \t\t", Ds)

    # plt.scatter(x_values, y_values, color="black")
    trimmed_y_values = y_values.copy()
    if len(local_max_indezies) > 1:
        local_minimum_indezies = []
        for i in range(1, len(y_values) - 1):
            if velocity_profile[i] < velocity_profile[i - 1] and velocity_profile[i] < velocity_profile[i + 1]:
                local_minimum_indezies.append(i)
        # print("max:", x_values[local_max_indezies], y_values[local_max_indezies])
        # print("min: ", x_values[local_minimum_indezies], y_values[local_minimum_indezies])
        local_minimum_indezies = np.array(local_minimum_indezies)
        condition = local_minimum_indezies > local_max_indezies[0]
        used_local_min = local_minimum_indezies[condition][0]
        trimmed_y_values = np.zeros_like(y_values)
        condition = x_values < x_values[used_local_min]
        trimmed_y_values[condition] = y_values[condition]
        # plt.plot(x_values, trimmed_y_values)
        # plt.show()

    colors = ["purple", "blue", "green"]
    bestfit = None
    bestMSE = float("inf")
    for sigma, meu, t_0, D, col in zip(sigmas, meus, t_0_list, Ds, colors):
        generated_profile = generate_lognormal_curve(D, sigma, meu, t_0)
        MSE = calculate_MSE(trimmed_y_values, generated_profile)
        if MSE < bestMSE:
            bestMSE = MSE
            bestfit = (D, sigma, meu, t_0, MSE)
            # print(f"color {col}: {MSE}")
            # plt.plot(x_values, generated_profile, color=col)
    # plt.show()
    return bestfit

def represent_curve(x_values, y_values):
    parameter_matrix = []
    velocity_profile_copy = y_values.copy()
    local_max_indezies = []
    for i in range(1, len(velocity_profile) - 1):
        if velocity_profile[i] > velocity_profile[i - 1] and velocity_profile[i] > velocity_profile[i + 1]:
            local_max_indezies.append(i)
    for i in range(len(local_max_indezies)):
        bestfit = get_bestfit(x_values, velocity_profile_copy, local_max_indezies)
        parameter_matrix.append(bestfit)
        generated_curve = generate_lognormal_curve(bestfit[0], bestfit[1], bestfit[2], bestfit[3])
        plt.plot(x_values, velocity_profile, color="red")
        plt.plot(x_values, generated_curve, color="black")
        plt.show()
        velocity_profile_copy -= generate_lognormal_curve(bestfit[0], bestfit[1], bestfit[2], bestfit[3])
        local_max_indezies = local_max_indezies[1:]
        if len(local_max_indezies) == 0:
            print("curve's end reached")
            break

    generated_curve = np.zeros_like(velocity_profile_copy)
    for vector in parameter_matrix:
        generated_curve += generate_lognormal_curve(vector[0], vector[1], vector[2], vector[3])
    return generated_curve, parameter_matrix


# main is used for test
if __name__ == '__main__':
    # Parameters for the sigma log-normal distribution
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
    x_04 = 0.6  # shifted range(0 -> 1)

    seconds = 1.5
    x_values = np.linspace(0.01,  seconds, int(200 * seconds))
    # Generate the curve
    velocity1 = generate_lognormal_curve(D_1, std_dev_1, mean_1, x_01)
    velocity2 = generate_lognormal_curve(D_2, std_dev_2, mean_2, x_02)
    velocity3 = generate_lognormal_curve(D_3, std_dev_3, mean_3, x_03)
    velocity4 = generate_lognormal_curve(D_4, std_dev_4, mean_4, x_04)
    velocity_profile = velocity1 + velocity2 + velocity3 + velocity4

    generated_curve, parameter = represent_curve(x_values, velocity_profile)

    plt.plot(x_values, velocity_profile, color="red", label="real")
    plt.plot(x_values, generated_curve, color="black")
    plt.show()

