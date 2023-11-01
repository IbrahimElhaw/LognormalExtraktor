import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import fsolve
import warnings

alpha_beta = [(2,3), (2,4), (3,4), (4,2)]   # (4, 2) added because of inconsistency between papers


def calculate_sigmas(x_values, y_values):
    plt.plot(x_values, y_values)
    sigmas = []
    v_3 = np.max(y_values)
    t_3= x_values[np.argmax(y_values)]
    first_derevitive = np.gradient(y_values, x_values)
    second_dervitive = np.gradient(first_derevitive, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_dervitive)))[0] + 1]
    t2_inf1, t4_inf2 = (inflection_points[inflection_points < t_3][-1],
                      inflection_points[inflection_points > t_3][0])
    v2_inf1, v4_inf2 = y_values[(x_values==t2_inf1) | (x_values==t4_inf2)]
    plt.scatter(t2_inf1, v2_inf1, color="orange")
    plt.scatter(t_3, v_3, color="red")
    plt.scatter(t4_inf2, v4_inf2, color="blue")
    plt.show()

    #1
    beta_23 = v2_inf1/v_3
    sigma_quadrad = -2 - 2 * np.log(beta_23) - (1/(2*np.log(beta_23)))
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    #2
    beta_24 = v2_inf1/v4_inf2
    sigma_quadrad = -2 + 2*np.sqrt(1+(np.log(beta_24))**2)
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    #3
    beta_43 = v4_inf2/v_3
    sigma_quadrad  = -2 - 2*np.log(beta_43) - (1/(2*np.log(beta_43)))
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    # 4 added because of inconsistency between papers.
    beta_42 = v4_inf2 / v2_inf1
    sigma_quadrad = -2 + 2 * np.sqrt(1 + (np.log(beta_42)) ** 2)
    sigma = np.sqrt(sigma_quadrad)
    sigmas.append(sigma)

    return sigmas


def calculate_meus(x_values, y_values, sigmas):
    meus = []
    a = [np.nan, np.nan]
    a.append((3/2)*sigmas[0]**2+sigmas[0]*np.sqrt(((sigmas[0]**2)/4)+1))
    a.append(sigmas[1]**2)
    a.append((3/2)*sigmas[2]**2-sigmas[2]*np.sqrt(((sigmas[2]**2)/4)+1))
    a.append(sigmas[1] ** 2)           # added because of inconsistency between papers

    first_derevitive = np.gradient(y_values, x_values)
    second_dervitive = np.gradient(first_derevitive, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_dervitive)))[0] + 1]

    t_3 = x_values[np.argmax(y_values)]  # the biggest summit in the remaining curve
    t2_inf1, t4_inf2 = (inflection_points[inflection_points < t_3][-1], inflection_points[inflection_points > t_3][0])
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]

    for alpha, beta in alpha_beta:
        meus.append(np.log((carachterstic_times[alpha] - carachterstic_times[beta]) / (np.exp(-a[alpha]) - np.exp(-a[beta]))))
    return meus

def calculate_t_0(x_values, y_values, sigmas, meus):
    modified_meus = meus.copy()
    t_0_liste = []
    a = [np.nan, np.nan]
    a.append((3/2)*sigmas[0]**2+sigmas[0]*np.sqrt(((sigmas[0]**2/4)+1)))
    a.append(sigmas[1]**2)
    a.append((3/2)*sigmas[0]**2-sigmas[0]*np.sqrt(((sigmas[0]**2/4)+1)))
    a.append(sigmas[1] ** 2)           # added because of inconsistency between papers. the next while loop also
    while len((modified_meus)) != len(a):
        modified_meus.insert(0, np.nan)

    first_derevitive = np.gradient(y_values, x_values)
    second_dervitive = np.gradient(first_derevitive, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_dervitive)))[0] + 1]

    t_3 = x_values[np.argmax(y_values)]  # the biggest summit in the remaining curve
    t2_inf1, t4_inf2 = (inflection_points[inflection_points < t_3][-1], inflection_points[inflection_points > t_3][0])
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]
    for alpha, beta in alpha_beta:
        t_0_liste.append(carachterstic_times[alpha] - np.exp(modified_meus[alpha] - a[alpha]))
    return t_0_liste

def calculate_Ds(x_values, y_values, sigmas, meus, t_0_list):
    modified_sigmas = sigmas.copy()
    modified_meus = meus.copy()
    D = []
    a = [np.nan, np.nan]
    a.append((3 / 2) * modified_sigmas[0] ** 2 + modified_sigmas[0] * np.sqrt(((modified_sigmas[0] ** 2 / 4) + 1)))
    a.append(modified_sigmas[1] ** 2)
    a.append((3 / 2) * modified_sigmas[0] ** 2 - modified_sigmas[0] * np.sqrt(((modified_sigmas[0] ** 2 / 4) + 1)))
    a.append(modified_sigmas[1] ** 2)           # added because of inconsistency between papers. the next 2 loops also
    while len((modified_sigmas)) != len(a):
        modified_sigmas.insert(0, np.nan)
    while len((modified_meus)) != len(a):
        modified_meus.insert(0, np.nan)

    first_derevitive = np.gradient(y_values, x_values)
    second_dervitive = np.gradient(first_derevitive, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_dervitive)))[0] + 1]

    t_3 = x_values[np.argmax(y_values)]  # the biggest summit in the remaining curve
    t2_inf1, t4_inf2 = (inflection_points[inflection_points < t_3][-1], inflection_points[inflection_points > t_3][0])
    carachterstic_times = [np.nan, np.nan, t2_inf1, t_3, t4_inf2]

    for alpha, beta in alpha_beta:
        v_0 = y_values[x_values==carachterstic_times[alpha]][0]
        D.append(v_0 * modified_sigmas[alpha] * np.sqrt(2 * np.pi) * np.exp(modified_meus[alpha] + ((a[alpha] ** 2) / (2 * modified_sigmas[alpha] ** 2)) - a[alpha]))
    return D


def generate_lognormal_curve(D, std_dev, mean, x_0):
    curve =  (D / ((x_values - x_0) * std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x_values - x_0) - mean) ** 2) / (2 * std_dev ** 2))
    return curve


if __name__ == '__main__':

    # Parameters for the log-normal distribution
    D_1 = 25  # Amplitude range(5 -> 70)
    std_dev_1 = 0.15  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_1 = -1.7  # Mean (meu) range(-2.2 -> -1.6)
    x_0 = 0.5  # shifted range(0 -> 1)

    # D_2 = 6  # Amplitude range(0.5 -> 7 ungefähr = 0.1*D_1)                               *****
    # std_dev_2 = 0.1  # Standard deviation (sigma) range(0.1 -> 0.45)                      *****
    # mean_2 = -1.65  # Mean (meu) range(-2.2 -> -1.6)                                      *****

    seconds = 1
    x_values = np.linspace(x_0 + 0.01, x_0 + seconds, 10000 * seconds)

    # Generate the curve
    velocity_agonist = generate_lognormal_curve(D_1, std_dev_1, mean_1, x_0)
    # velocity_antagonist = generate_lognormal_curve(D_2, std_dev_2, mean_2, x_0)
    velocity_profile = velocity_agonist  #'- velocity_antagonist                            *****

    # estimate sigma, meu, t_0
    sigmas = calculate_sigmas(x_values, velocity_profile)
    print("sigmas\t\t", sigmas)
    meus = calculate_meus(x_values,velocity_profile, sigmas)
    print("meus\t\t", meus)
    t_0_list = calculate_t_0(x_values, velocity_profile, sigmas, meus)
    print("t_0 list\t", t_0_list)
    Ds = calculate_Ds(x_values, velocity_profile, sigmas, meus, t_0_list)
    print("D    \t\t", Ds)
    plt.plot(x_values, velocity_profile, color="black")
    colors = ["red", "blue", "orange", "green"]
    for sigma, meu, t_0, D, col in zip(sigmas, meus, t_0_list, Ds, colors):
        generated_profile = generate_lognormal_curve(D, sigma, meu, t_0)
        plt.plot(x_values, generated_profile, color=col)
    plt.show()