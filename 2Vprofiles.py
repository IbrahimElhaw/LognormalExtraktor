import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import fsolve
import warnings

# this functon solves sigma, meu and t_0
def get_sigma_und_meu_parameters(x_values, y_values, procent=0.9973):
    t_m = x_values[np.argmax(y_values)]
    area_under_curve = trapz(y_values, x_values)
    searched_area = ((1-procent) * area_under_curve)/2
    iterator = 0
    area_tmin = 0
    while area_tmin < searched_area:
        iterator += 1
        x_temp = x_values[0: iterator]
        y_temp = y_values[0: iterator]
        area_tmin = trapz(y_temp, x_temp)
    t_min = (x_values[iterator]+x_values[iterator-1])/2
    area_tmax = 0
    iterator = 0
    while area_tmax < searched_area:
        iterator += 1
        x_temp = x_values[-iterator: -1]
        y_temp = y_values[-iterator: -1]
        area_tmax = trapz(y_temp, x_temp)
    t_max = (x_values[-iterator]+x_values[-iterator-1]) / 2
    sigma = estimate_sigma(t_max, t_min, t_m)
    meu = calculate_meu(x_values, y_values, sigma)
    t_0 = x_values[np.argmax(y_values)] - np.exp(meu - sigma ** 2)
    return sigma, meu, t_0



def calculate_meu(x_values, y_values, sigma):
    first_derevitive = np.gradient(y_values, x_values)
    second_dervitive = np.gradient(first_derevitive, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_dervitive)))[0]+1]
    print(inflection_points)
    if len(inflection_points) != 2:
        warnings.warn(f"agonist: mehr als 2 x_inf: {len(inflection_points)}")
    try:
        x_inf1, x_inf2 = inflection_points[-2], inflection_points[-1]
    except IndexError:
        raise Exception("no enought date points, x_inf2 can not be found, longer domain in the right side is needed")
    a_1 = sigma * ((sigma + np.sqrt(sigma**2+4)) / 2)
    a_2 = sigma * ((sigma - np.sqrt(sigma**2+4)) / 2)
    alpha_1 = np.exp(-a_1)
    alpha_2 = np.exp(-a_2)
    calculated_meu = (sigma**2) + np.log((x_inf2-x_inf1)/(alpha_2-alpha_1))
    return calculated_meu


def estimate_sigma(t_max, t_min, t_m, initial_guess=0.5):
    def equation(x, t1, t2, t3):
        return (t1 - t2) / (2 * np.sinh(3 * x)) * (np.exp(-x ** 2) - np.exp(-3 * x)) - (t3 - t2)
    x_solution = fsolve(equation, initial_guess, args=(t_max, t_min, t_m))
    return x_solution[0]

def estimate_sigma_antagonist(x_values, y_values, t_0):
    first_derivitive = np.gradient(x_values,y_values)
    second_derivitive = np.gradient(x_values,first_derivitive)
    inflection_points = x_values[np.where(np.diff(np.sign(second_derivitive)))[0] + 1]
    if len(inflection_points) != 2:
        warning = f"antagonist: mehr als 2 x_inf: {len(inflection_points)}"
        warnings.warn(warning)
    print(inflection_points[-2:])
    print(inflection_points)
    plt.plot(x_values, y_values, color="red")
    plt.grid()
    plt.show()



if __name__ == '__main__':
    # Parameters for the log-normal distribution
    D_1 = 20         # Amplitude range(-10 -> 70)
    std_dev_1 = 0.3  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_1 = -2      # Mean (meu) range(-2.2 -> -1.6)
    x_0 = 0         # shifted range(0 -> 1)

    D_2 = 1         # Amplitude
    std_dev_2 = 0.2  # Standard deviation (sigma)
    mean_2 = -1.7     # Mean (meu)

    # Generate the curve
    seconds = 1
    x_values = np.linspace(x_0+0.01, seconds, 200*seconds)
    velocity_agonist = (D_1 / ((x_values - x_0) * std_dev_1 * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x_values - x_0) - mean_1) ** 2) / (2 * std_dev_1 ** 2))
    velocity_antagonist = (D_2 / ((x_values - x_0) * std_dev_2 * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x_values - x_0) - mean_2) ** 2) / (2 * std_dev_2 ** 2))
    velocity_profile = velocity_agonist - velocity_antagonist

    # sigma, meu, t_0
    sigma_agonist, meu_agonist, t_0 = get_sigma_und_meu_parameters(x_values, velocity_profile)
    D_agonist = velocity_profile[np.argmax(velocity_profile)] * sigma_agonist * np.sqrt(2 * np.pi) * np.exp(meu_agonist - (sigma_agonist ** 2 / 2))
    print(f"Sigma: {sigma_agonist}, meu {meu_agonist}, D: {D_agonist}, t_0: {t_0}")

    x_values+=t_0
    calulated_vilocity_aganostic = (D_agonist / ((x_values-t_0) * sigma_agonist * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x_values-t_0) - meu_agonist)**2) / (2 * sigma_agonist**2))
    calculated_vilocity_antagonist = velocity_profile - calulated_vilocity_aganostic
    plt.plot(x_values, calculated_vilocity_antagonist)
    plt.show()
    estimate_sigma_antagonist(x_values, calculated_vilocity_antagonist, t_0)