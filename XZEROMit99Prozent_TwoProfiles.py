import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.optimize import fsolve
import warnings


# this functon solves sigma, meu and t_0
def get_sigma_und_meu_parameters(x_values, y_values, procent=0.9973):
    t_m = x_values[np.argmax(y_values)]
    area_under_curve = trapz(y_values, x_values)
    searched_area = ((1 - procent) * area_under_curve) / 2  # half of the ignored area under the curve
    iterator = 0
    area_tmin = 0  # this area is being increased from begin until reaching searched area
    while area_tmin < searched_area:
        iterator += 1
        x_temp = x_values[0: iterator]
        y_temp = y_values[0: iterator]
        area_tmin = trapz(y_temp, x_temp)
    t_min = (x_values[iterator] + x_values[iterator - 1]) / 2  # the average to get a better result
    area_tmax = 0
    iterator = 0
    while area_tmax < searched_area:
        iterator += 1
        x_temp = x_values[-iterator: -1]
        y_temp = y_values[-iterator: -1]
        area_tmax = trapz(y_temp, x_temp)
    t_max = (x_values[-iterator] + x_values[-iterator - 1]) / 2
    sigma = estimate_sigma(t_max, t_min, t_m)
    meu = calculate_meu(x_values, y_values, sigma)
    t_0 = x_values[np.argmax(y_values)] - np.exp(meu - sigma ** 2)
    return sigma, meu, t_0


def calculate_meu(x_values, y_values, sigma):
    first_derevitive = np.gradient(y_values, x_values)
    second_dervitive = np.gradient(first_derevitive, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_dervitive)))[0] + 1]
    x_coordinate_max_value = x_values[np.argmax(y_values)]  # the biggest summit in the remaining curve
    print(inflection_points)
    if len(inflection_points) != 2:
        warnings.warn(f"agonist: mehr als 2 x_inf: {len(inflection_points)}")
    try:
        #x_inf1, x_inf2 = inflection_points[-2], inflection_points[-1]
        x_inf1, x_inf2 = (inflection_points[inflection_points < x_coordinate_max_value][-1],
                          inflection_points[inflection_points > x_coordinate_max_value][0])
    except IndexError:
        raise Exception("no enought date points, x_inf2 can not be found, longer domain in the right side is needed")
    a_1 = sigma * ((sigma + np.sqrt(sigma ** 2 + 4)) / 2)
    a_2 = sigma * ((sigma - np.sqrt(sigma ** 2 + 4)) / 2)
    alpha_1 = np.exp(-a_1)
    alpha_2 = np.exp(-a_2)
    calculated_meu = (sigma ** 2) + np.log((x_inf2 - x_inf1) / (alpha_2 - alpha_1))
    return calculated_meu


def estimate_sigma(t_max, t_min, t_m, initial_guess=0.5):
    def equation(x, t1, t2, t3):
        return (t1 - t2) / (2 * np.sinh(3 * x)) * (np.exp(-x ** 2) - np.exp(-3 * x)) - (t3 - t2)

    x_solution = fsolve(equation, initial_guess, args=(t_max, t_min, t_m))
    return x_solution[0]


def estimate_sigma_antagonist(x_values, calculated_antagonist, t_0):
    x_coordinate_max_value = x_values[np.argmax(calculated_antagonist)]  # the biggest summit in the remaining curve
    first_derivitive = np.gradient(x_values, calculated_antagonist)
    second_derivitive = np.gradient(x_values, first_derivitive)
    inflection_points = x_values[np.where(np.diff(np.sign(second_derivitive)))[0] + 1]
    if len(inflection_points) != 2:
        warning = f"antagonist: mehr als 2 x_inf: {len(inflection_points)}"
        warnings.warn(warning)
    t_inf1, t_inf2 = (inflection_points[inflection_points < x_coordinate_max_value][-1],
                      inflection_points[inflection_points > x_coordinate_max_value][0])
    print(t_inf1, t_inf2 )
    B = np.log((t_inf2 - t_0) / (t_inf1 - t_0))
    calculated_sigma_antagonist = np.sqrt(np.sqrt(B ** 2 + 4) - 2)
    a_1 = calculated_sigma_antagonist * ((calculated_sigma_antagonist + np.sqrt(calculated_sigma_antagonist ** 2 + 4)) / 2)
    a_2 = calculated_sigma_antagonist * ((calculated_sigma_antagonist - np.sqrt(calculated_sigma_antagonist ** 2 + 4)) / 2)
    A = np.exp(-a_2)-np.exp(-a_1)
    calculated_meu_antagonist = calculated_sigma_antagonist ** 2 + np.log((t_inf2-t_inf1)/A)
    calculated_D_antagonist = np.max(calculated_antagonist) * calculated_sigma_antagonist * np.sqrt(2*np.pi) *\
                              np.exp(calculated_meu_antagonist - 0.5 * calculated_sigma_antagonist**2)
    return calculated_sigma_antagonist, calculated_meu_antagonist, calculated_D_antagonist




if __name__ == '__main__':
    # Parameters for the log-normal distribution
    D_1 = 60  # Amplitude range(5 -> 70)
    std_dev_1 = 0.3  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_1 = -2  # Mean (meu) range(-2.2 -> -1.6)
    x_0 = 0.4  # shifted range(0 -> 1)

    D_2 = 6  # Amplitude range(0.5 -> 7 ungefähr = 0.1*D_1)
    std_dev_2 = 0.1  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_2 = -1.65  # Mean (meu) range(-2.2 -> -1.6)

    # Generate the curve
    seconds = 1
    x_values = np.linspace(x_0 + 0.01, x_0 + seconds, 200 * seconds)
    velocity_agonist = (D_1 / ((x_values - x_0) * std_dev_1 * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x_values - x_0) - mean_1) ** 2) / (2 * std_dev_1 ** 2))
    velocity_antagonist = (D_2 / ((x_values - x_0) * std_dev_2 * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x_values - x_0) - mean_2) ** 2) / (2 * std_dev_2 ** 2))
    velocity_profile = velocity_agonist - velocity_antagonist
    plt.plot(x_values, velocity_profile, color="blue")
    plt.show()
    # estimate sigma, meu, t_0
    sigma_agonist, meu_agonist, t_0 = get_sigma_und_meu_parameters(x_values, velocity_profile)
    D_agonist = velocity_profile[np.argmax(velocity_profile)] * sigma_agonist * np.sqrt(2 * np.pi) * np.exp(
        meu_agonist - (sigma_agonist ** 2 / 2))
    print(f"Sigma: {sigma_agonist}, meu {meu_agonist}, D: {D_agonist}, t_0: {t_0}")

    if x_values[0] - t_0 < 0:
        x_values += (t_0 - x_values[0])
    generated_velocity_aganostic = (D_agonist / ((x_values - t_0) * sigma_agonist * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x_values - t_0) - meu_agonist) ** 2) / (2 * sigma_agonist ** 2))

    # velocity = aganostic - antagonist -> antagonist = aganostic - velocity
    calculated_vilocity_antagonist = -(velocity_profile - generated_velocity_aganostic)

    sigma_antagonist, meu_antagonist, D_antagonist = estimate_sigma_antagonist(x_values, calculated_vilocity_antagonist, t_0)

    generated_vilocity_antagonist = (D_antagonist / ((x_values - t_0) * sigma_antagonist * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x_values - t_0) - meu_antagonist) ** 2) / (2 * sigma_antagonist ** 2))

    generated_velocity_profile = generated_velocity_aganostic - generated_vilocity_antagonist

    calculated_vilocity_antagonist3 = -(velocity_profile - generated_velocity_profile)
    sigma_antagonist3, meu_antagonist3, D_antagonist3 = estimate_sigma_antagonist(x_values, calculated_vilocity_antagonist3, t_0)
    generated_vilocity_antagonist3 = (D_antagonist3 / (
                (x_values - t_0) * sigma_antagonist3 * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(x_values - t_0) - meu_antagonist3) ** 2) / (2 * sigma_antagonist3 ** 2))

    generated_velocity_profile = generated_velocity_profile - generated_vilocity_antagonist3
    plt.plot(x_values, velocity_profile, color="blue")
    plt.plot(x_values, generated_velocity_profile, color="black")
    plt.show()