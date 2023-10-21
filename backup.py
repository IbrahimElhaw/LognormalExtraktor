
#######################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
import sympy as sp
from scipy.optimize import fsolve
from sympy import lambdify
import warnings


# this fincton should solve sigma

# @returns: (start index, end index, Area)
def get_sigma_und_meu_parameters(x_values, y_values, rounds = 30, procent=99.73):
    sigma, meu, t_0 = 0,0,0
    i = 0
    t_m = x_values[np.argmax(y_values)]
    area_under_curve = trapz(y_values, x_values)
    begin_search_point = x_values[0]
    for _round in range(rounds):
        iterator=0
        while True:
            if(iterator>len(x_values)):
                warnings.warn("no enought date points, t_max can not be found, longer domain in the right side is needed")
                return sigma, meu, t_0
            x_temp = x_values[i:i+iterator]
            y_temp = y_values[i:i+iterator]
            auc_temp = trapz(y_temp, x_temp)  # area_under_the_curve
            if auc_temp/area_under_curve >= (procent/100):
                t_min_index, t_max_index, auc = i, i+iterator, auc_temp
                t_min = x_values[t_min_index]
                t_max = x_values[t_max_index]
                print(f"Area under the curve: {auc}, Anteil=  {auc / trapz(y_values, x_values)}\n t_min= {t_min}\nt_m= {t_m}\nt_max={t_max}")
                break
            iterator+=1
        sigma = estimate_sigma(t_max, t_min, t_m)
        meu = calculate_meu(x_values, y_values, sigma)
        t_0 = x_values[np.argmax(y_values)] - np.exp(meu - sigma ** 2)
        begin_search_point = (t_0 + np.exp(meu - (3 * sigma)))
        print(f"int {_round+1}. round \n----\n sigma: {sigma} || meu: {meu} || t_0: {t_0} || begin_point: {begin_search_point}")
        i =  np.searchsorted(x_values, begin_search_point)
        print(f"new i= {i}")
        print("\n___________________\n")
    return sigma, meu, t_0



def calculate_meu(x_values, y_values, sigma):
    first_derevitive = np.gradient(y_values, x_values)
    second_dervitive = np.gradient(first_derevitive, x_values)
    inflection_points = x_values[np.where(np.diff(np.sign(second_dervitive)))[0]+1]
    try:
        x_inf1, x_inf2 = inflection_points[0], inflection_points[1]
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
    #print(t_min, "|", t_m, "|", t_max)
    x_solution = fsolve(equation, initial_guess, args=(t_max, t_min, t_m))
    #print("Approximate solution for x:", x_solution[0])
    return x_solution[0]


if __name__ == '__main__':
    # Parameters for the log-normal distribution
    mean = 0.5      # Mean (meu)
    std_dev = 0.5  # Standard deviation (sigma)
    D = 3         # Amplitude
    x_0 = 2         # shifted

    # Generate the curve
    x_values = np.linspace(2.01, 10 ,15000)
    y_values = (D / ((x_values-x_0) * std_dev * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x_values-x_0) - mean)**2) / (2 * std_dev**2))

    # t_max, t_min, t_m
    sigma, meu, t_0 = get_sigma_und_meu_parameters(x_values,y_values)
    D = y_values[np.argmax(y_values)] * std_dev * np.sqrt(2 * np.pi) * np.exp(mean - (std_dev ** 2 / 2))
    print(f"Sigma: {sigma}, meu {meu}, D: {D}, t_0: {t_0}")

# additional Code
'''def get_index_of_prcent_auc(x_values, y_values, procent=99.73):
    #area_under_curve = trapz(y_values, x_values)
    area_under_curve = 2.5
    #i =  int(len(x_values)/2)
    iterator=0
    while True:
        #x_temp = x_values[i-iterator:i+iterator]
        #y_temp = y_values[i-iterator:i+iterator]
        x_temp = x_values[0:iterator]
        y_temp = y_values[0:iterator]
        auc_temp = trapz(y_temp, x_temp)  # area_under_the_curve
        if auc_temp/area_under_curve >= (procent/100):
            print(iterator)
            #return i-iterator, i+iterator, auc_temp
            return 0, iterator, auc_temp
        iterator+=1'''