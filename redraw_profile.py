# this file is used to draw a lognormal profile after finding all parameters
import numpy
import numpy as np
from matplotlib import pyplot as plt

import utilities
from utilities import calculate_MSE as MSE

# defines phi which is used in the drawing of the new strokes
# ref https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# section 2, function (2)


def phi(x, x_0, meu, sigma, theta_s, theta_e):
    modified_x = x - x_0
    modified_x = np.maximum(modified_x, 0.0000001)
    term2 = (theta_e - theta_s) / 2
    term3 = 1 + np.math.erf((np.log(modified_x) - meu) / (sigma * 2**0.5))
    return_value = theta_s + term2 * term3
    return return_value


# draws a one new stroke given parameters of lognormal profile
# ref https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# section 2, function (11), (12)
def draw_stroke(D, theta_s, theta_e, time, x_0, meu, sigma, left_before_right=False):
    S_x = []
    S_y = []
    print(sigma)
    for t in time:
        if left_before_right:
            point_x = (D / (theta_e - theta_s)) * (np.sin(phi(t, x_0, meu, sigma, theta_s, theta_e)) - np.sin(theta_s))
            S_x.append(point_x)
            point_y = (D / (theta_e - theta_s)) * (-np.cos(phi(t, x_0, meu, sigma, theta_s, theta_e)) + np.cos(theta_s))
            S_y.append(point_y)
        else:
            point_x = (D / (theta_e - theta_s)) * (-np.sin(phi(t, x_0, meu, sigma, theta_s, theta_e)) + np.sin(theta_s))
            S_x.append(point_x)
            point_y = (D / (theta_e - theta_s)) * (-np.cos(phi(t, x_0, meu, sigma, theta_s, theta_e)) + np.cos(theta_s))
            S_y.append(-point_y) # negative?

    return numpy.array(S_x), numpy.array(S_y)

def draw_stroke_original(D, theta_s, theta_e, time, x_0, meu, sigma):
    S_x = []
    S_y = []
    denominator = theta_e - theta_s
    if denominator==0:
        denominator+=0.01
    for t in time:
        point_x = (D / denominator) * (np.sin(phi(t, x_0, meu, sigma, theta_s, theta_e)) - np.sin(theta_s))
        S_x.append(point_x)
        point_y = (D / denominator) * (-np.cos(phi(t, x_0, meu, sigma, theta_s, theta_e)) + np.cos(theta_s))
        S_y.append(point_y)
    return numpy.array(S_x), numpy.array(S_y)

def calculate_change_angles(x, y):
    slope = np.gradient(y, x)
    angles = np.arctan(slope)
    return np.gradient(angles) / np.gradient(x)

def correct_signX(x_values, y_values, acx, acy, trueX, trueY, char_points):
    begin = char_points[1]
    end = char_points[3]

    original = calculate_change_angles(x_values[begin:end], y_values[begin:end])

    condition = ~np.isnan(trueX) & ~np.isnan(trueY)
    neg_new_acx = acx[condition] - trueX[condition]
    neg_new_acy = acy[condition] + trueY[condition]
    negative = calculate_change_angles(neg_new_acx[begin:end], neg_new_acy[begin:end])

    pos_new_acx = acx[condition] + trueX[condition]
    pos_new_acy = acy[condition] + trueY[condition]
    positive = calculate_change_angles(pos_new_acx[begin:end], pos_new_acy[begin:end])

    condition2 = ~np.isnan(original) & ~np.isnan(positive)  & ~np.isnan(negative)
    currentMSE = MSE(original[condition2], positive[condition2])
    newMSE = MSE(original[condition2], negative[condition2])

    if newMSE < currentMSE:
        print("correctedd sign new: ", newMSE , " vs ", currentMSE)
        return -trueX
    print("not corrected sign new: ", newMSE , " vs ", currentMSE)
    return trueX


def draw_unknown_direction(D, theta_s, theta_e, time, x_0, meu, sigma, x_values, y_values, acx, acy, i=-1):
    Xl_r, Yl_r = draw_stroke(D, theta_s, theta_e, time, x_0, meu, sigma, left_before_right=True)
    condition1 = ~np.isnan(Xl_r) & ~np.isnan(Yl_r)
    acx_l_r = acx[condition1] + Xl_r[condition1]
    acy_l_r = acy[condition1] + Yl_r[condition1]
    acx_l_r -= np.min(acx_l_r)
    acy_l_r -= np.min(acy_l_r)
    MSEl_r = MSE(x_values, acx_l_r) + MSE(y_values, acy_l_r)

    Xr_l, Yr_l = draw_stroke(D, theta_s, theta_e, time, x_0, meu, sigma, left_before_right=False)
    condition2 = ~np.isnan(Xr_l) & ~np.isnan(Yr_l)
    acx_r_l = acx[condition2] + Xr_l[condition2]
    acy_r_l = acy[condition2] + Yr_l[condition2]
    acy_r_l -= np.min(acy_r_l)
    acx_r_l -= np.min(acx_r_l)
    MSEr_l = MSE(x_values, acx_r_l) + MSE(y_values, acy_r_l)

    print("MSE", MSEr_l, " ||| ", MSEl_r)
    print("This None is: ", MSEl_r < MSEr_l)

    if MSEl_r < MSEr_l:
        trueX, trueY, currentMSE = Xl_r, Yl_r, MSEl_r
    else:
        trueX, trueY, currentMSE =  Xr_l, Yr_l, MSEr_l
    char_points = utilities.find_char_points_lognormal(time, sigma, meu, x_0)
    trueX = correct_signX(x_values, y_values, acx, acy, trueX, trueY, char_points)

    return trueX, trueY


