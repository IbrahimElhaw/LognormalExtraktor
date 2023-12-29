# this file is used to draw a lognormal profile after finding all parameters
import numpy
import numpy as np
from matplotlib import pyplot as plt
import utilities

# defines phi which is used in the drawing of the new strokes
# ref https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# section 2, function (2)
def phi(x, x_0, meu, sigma, theta_s, theta_e):
    term2 = (theta_e - theta_s) / 2
    term3 = 1 + np.math.erf((np.log(x - x_0) - meu) / (sigma * 2**0.5))
    return_value = theta_s + term2 * term3
    return return_value


# draws a one new stroke given parameters of lognormal profile
# ref https://www.sciencedirect.com/science/article/pii/S0031320308004470?fr=RR-2&ref=pdf_download&rr=830343019bb9faee
# section 2, function (11), (12)
def draw_stroke(D, theta_s, theta_e, time, x_0, meu, sigma):
    S_x = []
    S_y = []
    for x, y in zip (time, time):
        point_x = (D / (theta_e - theta_s)) * (np.sin(phi(x, x_0, meu, sigma, theta_s, theta_e)) - np.sin(theta_s))
        S_x.append(point_x)
        point_y = (D / (theta_e - theta_s)) * (-np.cos(phi(y, x_0, meu, sigma, theta_s, theta_e)) + np.cos(theta_s))
        S_y.append(point_y)
    return numpy.array(S_x), numpy.array(S_y)


