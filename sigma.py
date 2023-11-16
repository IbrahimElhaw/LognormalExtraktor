import numpy as np
from matplotlib import pyplot as plt

alpha_beta = [(2, 3), (4, 2), (3, 4)]
# t_2/t_3 range (0.43 - 0.60)
# t_4/t_3 range (0.60 - 0.75)
def generate_lognormal_curve(D, std_dev, mean, x_0):
    curve = np.zeros_like(time)
    # Calculate the curve only for values greater than or equal to x_0
    condition = time >= x_0
    curve[condition] = (D / ((time[condition] - x_0) * std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(time[condition] - x_0) - mean) ** 2) / (2 * std_dev ** 2))
    return curve


def d_func(D, sigma, i):
    if i == 1:
        return 0
    if i == 5:
        return D
    if 1 < i < 5:
        a = [np.nan, np.nan]
        a.append((3 / 2) * sigma ** 2 + sigma * np.sqrt(((sigma ** 2) / 4) + 1))
        a.append(sigma ** 2)
        a.append((3 / 2) * sigma ** 2 - sigma * np.sqrt(((sigma ** 2) / 4) + 1))
        return (D / 2) * (1 + np.math.erf(-a[i] / (sigma * (2 ** 0.5))))
    return None

def delt_phi(t_4, t_2, D, sigma):
    angle_t4 = angle(t_4, x_values, y_values)
    angle_t2 = angle(t_2, x_values, y_values)
    d_t4 = d_func(D, sigma, 4)
    d_t2 = d_func(D, sigma, 2)
    return (angle_t4 - angle_t2) / (d_t4 - d_t2)


def angle(x_point, x_values, y_values):
    first_derevitive = np.gradient(y_values, x_values)
    index = np.argmin(np.abs(x_values - x_point))
    return np.arctan(first_derevitive[index])


def phi(x_values, x_0, meu, sigma, theta_s, theta_e):
    x_values[x_values - x_0 <= 0] *= -1
    term2 = (theta_e - theta_s) / 2
    term3 = np.array([1 + np.math.erf((np.log(x_i - x_0) - meu) / (sigma * 2**0.5)) for x_i in x_values])
    return_value = theta_s + term2 * term3
    return return_value



def draw_stroke(D, theta_s, theta_e, x_values, x_0, meu, sigma):
     S_x = (D/(theta_e-theta_s)) * (np.sin(phi(x_values,x_0, meu,sigma,theta_s,theta_e)) - np.sin(theta_s))
     S_y = (D/(theta_e-theta_s)) * (-np.cos(phi(x_values,x_0, meu,sigma,theta_s,theta_e)) + np.cos(theta_s))
     return S_x, S_y

if __name__ == '__main__':
    seconds = 1
    D_1 = 0.5 # Amplitude range(5 -> 70)
    std_dev_1 = 0.1  # Standard deviation (sigma) range(0.1 -> 0.45)
    mean_1 = -1.8  # Mean (meu) range(-2.2 -> -1.6)
    x_01 = 0.4  # shifted range(0 -> 1)

    time = np.linspace(0.01, seconds, int(200 * seconds))
    velocity_profile = generate_lognormal_curve(D_1, std_dev_1, mean_1, x_01)

    first_derevitive = np.gradient(velocity_profile, time)
    second_dervitive = np.gradient(first_derevitive, time)
    inflection_points = time[np.where(np.diff(np.sign(second_dervitive)))[0] + 1]

    t_3 = time[np.argmax(velocity_profile)]  # the biggest summit in the remaining curve
    t_1 = time[(velocity_profile < (0.01 * velocity_profile[np.argmax(velocity_profile)])) & (time < t_3)][-1]
    t_2, t_4 = (inflection_points[inflection_points < t_3][-1], inflection_points[inflection_points > t_3][0])
    t_5 = time[(velocity_profile < (0.01 * velocity_profile[np.argmax(velocity_profile)])) & (time > t_3)][0]

    # Erstellen des Plots
    x_values = np.linspace(t_1, t_5, int((t_5-t_1) * 200))
    y_values = 0.25 * np.log(x_values - 0.5) + 1.2

    term1 =  angle(t_3, x_values, y_values)                 #t3
    term2 = delt_phi(t_4, t_2, D_1, std_dev_1)              #delta phi
    term3 = d_func(D_1, std_dev_1, 5) - d_func(D_1, std_dev_1, 3) #d(t5)-d(t3)
    theta_e = term1 - term2 * term3

    term1 =  angle(t_3, x_values, y_values)                 #t3
    term2 = delt_phi(t_4, t_2, D_1, std_dev_1)              #delta phi
    term3 = d_func(D_1, std_dev_1, 3) - d_func(D_1, std_dev_1, 1) #d(t3)-d(t1)
    theta_s = term1 - term2 * term3

    print(theta_s)
    theta_s_2 = angle(t_1, x_values, y_values)
    print(theta_s_2)
    print()
    print(theta_e)
    theta_e_2 = angle(t_5, x_values, y_values)
    print(theta_e_2)

    X, Y = draw_stroke(D_1, theta_s_2, theta_e_2, x_values, x_01-0.01, mean_1, std_dev_1)
    print(X)
    print("_____________________________________________")
    print(Y)

    plt.plot(time, velocity_profile, color="black")
    plt.figure(2)
    plt.plot(x_values, y_values, color="red")
    plt.xlim(t_1-0.02, t_5+0.02)
    plt.ylim(t_1-0.4, t_5+0.1)
    plt.figure(3)
    plt.plot(X, Y, color="blue")
    plt.show()

    # theta_s = angle(t_1 , x_values, y_values)
    # print(theta_s)
    # theta_e = angle(t_5 , x_values, y_values)



