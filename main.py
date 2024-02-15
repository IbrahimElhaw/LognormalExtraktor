import os
import warnings

from matplotlib import pyplot as plt

import draw_movment
import ranged_infelction_points
import estimate_angles
import numpy as np
from fastdtw import fastdtw


def redraw_lognormal(wanted_examples, tolerence=1.30, tries=20):
    x_values, y_values, timestamps_arr, smoothed_velocity, _ = draw_movment.get_preprocessed_input()
    strokes1 = ranged_infelction_points.extract_parameters_first_mode(timestamps_arr, smoothed_velocity.copy())
    angles1 = estimate_angles.estimate_angles(x_values, y_values, strokes1, timestamps_arr)
    regenerated_log_curve1 = ranged_infelction_points.generate_curve_from_parameters(strokes1, timestamps_arr)
    # plt.plot(timestamps_arr, smoothed_velocity)
    # plt.plot(timestamps_arr, regenerated_log_curve1)
    # plt.show()
    acX1, acY1 = estimate_angles.redraw(smoothed_velocity, timestamps_arr, strokes1, angles1, regenerated_log_curve1)

    original_graph = np.column_stack((x_values, y_values))
    redrawn_graph = np.column_stack((acX1, acY1))
    redrawn_distance, _ = fastdtw(original_graph, redrawn_graph)
    normalized_redrawn_distance = redrawn_distance/len(x_values)
    if normalized_redrawn_distance > 0.18:
        warnings.warn("bad quality, to make it better, Make sure to draw fast and with out gaps")
    elif normalized_redrawn_distance > 0.12:
        warnings.warn("medium quality, to make it better, Make sure to draw fast and with out gaps")
    else:
        print("good quality")

    example=0
    canceled_iterations=0
    last_canceled =float("inf")
    while example < wanted_examples:
        strokes2, angles2 = modify_all_parameters(strokes1, angles1)
        regenerated_log_curve2 = ranged_infelction_points.generate_curve_from_parameters(strokes2, timestamps_arr)
        acX2, acY2 = estimate_angles.redraw(smoothed_velocity, timestamps_arr, strokes2, angles2,
                                            regenerated_log_curve2)

        original_sequence = np.column_stack((x_values, y_values))
        redrawn_sequence = np.column_stack((acX2, acY2))
        distance, _ = fastdtw(original_sequence, redrawn_sequence)
        if distance > redrawn_distance * tolerence and canceled_iterations<tries*wanted_examples:
            # print(f"canceled {canceled_iterations} out of {5*wanted_examples}")
            canceled_iterations+=1
            last_canceled = example
            continue
        print(f"DTW distance: {distance}")
        save_example(acX2, acY2, f"example_{example}.npz")
        example+=1
        # plt.plot(x_values, y_values, color="red", label="original")
        # plt.figure(2)
        # plt.plot(acX2, acY2, color="black", label="regeneratd")
        # plt.title("final result")
        # plt.figure(3)
        # plt.plot(acX2, acY2, color="black", label="regeneratd")
        # plt.plot(x_values, y_values, color="red", label="original")
        # plt.legend()
        # plt.show()
    if canceled_iterations>=20*wanted_examples:
        warnings.warn(f"examples after {last_canceled-1} can have a great deviation")




def modify_all_parameters(strokes, angles):
    modified_strokes = []
    modified_angles = []
    for stroke, angle_pair in zip(strokes, angles):
        D, sigma, mu, t0 = stroke
        theta_s, theta_e = angle_pair
        mod_D, mod_sigma, mod_mu, mod_t0, mod_thetas, mod_thetae = modify_stroke_parameters(D, sigma, mu, t0, theta_s,
                                                                                            theta_e)
        modified_strokes.append((mod_D, mod_sigma, mod_mu, mod_t0))
        modified_angles.append((theta_s, theta_e))
    return np.array(modified_strokes), np.array(modified_angles)


def modify_stroke_parameters(D, sigma, mu, t_0, theta_s, theta_e, d_mu=0.05, d_sigma=0.05, d_t0=0.0025, d_D=0.05,
                             d_theta_s=0.05,
                             d_theta_e=0.05):
    changed_D = np.random.normal(D, (D * d_D) ** 2)
    changed_sigma = np.random.normal(sigma, (sigma * d_sigma) ** 2)
    changed_mu = np.random.normal(mu, (mu * d_mu) ** 2)
    changed_t_0 = t_0 + np.random.normal(0, (d_t0) ** 2)
    changed_theta_s = theta_s + np.random.normal(0, (d_theta_s) ** 2)
    changed_theta_e = theta_e + np.random.normal(0, (d_theta_e) ** 2)

    changed_D = np.clip(changed_D, D - 2 * d_D, D + 2 * d_D)
    changed_sigma = np.clip(changed_sigma, sigma - 2 * d_sigma, sigma + 2 * d_sigma)
    changed_mu = np.clip(changed_mu, mu - 2 * d_mu, mu + 2 * d_mu)
    changed_t_0 = np.clip(changed_t_0, t_0 - 2 * d_t0, t_0 + 2 * d_t0)
    changed_theta_s = np.clip(changed_theta_s, theta_s - 2 * d_theta_s, theta_s + 2 * d_theta_s)
    changed_theta_e = np.clip(changed_theta_e, theta_e - 2 * d_theta_e, theta_e + 2 * d_theta_e)

    if changed_sigma == 0:
        changed_sigma += 0.01
    if changed_D == 0:
        changed_D += 0.01
    return changed_D, changed_sigma, changed_mu, changed_t_0, changed_theta_s, changed_theta_e


def save_example(x_values, y_values, filename="data.npz"):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, filename)
    np.savez(file_path, array1=x_values, array2=y_values)


if __name__ == '__main__':
    try:
        redraw_lognormal(30)
    except IndexError:
        print("unexpected error, try again. Make sure to draw fast and with out gaps")

