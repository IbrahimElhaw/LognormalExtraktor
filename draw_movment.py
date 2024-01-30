import sys
import matplotlib.pyplot as plt
import numpy as np
import pygame
import time

from scipy.signal import savgol_filter

frequency = 150


def calculate_velocity(x, y, timestamps):
    velocity = [0]
    for i in range(1, len(x)):
        distance = ((x[i] - x[i - 1]) ** 2 + (
                y[i] - y[i - 1]) ** 2) ** 0.5
        time = timestamps[i] - timestamps[i - 1]
        if time == 0:
            velocity.append(velocity[-1])
            continue
        velocity.append(distance / time)
    return velocity

def normalize(x, y):
    m_x = np.min(x)
    m_y = np.min(y)
    M_x = np.max(x, axis=0)
    M_y = np.max(y, axis=0)
    # normalized_X = (x - (M_x + m_x) / 2.0)  / np.max(M_x - m_x)
    # normalized_Y = (y - (M_y + m_y)  / 2.0)  / np.max(M_y - m_y)
    normalized_X = (x - m_x) / np.max(M_x - m_x)
    normalized_Y = (y - m_y) / np.max(M_y - m_y)
    return normalized_X, normalized_Y


def get_input():
    pygame.init()

    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w*0.75
    screen_height = screen_info.current_h*0.75
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    recording = False
    x_values = []
    y_values = []
    timestamps = []
    exit_flag = False  # Flag to control the outer loop

    while not exit_flag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEMOTION and recording:
                x, y = event.pos
                x_values.append(x)
                y_values.append(y)
                timestamps.append(time.time())  # Record the current timestamp
                # Your processing logic here

            elif event.type == pygame.MOUSEBUTTONDOWN:
                recording = True
                x, y = event.pos
                x_values.append(x)
                y_values.append(y)
                timestamps.append(time.time())  # Record the current timestamp

            elif event.type == pygame.MOUSEBUTTONUP:
                recording = False
                exit_flag = True
                break

        screen.fill((255, 255, 255))  # Fill the screen with white background

        if len(x_values) > 1:
            points = list(zip(x_values, y_values))
            pygame.draw.lines(screen, (0, 0, 0), False, points, 2)  # Draw lines on the screen

        pygame.display.flip()  # Update the display
        clock.tick(frequency)  # Set the desired capture frequency

    x_values = np.array(x_values)
    y_values = np.array(y_values)
    timestamps = np.array(timestamps)
    y_values *= -1

    # Close the pygame window after drawing is complete
    pygame.quit()
    if len(x_values) < 5:
        return get_input()
    return np.array(x_values), np.array(y_values), np.array(timestamps)


def show_input(x_values, y_values, timestamps):
    print("Number of captured points:", len(x_values))
    velocity = calculate_velocity(x_values, y_values, timestamps)

    plt.plot(x_values, y_values, marker='o', color="black")
    plt.scatter(x_values[0], y_values[0])
    plt.figure(2)
    plt.plot(timestamps, velocity, marker='o', color="black")
    plt.show()


# smoothes the a given curve through savgol_filter. the applies the filter once
def smooth_curve_2(velocity_data, window_size=20, poly_order=5):
    window_size = window_size  # Adjust as needed
    poly_order = poly_order  # Adjust as needed
    smoothed_velocity = savgol_filter(velocity_data, window_size, poly_order)
    return smoothed_velocity


# smoothes the a given curve through savgol_filter. the applies the filter twice.
# emperical seen, it gets better reaults.
def extra_smooth(velocity, window_size, poly=2): # int((global_number/5))
    smoothed_velocity1 = smooth_curve_2(velocity, window_size, poly)
    d_window = 10
    while window_size -d_window > poly:
        smoothed_velocity1 = smooth_curve_2(smoothed_velocity1, window_size -10 , poly)
        d_window+=10
    return smoothed_velocity1


def preprocess(t, x, y):

    # Normalize X and Y coordinates
    x, y = normalize(x, y)

    t = (t - np.min(t))  # / 1000

    # Calculate Velocity
    velocity = calculate_velocity(x, y, t)

    # Smooth Velocity
    smoothed_velocity = extra_smooth(velocity, int(frequency/10))  # int(n_points/5)

    x = extra_smooth(x, int(frequency/10))  # int(n_points/5)
    y = extra_smooth(y, int(frequency/10))  # int(n_points/5)


    return np.array(x), np.array(y), np.array(t), np.array(smoothed_velocity), np.array(velocity)


def get_preprocessed_input():
    x, y, t = get_input()
    x, y, t, s_v, v = preprocess(t, x, y)
    return x, y, t, s_v, v



if __name__ == '__main__':
    x, y, t, s_v, v = get_preprocessed_input()
    plt.plot(x, y, marker="o")
    plt.figure(2)
    plt.plot(t, v, color="cyan")
    plt.plot(t, s_v, color="red", marker="o")
    plt.show()
