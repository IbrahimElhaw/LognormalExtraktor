import warnings

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.interpolate import CubicSpline
import numpy as np
from scipy.ndimage import gaussian_filter


def open_file(directory: str):
    '''txt_file = open(directory, 'r')
    lines = txt_file.readlines()
    x = [int(line.split()[0]) for line in lines[1:]]
    y = [int(line.split()[1]) for line in lines[1:]]
    timestamps = [int(line.split()[2]) for line in lines[1:]]'''

    x = []
    y = []
    timestamps = []

    # XML-Datei öffnen und parsen
    tree = ET.parse(directory)
    root = tree.getroot()
    x = [int(point.get('X')) for point in root.findall('.//Point')]
    y = [int(point.get('Y')) for point in root.findall('.//Point')]
    timestamps = [int(point.get('T')) for point in root.findall('.//Point')]

    #velocity berechnen
    velocity = [0]
    for i in range(1, len(x)):
        distance = ((x[i] - x[i - 1]) ** 2 + (y[i] - y[i - 1]) ** 2) ** 0.5
        time = timestamps[i] - timestamps[i - 1]
        try:
            velocity.append(distance / time)
        except ZeroDivisionError:
            warnings.warn("devide by 0.")
            velocity.append(velocity[-1])
    return np.array(x), np.array(y), np.array(timestamps), np.array(velocity)

def smooth_curve(velocity):
    window_size = 5
    smoothed_y = np.convolve(velocity, np.ones(window_size) / window_size, mode='same')
    for i in range(2):
        smoothed_y = np.convolve(smoothed_y, np.ones(window_size) / window_size, mode='same')
    window_size = 2
    for i in range(3):
        smoothed_y = np.convolve(smoothed_y, np.ones(window_size) / window_size, mode='same')
    return (smoothed_y*1.5)**2


if __name__ == '__main__':
    x, y, timestamps, velocity = open_file('C:\\Users\\himaa\\Desktop\\Studium\\BP2\\DB\\1 unistrokes\\s02\\fast\\star07.xml')
    smoothed_y = smooth_curve(velocity)
    first_derivitive = np.gradient(x, smoothed_y)
    second_derivitive = np.gradient(x, first_derivitive)
    inflection_points = timestamps[np.where(np.diff(np.sign(second_derivitive)))[0] + 1]
    print(inflection_points)



    plt.plot(timestamps,velocity, color="black")
    plt.plot(timestamps, smoothed_y,  color="red")
    # plt.plot(timestamps, second_derivitive,  color="blue")
    plt.scatter(inflection_points, smoothed_y[np.where(np.diff(np.sign(second_derivitive)))[0] + 1])
    plt.figure(2)
    plt.plot(x, y)
    # plt.ylim(-1,3.5)
    plt.show()

