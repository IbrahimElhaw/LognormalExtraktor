import xml.etree.ElementTree as ET

import numpy as np
from scipy.signal import savgol_filter


def open_file_signature(directory: str):
    x_coords = []
    y_coords = []
    timestamps = []
    with open(directory, 'r') as file:
        lines = file.readlines()[1:]  # Skip the first line (number of signatures)
        for line in lines:
            parts = line.split()
            x_coords.append(float(parts[0]))
            y_coords.append(float(parts[1]))
            timestamps.append(int(parts[2]))

    return np.array(x_coords), np.array(y_coords), np.array(timestamps)


def open_file_unistroke(directory: str):
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

    return np.array(x), np.array(y), np.array(timestamps)


def open_file_children(path, stroke=0):
    with open(path, 'r') as file:
        xml_data = file.read()

    root = ET.fromstring(xml_data)

    x_coordinates = []
    y_coordinates = []
    timestamps = []

    touch = root.findall('.//Touch')[stroke]
    for stroke in touch.findall('.//Stroke'):
        for point in stroke.findall('.//Point'):
            x = float(point.get('X'))
            y = float(point.get('Y'))
            t = float(point.get('T'))
            x_coordinates.append(x)
            y_coordinates.append(y)
            timestamps.append(t)
    timestamps_arr = np.array(timestamps)
    y_coordinates = np.array(y_coordinates)
    x_coordinates = np.array(x_coordinates)
    if timestamps_arr[0]>3:
        timestamps_arr -= timestamps_arr[0]
        timestamps_arr /=1000
        print("done")
    x_coordinates -= np.min(x_coordinates)
    y_coordinates -= np.min(y_coordinates)

    return x_coordinates, y_coordinates, timestamps_arr


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
    return np.array(velocity)


def smooth_curve_2(velocity_data, window_size=20, poly_order=5):
    window_size = window_size  # Adjust as needed
    poly_order = poly_order # Adjust as needed
    smoothed_velocity = savgol_filter(velocity_data, window_size, poly_order)
    return smoothed_velocity


def extra_smooth(velocity):
    smoothed_velocity1 = smooth_curve_2(velocity, 50, 5)
    smoothed_velocity2 = smooth_curve_2(smoothed_velocity1, 50, 5)
    return smoothed_velocity2
