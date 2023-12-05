# this file is used to read the datasets and preprocess the signal before it is forwarded to the algorithm.

import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

global_number = 300  # this number is used to interpolate the graph. it represents the number pf points, the graph
# should have. when the graph is only 50 points, 250 point will be added to fine the graph


# reads a row from the dataset of signatures. it is saved local not in git.
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


# reads a row from the dataset of unistrokes. it is saved in git. then the graph
# is normalized and interpolated.

def open_file_unistroke(directory: str):
    tree = ET.parse(directory)
    root = tree.getroot()
    x = np.array([int(point.get('X')) for point in root.findall('.//Point')], dtype=np.float64)
    y = np.array([int(point.get('Y')) for point in root.findall('.//Point')], dtype=np.float64)
    timestamps = np.array([int(point.get('T')) for point in root.findall('.//Point')], dtype=np.float64)
    smoothed_velocity, velocity = calculate_velocity(x, y, timestamps)
    x, y = normalize(x, y)
    x = interpolate(x, n_points=global_number)
    y = interpolate(y, n_points=global_number)
    timestamps = interpolate(timestamps, n_points=global_number)
    timestamps-=np.min(timestamps)
    smoothed_velocity = interpolate(smoothed_velocity, n_points=global_number)
    velocity = interpolate(velocity, n_points=global_number)
    return np.array(x), np.array(y), np.array(timestamps), np.array(smoothed_velocity), np.array(velocity)


# reads a row from the dataset of children. it is saved in git. then the graph
# is normalized and interpolated. However it may not work like intended due to some changes in the structure
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
    x_coordinates, y_coordinates = normalize(x_coordinates, y_coordinates)
    timestamps_arr = interpolate(timestamps_arr, n_points=global_number)
    return x_coordinates, y_coordinates, timestamps_arr


# calculates the velocity, smoothes and interpolate it,
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
    smoothed_velocity = extra_smooth(velocity, 9, 2)
    smoothed_velocity = interpolate(smoothed_velocity, n_points=global_number)
    return np.array(smoothed_velocity), velocity


# smoothes the a given curve through savgol_filter. the applies the filter once
def smooth_curve_2(velocity_data, window_size=20, poly_order=5):
    window_size = window_size  # Adjust as needed
    poly_order = poly_order  # Adjust as needed
    smoothed_velocity = savgol_filter(velocity_data, window_size, poly_order)
    return smoothed_velocity


# smoothes the a given curve through savgol_filter. the applies the filter twice.
# emperical seen, it gets better reaults.
def extra_smooth(velocity, window_size, poly):
    smoothed_velocity1 = smooth_curve_2(velocity, window_size, poly)
    smoothed_velocity2 = smooth_curve_2(smoothed_velocity1, window_size, poly)
    return smoothed_velocity2


# centerize the graph around the origin
def normalize(x, y):
    m_x = np.min(x)
    m_y = np.min(y)
    M_x = np.max(x, axis=0)
    M_y = np.max(y, axis=0)
    normalized_X = (x - (M_x + m_x) / 2.0)  # / np.max(M_x - m_x)
    normalized_Y = (y - (M_y + m_y) / 2.0)  # / np.max(M_y - m_y)
    return normalized_X, normalized_Y


# add points to a give curve. n_points is the number of points the graph should be.
# nfs is not used yet in the algorith, and it represents how many points are added between every 2 points
def interpolate(y_values, nfs=2, n_points=None, interp="cubic"):
    time = np.linspace(0, len(y_values) - 1, len(y_values), endpoint=True)
    if n_points == None:
        time_inter = np.linspace(0, len(y_values) - 1, 1 + nfs * len(y_values - 1), endpoint=True)
    else:
        time_inter = np.linspace(0, len(y_values) - 1, n_points, endpoint=True)
    f = interp1d(time, y_values, kind=interp)
    return f(time_inter)
