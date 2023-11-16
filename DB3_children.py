import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def smooth_curve_1(velocity):
    window_size = 5
    smoothed_y = np.convolve(velocity, np.ones(window_size) / window_size, mode='same')
    for i in range(2):
        smoothed_y = np.convolve(smoothed_y, np.ones(window_size) / window_size, mode='same')
    window_size = 2
    for i in range(3):
        smoothed_y = np.convolve(smoothed_y, np.ones(window_size) / window_size, mode='same')
    return np.array(smoothed_y*1.2)**1.35


def smooth_curve_2(velocity_data):
    window_size = 20  # Adjust as needed
    poly_order = 5 # Adjust as needed

    smoothed_velocity = savgol_filter(velocity_data, window_size, poly_order)
    return smoothed_velocity

file_path = 'C:\\Users\\himaa\\Desktop\\Studium\\BP2\\DB\\3\\Dataset\\tablet\\22\\singletouch-draganddrop.xml'

with open(file_path, 'r') as file:
    xml_data = file.read()

root = ET.fromstring(xml_data)

x_coordinates = []
y_coordinates = []
timestamps = []

touch = root.findall('.//Touch')[1]
for stroke in touch.findall('.//Stroke'):
    for point in stroke.findall('.//Point'):
        x = float(point.get('X'))
        y = float(point.get('Y'))
        t = float(point.get('T'))
        x_coordinates.append(x)
        y_coordinates.append(y)
        timestamps.append(t)


velocity = [0]
for i in range(1, len(x_coordinates)):
    distance = ((x_coordinates[i] - x_coordinates[i - 1]) ** 2 + (y_coordinates[i] - y_coordinates[i - 1]) ** 2) ** 0.5
    time = timestamps[i] - timestamps[i - 1]
    if time == 0:
        velocity.append(velocity[-1])
        continue
    velocity.append(distance / time)

plt.figure(1)
plt.plot(x_coordinates, y_coordinates, marker='o', linestyle='-')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')

timestamps = np.array(timestamps)
smoothed_y = np.array(smooth_curve_2(velocity))
first_derivitive = np.gradient(timestamps, smoothed_y)
second_derivitive = np.gradient(timestamps, first_derivitive)
inflection_points = timestamps[np.where(np.diff(np.sign(second_derivitive)))[0] + 1]
print(inflection_points)

plt.figure(2)
plt.xlabel('time')
plt.ylabel('speed')
plt.plot(timestamps, velocity , linestyle='-')
plt.plot(timestamps, smoothed_y, label="smoothed curve", color="red")
plt.scatter(inflection_points,  smoothed_y[np.where(np.diff(np.sign(second_derivitive)))[0] + 1], label="smoothed curve", color="red")

plt.show()
