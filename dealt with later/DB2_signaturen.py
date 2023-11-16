import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def smooth_curve(velocity):
    window_size = 2
    smoothed_y = np.convolve(velocity, np.ones(window_size) / window_size, mode='same')
    for i in range(2):
        smoothed_y = np.convolve(smoothed_y, np.ones(window_size) / window_size, mode='same')
    window_size = 2
    for i in range(3):
        smoothed_y = np.convolve(smoothed_y, np.ones(window_size) / window_size, mode='same')
    return (smoothed_y*1.5)**2

# Initialize lists to store data
x_coords = []
y_coords = []
timestamps = []

# Read data from the text file
with open('C:\\Users\\himaa\\Desktop\\Studium\\BP2\\DB\\2 SVC\\Task1\\U1S10.TXT', 'r') as file:
    lines = file.readlines()[1:]  # Skip the first line (number of signatures)
    for line in lines:
        parts = line.split()
        x_coords.append(float(parts[0]))
        y_coords.append(float(parts[1]))
        timestamps.append(int(parts[2]))


# Calculate velocity (speed) based on interpolated data
velocity = [0]
for i in range(1, len(x_coords)):
    dx = x_coords[i] - x_coords[i - 1]
    dy = y_coords[i] - y_coords[i - 1]
    dt = timestamps[i] - timestamps[i - 1]
    if (dt == 0):
        dt = 10
    speed = ((dx ** 2 + dy ** 2) ** 0.5) / dt
    velocity.append(speed)

# Plot the signature
plt.figure(1)
plt.plot(x_coords, y_coords, marker='o', linestyle='-', label='Original Signature')
plt.title('Signature')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.legend()

velocity_smoothed = smooth_curve(velocity.copy())

plt.figure(2)
plt.plot(timestamps, velocity, label='Velocity Curve', color="black")
plt.figure(3)
plt.plot(timestamps, velocity_smoothed, label='smoothed Velocity Curve', color="red")
plt.title('Velocity Curve')
plt.xlabel('Timestamp')
plt.ylabel('Velocity')
plt.grid(True)
plt.legend()

plt.show()