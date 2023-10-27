import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

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

# Sort timestamps and corresponding coordinates
sorted_data = sorted(zip(timestamps, x_coords, y_coords))
sorted_timestamps, sorted_x_coords, sorted_y_coords = zip(*sorted_data)

# Perform cubic spline interpolation
spline_x = CubicSpline(sorted_timestamps, sorted_x_coords)
interpolated_timestamps = np.linspace(min(sorted_timestamps), max(sorted_timestamps), 10*len(sorted_timestamps))
interpolated_x_coords = spline_x(interpolated_timestamps)

spline_y = CubicSpline(sorted_timestamps, sorted_y_coords)
interpolated_y_coords = spline_y(interpolated_timestamps)

# Calculate velocity (speed) based on interpolated data
velocity = [0]
for i in range(1, len(interpolated_x_coords)):
    dx = interpolated_x_coords[i] - interpolated_x_coords[i - 1]
    dy = interpolated_y_coords[i] - interpolated_y_coords[i - 1]
    dt = interpolated_timestamps[i] - interpolated_timestamps[i - 1]
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

# Plot the smoothed velocity curve
plt.figure(2)
plt.plot(interpolated_timestamps, velocity, marker='o', linestyle='-', label='Smoothed Velocity Curve')
plt.title('Velocity Curve')
plt.xlabel('Timestamp')
plt.ylabel('Velocity')
plt.grid(True)
plt.legend()

plt.show()
