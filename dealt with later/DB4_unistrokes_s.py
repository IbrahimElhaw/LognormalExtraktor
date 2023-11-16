import json
import matplotlib.pyplot as plt
import numpy as np


def smooth_curve(velocity):
    window_size = 5
    smoothed_y = np.convolve(velocity, np.ones(window_size) / window_size, mode='same')
    for i in range(2):
        smoothed_y = np.convolve(smoothed_y, np.ones(window_size) / window_size, mode='same')
    window_size = 2
    for i in range(3):
        smoothed_y = np.convolve(smoothed_y, np.ones(window_size) / window_size, mode='same')
    return (smoothed_y*1.5)**1.2

# Specify the path to your JSON file
file_path = 'C:\\Users\\himaa\\Desktop\\Studium\\BP2\\DB\\4\\UnistrokeDataset\\Ziel\\arrow\\arrow43.json'

# Read the JSON data from the file
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract X-Y coordinates and timestamps
x_coordinates = []
y_coordinates = []
timestamps = []

for message in data["Tuio_Messages"]:
    ptr_data = message["/tuio2/ptr"].split()
    x = float(ptr_data[3])  # X coordinate
    y = float(ptr_data[4])  # Y coordinate
    ptr_data = message["/tuio2/frm"].split()
    timestamp = int(ptr_data[1])  # Timestamp

    x_coordinates.append(x)
    y_coordinates.append(y)
    timestamps.append(timestamp)

# Plot the X-Y coordinates
plt.figure(figsize=(8, 6))
plt.plot(x_coordinates, y_coordinates, marker='o', linestyle='-')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Human Movement')
plt.grid(True)


# Calculate movement speed using timestamps and position changes
speeds = [0]
for i in range(1, len(timestamps)):
    time_diff = (timestamps[i] - timestamps[i-1]) / 1000.0  # Convert to seconds
    distance = ((x_coordinates[i] - x_coordinates[i-1])**2 + (y_coordinates[i] - y_coordinates[i-1])**2)**0.5
    if time_diff == 0:
        speeds.append(speeds[-1])
        continue
    speed = distance / time_diff
    speeds.append(speed)

timestamps = np.array(timestamps)
smoothed_y = np.array(smooth_curve(speeds))
first_derivitive = np.gradient(timestamps, smoothed_y)
second_derivitive = np.gradient(timestamps, first_derivitive)
inflection_points = timestamps[np.where(np.diff(np.sign(second_derivitive)))[0] + 1]
print(inflection_points)

plt.figure(2)
plt.plot(timestamps, speeds, color="black")
plt.plot(timestamps, smoothed_y, color="red")
plt.scatter(inflection_points,  smoothed_y[np.where(np.diff(np.sign(second_derivitive)))[0] + 1], label="smoothed curve", color="red")


plt.show()