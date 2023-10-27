import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET


def calculate_gesture_speed(x_values, y_values, timestamps):
    # Berechne die Entfernung zwischen aufeinanderfolgenden Punkten
    distances = [0]
    for i in range(1, len(x_values)):
        dx = x_values[i] - x_values[i - 1]
        dy = y_values[i] - y_values[i - 1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        distances.append(distance)

    # Berechne die Zeitdifferenzen zwischen den Zeitstempeln
    time_diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]

    # Berechne die Geschwindigkeit als Durchschnitt der Geschwindigkeiten zwischen den Punkten
    speeds = [dist / time_diff for dist, time_diff in zip(distances, time_diffs)]

    return speeds


# Dateipfad zur XML-Datei
xml_file = "C:\\Users\\himaa\\Desktop\\Studium\\BP2\\DB\\1 unistrokes\\s02\\slow\\arrow04.xml"

try:
    # XML-Datei öffnen und analysieren
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Punktkoordinaten und Zeitstempel extrahieren
    x_values = [int(point.get('X')) for point in root.findall('.//Point')]
    y_values = [int(point.get('Y')) for point in root.findall('.//Point')]
    timestamps = [int(point.get('T')) for point in root.findall('.//Point')]

    # Geste zeichnen
    plt.figure(1)
    plt.plot(x_values, y_values, marker='o', linestyle='-')
    plt.title('Gezeichnete Geste')
    plt.xlabel('X-Koordinaten')
    plt.ylabel('Y-Koordinaten')
    plt.grid(True)

    # Geschwindigkeiten berechnen
    speeds = calculate_gesture_speed(x_values, y_values, timestamps)

    # Geschwindigkeitsgraph erstellen
    plt.figure(2)
    plt.plot(range(1, len(speeds) + 1), speeds, marker='o', linestyle='-')
    plt.title('Geschwindigkeitsgraph')
    plt.xlabel('Zeitpunkt')
    plt.ylabel('Geschwindigkeit (Pixel/ms)')
    plt.grid(True)

    plt.show()

except FileNotFoundError:
    print(f"Die Datei '{xml_file}' wurde nicht gefunden.")
except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")
