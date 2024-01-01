import sys
import matplotlib.pyplot as plt
import numpy as np
import pygame
import time

import openfiles

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
            pygame.draw.lines(screen, (0, 0, 255), False, points, 2)  # Draw lines on the screen

        pygame.display.flip()  # Update the display
        clock.tick(150)  # Set the desired capture frequency

    x_values = np.array(x_values)
    y_values = np.array(y_values)
    timestamps = np.array(timestamps)
    y_values *= -1

    # Close the pygame window after drawing is complete
    pygame.quit()

    return np.array(x_values), np.array(y_values), np.array(timestamps)


def show_input(x_values, y_values, timestamps):
    print("Number of captured points:", len(x_values))
    velocity = openfiles.calculate_velocity(x_values, y_values, timestamps)

    plt.plot(x_values, y_values, marker='o', color="black")
    plt.scatter(x_values[0], y_values[0])
    plt.figure(2)
    plt.plot(timestamps, velocity, marker='o', color="black")
    plt.show()


x, y, t = get_input()
show_input(x, y, t)



    ##########################################################################

    # import matplotlib.pyplot as plt
    # import time
    #
    # import openfiles
    #
    # recording = False
    # x_values = []
    # y_values = []
    # timestamps = []
    #
    # def on_mouse_event(event):
    #     global recording
    #
    #     if event.inaxes:
    #         x = event.xdata
    #         y = event.ydata
    #         timestamp = time.time()  # Capture the current time
    #
    #         if event.name == 'button_press_event':
    #             recording = True
    #             x_values.append(x)
    #             y_values.append(y)
    #             timestamps.append(timestamp)
    #             plt.scatter(x, y, color='blue')
    #             plt.draw()
    #             plt.xlim(0, 1)
    #             plt.ylim(0, 1)
    #
    #         elif event.name == 'motion_notify_event' and recording:
    #             x_values.append(x)
    #             y_values.append(y)
    #             timestamps.append(timestamp)
    #             plt.scatter(x, y, color='blue')
    #             plt.draw()
    #
    #         elif event.name == 'button_release_event':
    #             recording = False
    #
    # fig, ax = plt.subplots()
    # fig.canvas.mpl_connect('button_press_event', on_mouse_event)
    # fig.canvas.mpl_connect('motion_notify_event', on_mouse_event)
    # fig.canvas.mpl_connect('button_release_event', on_mouse_event)
    #
    # plt.show()

    #############################
