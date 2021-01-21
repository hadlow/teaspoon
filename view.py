import random
import matplotlib.pyplot as plt
import numpy as np

def draw_edge(a, b):
    x = [a[0], b[0]]
    y = [a[1], b[1]]

    plt.plot(x, y, 'k-')

def draw_route(permutation, points):
    for position_index, location_index in enumerate(permutation):
        if len(permutation) == position_index + 1:
            next_location_index = permutation[0]
        else:
            next_location_index = permutation[position_index + 1]

        location_from = points[location_index]
        location_to = points[next_location_index]
        
        draw_edge(location_from, location_to)

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def show_solutions():
    locations = []
    orders = []

    inputs = np.genfromtxt('./data/x_raw.csv', delimiter=',')
    outputs = np.genfromtxt('./data/y_raw.csv', delimiter=',')
    outputs = np.delete(outputs, len(outputs[0]) - 1, axis=1)

    for index, permutation in enumerate(outputs):
        if index > 5:
            break

        random_index = random.randint(0, len(outputs))

        print(f'Showing solution {random_index}')

        locations = inputs[random_index].reshape(-1, 2)

        draw_route(outputs[random_index].astype(int), locations)

show_solutions()
