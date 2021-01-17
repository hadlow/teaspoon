import matplotlib.pyplot as plt
import numpy as np

train_test = 'train'

def draw_edge(a, b):
    x = [a[0], b[0]]
    y = [a[1], b[1]]

    plt.plot(x, y, 'k-')

def show_first_solution():
    locations = []
    orders = []

    inputs = np.genfromtxt('./data/x_' + train_test + '.csv', delimiter=',')
    outputs = np.genfromtxt('./data/y_' + train_test + '.csv', delimiter=',')
    outputs = np.delete(outputs, 10, axis=1)

    for x in inputs:
        locations.append(x.reshape(-1, 2))

    for index, row in enumerate(outputs):
        visits = enumerate(row)
        orders = sorted(visits, key=lambda x:x[1])
        print(orders)

        for order, location in enumerate(orders):
            if order + 1 >= len(orders):
                break

            location_x, value = location
            location_y, value = orders[order + 1]

            location_from = locations[index][location_x]
            location_to = locations[index][location_y]
            print('Location')
            print(location_from)
            print(location_to)

            draw_edge(location_from, location_to)

        plt.axis('equal')
        plt.show()

show_first_solution()
