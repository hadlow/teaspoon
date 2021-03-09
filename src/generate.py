import csv
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import euclidean_distance_matrix

n_solutions = 50
n_locations = 10

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

def get_solution(locations):
    distance_matrix = euclidean_distance_matrix(locations)

    return solve_tsp_dynamic_programming(distance_matrix)

def write_header(values, file_name):
    with open(file_name, 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(values)

def generate_solution(start_time, i, n_locations):
    locations = np.random.rand(n_locations, 2)
    permutation, distance = get_solution(locations)
 
    x_file = './data/x_raw.csv'
    y_file = './data/y_raw.csv'

    with open(x_file, 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(locations.flatten())

    with open(y_file, 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(np.append(permutation, distance))
 
    time_taken = round(time.time() - start_time, 4)
    print(f"Solution {i+1} generated, took {time_taken} seconds")

if __name__ == "__main__":
    program_start_time = time.time()

    for i in range(n_solutions):
        start_time = time.time()

        generate_solution(start_time, i, n_locations)
    
    total_time_taken = round(time.time() - program_start_time, 4)
    print(f"{n_solutions} solutions generated in a total of {total_time_taken} seconds")

