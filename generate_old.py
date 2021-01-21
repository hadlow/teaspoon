import os
import time
import math
import csv
import threading
import itertools
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.exact import solve_tsp_brute_force
from python_tsp.distances import great_circle_distance_matrix
from python_tsp.distances import euclidean_distance_matrix

n_locations = 6
n_solutions = 1
train_test = 'train'

def create_location_set(n_locations):
    return np.random.rand(n_locations, 2)

def distance(location_a, location_b):
    return math.hypot(location_b[0] - location_a[0], location_b[1] - location_a[1])

def get_distance_matrix(location_set):
    distance_matrix = np.zeros([len(location_set), len(location_set)])

    for index_x, location_x in enumerate(location_set):
        for index_y, location_y in enumerate(location_set):
            distance_matrix[index_x, index_y] = distance(location_x, location_y)
    
    return distance_matrix

def permutation_length(distance_matrix, permutation):
    ind1 = permutation[:-1]
    ind2 = permutation[1:]

    return distance_matrix[ind1, ind2].sum()

def solve(location_set):
    best_distance = np.inf
    best_permutation = None

    distance_matrix = get_distance_matrix(location_set)
    points = range(0, distance_matrix.shape[0])

    for partial_permutation in itertools.permutations(points):
        permutation = list(partial_permutation)
        distance = permutation_length(distance_matrix, permutation)

        if distance < best_distance:
            best_distance = distance
            best_permutation = permutation

    return best_permutation, best_distance

def encode(solution):
    encoded_solution = np.zeros(len(solution))
    encoding = np.round(np.linspace(0, 1, n_locations, endpoint=True), 3)

    for index, node in enumerate(solution):
        encoded_solution[index] = encoding[node]

    return encoded_solution

def write_header(values, file_name):
    with open(file_name, 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(values)

def generate_solution(start_time, i):
    location_set = create_location_set(n_locations)
    distance_matrix = get_distance_matrix(location_set)
    solution, distance = solve_tsp_brute_force(distance_matrix)
    solution = encode(solution)

    x_file = './data/x_' + train_test + '.csv'
    y_file = './data/y_' + train_test + '.csv'

    if os.stat(x_file).st_size == 0:
        write_header(range(1, (n_locations * 2) + 1), x_file)

    if os.stat(y_file).st_size == 0:
        write_header(range(1, n_locations + 2), y_file)

    with open('./data/x_' + train_test + '.csv', 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(location_set.flatten())

    with open('./data/y_' + train_test + '.csv', 'a', newline = '') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(np.append(solution, distance))
 
        time_taken = round(time.time() - start_time, 4)
        print(f"Solution {i+1} generated, took {time_taken} seconds")

def generate_solutions(n_locations, n_solutions):
    for i in range(n_solutions):
        start_time = time.time()

        # thread = threading.Thread(target=generate_solution, args=[start_time, i])
        # thread.start()
        generate_solution(start_time, i)

generate_solutions(n_locations, n_solutions)

