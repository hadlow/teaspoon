import numpy as np

from encoder_decoder import encode

def encode_solutions(file):
    locations = []
    orders = []

    outputs = np.genfromtxt('./data/y_raw.csv', delimiter=',')
    outputs = np.delete(outputs, len(outputs[0]) - 1, axis=1).astype('int')

    for index, permutation in enumerate(outputs):
        print(f'Showing solution {index + 1}')

        print(permutation)
        print(encode(permutation, distance_removed=True))

encode_solutions('./data/y_encoded.csv')
