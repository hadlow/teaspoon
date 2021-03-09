import csv
import numpy as np

from encoder_decoder import encode

def encode_solutions(file):
    locations = []
    orders = []

    outputs = np.genfromtxt('./data/y_raw.csv', delimiter=',')

    for index, permutation in enumerate(outputs):
        print(f'Encoding solution {index + 1}')

        encoded = encode(permutation[:-1].astype('int'))

        with open('./data/y_encoded.csv', 'a', newline = '') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(np.append(encoded, permutation[-1]))

encode_solutions('./data/y_encoded.csv')
