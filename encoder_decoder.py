import numpy as np

def encode(solution, distance_removed=False):
    """
    Take permutation along with the total distance, say
    3, 6, 1, 0, 2, 4, 5, 1.724723523
    and convert to values between 0 and 1 where the values
    correspond to the order in the permutation, leaving
    the distance as it is.
    """

    n_locations = len(solution)

    if remove_distance:
        n_locations = n_locations - 1

    encoded_solution = np.zeros(n_locations)
    encoding = np.round(np.linspace(0, 1, n_locations, endpoint=True), 3)

    for index, node in enumerate(solution[:-1]):
        encoded_solution[node] = encoding[index]

    return np.append(encoded_solution, solution[-1])

def decode(solution):
    """
    Take encoded solution with each index's value as
    the order it appears in the permutation
    """

    permutation = sorted(enumerate(solution[:-1]), key=lambda x:x[1])
    permutation = np.array(permutation).reshape(-1, 2)

    return np.append(np.delete(permutation, 1, axis=1).flatten(), solution[-1])

