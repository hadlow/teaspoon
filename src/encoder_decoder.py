import numpy as np

def encode(solution):
    """
    Take permutation, say:
    3, 6, 1, 0, 2, 4, 5
    and convert to values between 0 and 1 where the values
    correspond to the order in the permutation
    """

    n_locations = len(solution)

    encoded_solution = np.zeros(n_locations)
    encoding = np.round(np.linspace(0, 1, n_locations, endpoint=True), 3)

    for index, node in enumerate(solution):
        encoded_solution[node] = encoding[index]

    return encoded_solution

def decode(solution):
    """
    Take encoded solution with each index's value as
    the order it appears in the permutation
    """

    permutation = sorted(enumerate(solution), key=lambda x:x[1])
    permutation = np.array(permutation).reshape(-1, 2)

    return np.delete(permutation, 1, axis=1).flatten()

