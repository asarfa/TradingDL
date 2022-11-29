import pandas as pd
import numpy as np
import random
from typing import List
from numpy.linalg import inv, eig

Vector = List[float]


if __name__ == '__main__':
    four_uniform_randoms = [random.random() for _ in range(4)]
    random.randrange(10)
    up_to_ten = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    random.shuffle(up_to_ten)
    my_best_friend = random.choice(["Alice", "Bob", "Charlie"])
    winning_numbers = random.sample(range(60), 6)

    i_hat = np.array([2, 0])
    j_hat = np.array([0, 3])
    # compose basis matrix using i-hat and j-hat
    # also need to transpose rows into columns
    basis = np.array([i_hat, j_hat]).transpose()

    A = np.array([
        [1, 2],
        [4, 5]
    ])
    eigenvals, eigenvecs = eig(A)
    Q = eigenvecs
    R = inv(Q)
    L = np.diag(eigenvals)
    B = Q @ L @ R
    assert A == B

    df = pd.read_csv('https://bit.ly/3goOAnt', delimiter=",")
    # Extract input variables (all rows, all columns but last column)
    X = df.values[:, :-1].flatten()
    # Add placeholder "1" column to generate intercept
    X_1 = np.vstack([X, np.ones(len(X))]).T
    # Extract output column (all rows, last column)
    Y = df.values[:, -1]
    # Calculate coefficents for slope and intercept
    b = inv(X_1.transpose() @ X_1) @ (X_1.transpose() @ Y)
    print(b)  # [1.93939394, 4.73333333]
    # Predict against the y-values
    y_predict = X_1.dot(b)

    print('end')