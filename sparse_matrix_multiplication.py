


import numpy as np





def sparse_matrix_multiplication(mat_a, mat_b, new_mat):
    row_a = len(mat_a)
    col_a = len(mat_a[0])

    row_b = len(mat_b)
    col_b = len(mat_b[0])

    # solution 1 : dot product of two matrix
    print(np.dot(mat_a, mat_b))

    # solution 2 : dot product of each vector in matrix 
    for i in range(row_a):
        for j in range(col_b):
            new_mat[i][j] = np.dot(mat_a[i, :], mat_b[:, j])

    print(new_mat)


def main():
    mat_a = [
        [0, 2, 0],
        [0, -3, 5]
    ]

    mat_b = [
        [0, 10, 0],
        [0, 0, 0],
        [0, 0, 4]
    ]

    row_a = len(mat_a)
    col_a = len(mat_a[0])

    row_b = len(mat_b)
    col_b = len(mat_b[0])

    new_mat = [[0] * col_b for _ in range(row_a)]

    # change to numpy matrix
    mat_a = np.array(mat_a)
    mat_b = np.array(mat_b)
    new_mat = np.array(new_mat)

    sparse_matrix_multiplication(mat_a, mat_b, new_mat)


if __name__ == "__main__":
    main()




