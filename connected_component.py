
import numpy as np


def add_to_hash(h_table, low_label, high_label):
    if low_label not in h_table:
        h_table[low_label] = []
        h_table[low_label].append(high_label)
    else:
        if high_label not in h_table[low_label]:
            h_table[low_label].append(high_label)

    return h_table


def change_val(image, old_val, new_val):

    for i in range(len(image)):
        for j in range(len(image[0])):
            if image[i][j] == old_val:
                image[i][j] = new_val
    return image


def main():
    """
    image = [
        [1, 0, 1, 2],
        [0, 0, 2, 2],
        [4, 0, 0, 0],
        [4, 4, 1, 0]
    ]
    """

    """
    image = [[0,0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 0],
    [0, 1, 1, 0, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 0, 1]]
    """

    image = [[1, 0, 1, 1, 1, 1],
 [1, 0, 0, 0, 1, 0],
 [1, 1, 0, 0, 0, 1],
 [0, 0, 0, 1, 1, 1],
 [0, 0, 1, 1, 0, 1],
 [0, 0, 0, 0, 0, 0]]




    # random.binomial(n, p, size=None) : draw sample from binomial distribution
    # n trials, p probability of success, size is output shape
    #image = np.random.binomial(n=1, p=0.5, size=(6,6))
    print('image : \n', image)


    # pass 1 : row-by-row visit, check left & up, if not 0, label starts with 1
    # if up and left are not zero and different, pick lower label and store larger label to lower label
    # pass 2 : change cell value with higher label value to lower label value

    num_row = len(image)
    num_col = len(image[0])

    label = 1
    connected_label = dict()

    # row by row
    for i in range(num_row):
        for j in range(num_col):

            # init cell(0,0)
            if i == 0 and j == 0 and image[i][j] != 0 :
                image[i][j] = label
            elif i == 0 and j > 0 and image[i][j] != 0:
                if image[i][j-1] != 0:
                    image[i][j] = image[i][j-1]
                else:
                    label += 1
                    image[i][j] = label
            elif i > 0 and j == 0 and image[i][j] != 0:
                if image[i-1][j] != 0:
                    image[i][j] = image[i-1][j]
                else:
                    label += 1
                    image[i][j] = label
            elif i > 0 and j > 0 and image[i][j] != 0 :
                if image[i-1][j] != 0 and image[i][j-1] != 0:
                    if image[i-1][j] == image[i][j-1]:
                        image[i][j] = image[i-1][j]
                    else:
                        c_label = min(image[i-1][j], image[i][j-1])
                        image[i][j] = c_label
                        if c_label == image[i-1][j] :
                            connected_label = add_to_hash(connected_label, c_label, image[i][j-1])
                elif image[i-1][j] == 0 and image[i][j-1] != 0:
                        image[i][j] = image[i][j-1]
                elif image[i-1][j] != 0 and image[i][j-1] == 0:
                        image[i][j] = image[i-1][j]
                else:
                    label += 1
                    image[i][j] = label


    print('pass 1 completed \n')
    print(image)

    print('connected_label : ', connected_label)

    # the bigger keys should be done first
    #connected_label.sort(key=lambda x:x[0]*-1) # WRONG syntax, dict has no sort
    sorted_key = sorted(connected_label.items(), key=lambda x: x[0]*-1)  # sorted_key is list of tuple
    print('sorted connected_label : ', sorted_key)

    for key, val in sorted_key:  # connected_label.items():
        # val is list
        for label in val:
            image = change_val(image, label, key)

    print('pass 2 completed \n')
    print(image)



if __name__ == "__main__":
    main()