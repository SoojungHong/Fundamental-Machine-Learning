import json
import glob
import os
import numpy as np
import pandas as pd

yes_data_path = "/Users/soojunghong/PycharmProjects/python_exercise/data/yes/"
no_data_path = "/Users/soojunghong/PycharmProjects/python_exercise/data/no/"

ridge2_x_move_diff_list = []
ridge2_y_move_diff_list = []
alaLeft2_x_move_diff_list = []
alaLeft2_y_move_diff_list = []
baseRight_x_move_diff_list = []
baseRight_y_move_diff_list = []
class_label = []


def form_dataframe(file_path, label):
    for f in glob.glob(file_path + "*.json"):
        fname = os.path.basename(f)

        with open(file_path + fname, "r") as yes_f:
            data = json.load(yes_f)  # JSON object

            ridge2_x_move_diff = np.max(data['Ridge2']['x']) - np.min(data['Ridge2']['x'])
            ridge2_y_move_diff = np.max(data['Ridge2']['y']) - np.min(data['Ridge2']['y'])

            # AlaLeft2
            alaLeft2_x_move_diff = np.max(data['AlaLeft2']['x']) - np.min(data['AlaLeft2']['x'])
            alaLeft2_y_move_diff = np.max(data['AlaLeft2']['y']) - np.min(data['AlaLeft2']['y'])

            # BaseRight
            baseRight_x_diff = np.max(data['BaseRight']['x']) - np.min(data['BaseRight']['x'])
            baseRight_y_diff = np.max(data['BaseRight']['y']) - np.min(data['BaseRight']['y'])

            ridge2_x_move_diff_list.append(ridge2_x_move_diff)
            ridge2_y_move_diff_list.append(ridge2_y_move_diff)
            alaLeft2_x_move_diff_list.append(alaLeft2_x_move_diff)
            alaLeft2_y_move_diff_list.append(alaLeft2_y_move_diff)
            baseRight_x_move_diff_list.append(baseRight_x_diff)
            baseRight_y_move_diff_list.append(baseRight_y_diff)
            class_label.append(label)

    #data_array = np.array([ridge2_x_move_diff_list, ridge2_y_move_diff_list, alaLeft2_x_move_diff_list, alaLeft2_y_move_diff_list, baseRight_x_move_diff_list, baseRight_y_move_diff_list, class_label])
    dict_array = {
        'ridge2_x_move_diff': ridge2_x_move_diff_list, #data_array[0],
        'ridge2_y_move_diff': ridge2_y_move_diff_list, #data_array[1],
        'alaLeft2_x_move_diff': alaLeft2_x_move_diff_list, #data_array[2],
        'alaLeft2_y_move_diff': alaLeft2_y_move_diff_list, #data_array[3],
        'baseRight_x_diff': baseRight_x_move_diff_list, #data_array[4],
        'baseRight_y_diff': baseRight_y_move_diff_list, #data_array[5],
        'class': class_label #data_array[6]
    }

    data = pd.DataFrame(dict_array)
    print(data)

    return data


def main():
    yes_df = form_dataframe(yes_data_path, 1)  # label 1 : Yes
    no_df = form_dataframe(no_data_path, 0)
    df = pd.concat([yes_df, no_df], axis=0)
    print(df)

    """
    model 
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # shuffle dataframe : shuffle first before split to train with test
    from sklearn.utils import shuffle
    df = shuffle(df)

    X = df[['ridge2_x_move_diff', 'ridge2_y_move_diff', 'alaLeft2_x_move_diff', 'alaLeft2_y_move_diff', 'baseRight_x_diff', 'baseRight_y_diff' ]]
    #X = df[['ridge2_x_move_diff', 'ridge2_y_move_diff', 'alaLeft2_x_move_diff', 'alaLeft2_y_move_diff']]
    #X = df[['ridge2_x_move_diff', 'ridge2_y_move_diff']]
    y = df['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(model.score(X_test, y_test)) # only using ridge : 0.97
    y_pred = model.predict(X_test)

    # Precision, Recall, F1-Score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    precision = precision_score(y_test, y_pred, average='macro')
    print('precision : ', precision)

    recall = recall_score(y_test, y_pred, average='macro')
    print('recall : ', recall)

    f1_score = f1_score(y_test, y_pred, average='macro')
    print('f1_score : ', f1_score)



if __name__ == "__main__":
    main()


"""
using only ridge 
0.94
precision :  0.95
recall :  0.9347826086956521
f1_score :  0.9388004895960833


using also ala 
0.96
precision :  0.9661016949152542
recall :  0.9555555555555555
f1_score :  0.9592003263973888


using also baseRight 
0.9666666666666667
precision :  0.9696969696969697
recall :  0.9655172413793103
f1_score :  0.9665178571428572

"""


