
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


"""
P(class|data) = (P(data|class) * P(class)/ P(data)

"""


def probability(x):
    return



"""
According to the formula we will need to calculate the probability of occurrence of every input feature as well as output feature
and their conditional probabilities given each class label.

주어진 피쳐 (즉, 칼럼 이름)의 확률, 예, 성별이면 여자 인 확률, 남자 인 확률 
"""
def get_probabilities_for_inputs(n, column_name, data_frame):
    temp = data_frame[column_name]  # isolate targetted column
    temp = temp.value_counts()  # get counts of occurences of each input variable
    # 템프는 하나의 값이 아니고, 피쳐가 가질 수 있는 값의 종류별 개수, (예, 성별이 피쳐 이면 팀프는 여성일 개수, 남성의 개수)

    # dataframe.value_counts() : Return a Series containing counts of unique rows in the DataFrame.

    #print('type(temp/n) : ', type(temp/n)) # panda Series type
    return (temp / n)  # return probiblity of occurence by dividing with total no. of data points


"""
calculate conditional probabilities for the input given an output class.
"""
def get_conditional_probabilities(data, n, target, given):
    focused_data = data[[target, given]]  # isolate target column an dfocus input column

    targets_unique = data[target].unique()  # list of unique outputs in data
    inputs_unique = data[given].unique()

    groups = focused_data.groupby(by=[given, target]).size().reset_index() # see explanation of size() and reset_index()
    #groups = focused_data.groupby(by=[given, target])
    #print(groups) # type is pandas.core.groupby.generic.DataFrameGroupBy -> so can't see directly
    #print(groups.size())  # this shows count, per given and per target,
    #print(groups.size().reset_index()) # clean up the index, the groups.size() shows differently, reset_index() show clean way
    groups[0] = groups[0] / n  #  dataframe 'groups' has column '0' and its value is the count of occurance target and give, if divide by n, then probability

    print(groups[0]) # group[0] shows only conditional probabilities

    for targets in targets_unique:
        current_target_length = len(focused_data[focused_data[target] == targets]) # 타켓 중에서, 예, 1 이면 1 , 0 이면 0
        groups[0] = np.where(groups[target] == targets, groups[0].div(current_target_length), groups[0]) # TODO
        """
        numpy.where(condition, [x, y, ]/)
        np.where() : Return elements chosen from x or y depending on condition.
        """

    return groups


"""
Our fit function : calculate and return all the necessary probabilities which we will then use for making classification
"""
def calculate_probabilities(data):
    # splititng input data
    keys = data.keys() # keys is Index(['sex', 'cp', ... ] Index data type
    x = data[data.keys()[:-1]] # x is Dataframe
    y = data[data.keys()[-1]] # y is Series, contains the target value (0 or 1)
    target = y.name

    # get length of dataframe
    n = len(data)

    # get probabilities for each individual input and output : P(x), P(y)
    """
    # this is original code but I don't understand this kind of code. So, I rewrite it. 
    f_in = lambda lst: get_probabilities_for_inputs(n, lst, x)
    input_probablities = list(map(f_in, x.keys())) # list of Series
    """
    input_probablities = []

    for key in x.keys():
        #print('key : ', key)
        #input_probablities.append(pd.Series(get_probabilities_for_inputs(n, key, x)))
        input_probablities.append(get_probabilities_for_inputs(n, key, x))

    #print(type(input_probablities[0])) # panda series
    #print(len(input_probablities[0]))
    # TODO : write this to easy way
    # input_probabilities 는 각 칼럼 이름 (피쳐, 인풋) 에 대한 확률 , 각 피쳐의 값에 대해 확률 (예, 성별이 1 일 확률은 0.69, 성별이 0일 확률은 0.30)
    # 씨피 하는 피쳐의 값이 1 일 확률 얼마, 2일 확률 얼마.. 등등 각 피처 별로 분포도가 틀리다.

    # map() function returns a map object(which is an iterator) of the results after applying the given function to each item of a given iterable (list, tuple etc.)

    # Panda.series : A Pandas Series is like a column in a table.
    # It is a one-dimensional array holding data of any type.

    output_probabilities = get_probabilities_for_inputs(n, target, y.to_frame())

    # get conditional probabilities for every input against every output
    """
    f1 = lambda lst: get_conditional_probabilities(data, n, target, lst)
    conditional_probabilities = list(map(f1, data.keys()[:-1]))
    """
    conditional_probabilities = []
    for key in data.keys()[:-1] :
        conditional_probabilities.append(get_conditional_probabilities(data, n, target, key))

    return input_probablities, output_probabilities, conditional_probabilities


# TODO
def naive_bayes_calculator(target_values, input_values, in_prob, out_prob, cond_prob):
    target_values.sort()  # sort the target values to assure ascending order
    classes = []  # initialise empty probabilites list

    for target_value in target_values:
        num = 1  # initilaise numerator
        den = 1  # initialise denominator

        # calculate denominator according to the formula : 나이브 베이시안 포물라의 분모 (denominator)
        for i, x in enumerate(input_values):
            den *= in_prob[i][x] # Series is a type of column dataframe or hashmap : in_prob contains all probabilities of input

        # calculate numerator according to the formula : 나이브 베이시안 포물라의 분자, 두 부분으로 구성된다.
        for i, x_1 in enumerate(input_values):
            temp_df = cond_prob[i] # Series type data that contains conditional probabilities
            # iloc is Purely integer-location based indexing for selection by position.
            # values() : Return a Numpy representation of the DataFrame.
            # 이 temp_df 는 i 번째 피쳐 (인풋) 에 대한 모든 경우의 확률값들, 아래 코드는 그중에 각 target 의 타입 별로 conditional probability 구한다
            temp1 = temp_df[(temp_df.iloc[:, 0] == x_1) & (temp_df.iloc[:, 1] == target_value)]
            print(temp1) # temp1 is one row (with 3 columns)

            temp = temp_df[(temp_df.iloc[:, 0] == x_1) & (temp_df.iloc[:, 1] == target_value)][0]
            print(temp) # one series
            num *= temp_df[(temp_df.iloc[:, 0] == x_1) & (temp_df.iloc[:, 1] == target_value)][0].values[0] # TODO
        num *= out_prob[target_value]

    final_probability = (num / den)  # final conditional probability value
    classes.append(final_probability)   # append probability for current class in a list

    return (classes.index(max(classes)), classes) # return the predicted class (i.e. 0 ro 1) and probability of it



def naive_bayes_predictor(test_data, outputs, in_prob, out_prob, cond_prob):
    final_predictions = []  # initialise empty list to store test predictions

    for row in test_data:
        # get prediction for current data
        predicted_class, probabilities = naive_bayes_calculator(outputs, row, in_prob, out_prob, cond_prob)

        # append to list
        final_predictions.append(predicted_class)

    return final_predictions


def get_data(data):

    # drop irrelevant columns
    data = data.drop(["age", "trestbps", "chol", "thalach", "oldpeak", "slope"], axis=1) # you have to write axis=1, because column based
    #print(data.head())

    #print(type(data.keys())) # only column names : data.keys() --> pandas.core.indexes.base.Index

    X = data[data.keys()[:-1]] # all column except the last column
    y = data[data.keys()[-1]] # only last column

    #print(X)
    #print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    data_train = pd.concat([X_train, y_train], axis=1) # column axis concatenation, thus axis = 1
    data_test = pd.concat([X_test, y_test], axis=1)

    return data_train, data_test


def main():

    # data read
    data = pd.read_csv('heart.csv')
    #print(data.head())

    data_train, data_test = get_data(data)

    in_prob, out_prob, cond_prob = calculate_probabilities(data_train)  # use training data for the initial calculations

    # testing with dummy data
    print(naive_bayes_calculator([1, 0], [1, 1, 0, 2, 1, 3, 3], in_prob, out_prob, cond_prob))

    # Test
    """
    X_test = data_test[:-1]
    y_test = data_test[-1]
    test_data_as_list = X_test.values.tolist()
    unique_targets = y_test.unique().tolist()

    predicted_y = naive_bayes_predictor(test_data_as_list, unique_targets, in_prob, out_prob, cond_prob)
    print("Accuracy:", (np.count_nonzero(y_test == predicted_y) / len(y_test)) * 100)
    """

if __name__ == "__main__":
    main()