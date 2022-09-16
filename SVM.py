"""
Reference : https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47

data download : https://www.kaggle.com/code/jchen2186/machine-learning-with-iris-dataset/data
"""

import pandas as pd

df = pd.read_csv('iris.csv')
df = df.drop(['Id'], axis=1)
target = df['Species']

s = set()
for val in target:
    s.add(val)



"""
Since the Iris dataset has three classes, we will remove one of the classes. 
This leaves us with a binary class classification problem.
"""

s = list(s)
rows = list(range(100, 150))
df = df.drop(df.index[rows]) # original num of row in data was 150, but by drop(), now df has 100 rows

print('df : \n', df)



"""
there are four features available for us to use. We will be using only two features, 
i.e Sepal length and Petal length. We take these two features and plot them to visualize. 
"""

import matplotlib.pyplot as plt
x = df['SepalLengthCm']
y = df['PetalLengthCm']

setosa_x = x[:50] # in data, setosa is until row 50
setosa_y = y[:50]

versicolor_x = x[50:]
versicolor_y = y[50:]

plt.figure(figsize=(8,6))
plt.scatter(setosa_x, setosa_y, marker='+', color='green')
plt.scatter(versicolor_x, versicolor_y, marker='_', color='red')
plt.show()


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

## Drop rest of the features and extract the target values
df = df.drop(['SepalWidthCm','PetalWidthCm'],axis=1)
Y = []
target = df['Species']

for val in target:
    if val == 'Iris-setosa':
        Y.append(-1)
    else:
        Y.append(1)

df = df.drop(['Species'],axis=1)
X = df.values.tolist() # TODO : dataframe's values() : return Numpy representation of dataframe

# Shuffle and split the data into training and test set
X, Y = shuffle(X, Y)
x_train = []
y_train = []
x_test = []
y_test = []

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9) # TODO : return data type is list

x_train = np.array(x_train) # TODO : why np.array() : create an array because original was list type, x_train shape is (90 x 2)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

y_train = y_train.reshape(90, 1) # TODO : why reshape? y_train shape was (1, 90), to match with x_train
y_test = y_test.reshape(10, 1)

train_f1 = x_train[:, 0]
train_f2 = x_train[:, 1]

train_f1 = train_f1.reshape(90,1)
train_f2 = train_f2.reshape(90,1)

w1 = np.zeros((90,1))
w2 = np.zeros((90,1))

epochs = 1
alpha = 0.0001

while(epochs < 10000):
    y = w1 * train_f1 + w2 * train_f2
    prod = y * y_train
    print(epochs)
    count = 0

    for val in prod:
        if val >= 1 : # y and pred have same sign, i.e. same class
            cost = 0
            w1 = w1 - alpha * (2 * (1/epochs) * w1)
            w2 = w2 - alpha * (2 * (1/epochs) * w2)

        else:
            cost = 1 - val
            w1 = w1 + alpha * (train_f1[count] * y_train[count] - 2 * 1/epochs * w1)
            w2 = w2 + alpha * (train_f2[count] * y_train[count] - 2 * 1/epochs * w2)

        count += 1
    epochs += 1


from sklearn.metrics import accuracy_score

# Clip the weights
index = list(range(10, 90))
w1 = np.delete(w1, index)
w2 = np.delete(w2, index)

w1 = w1.reshape(10, 1)
w2 = w2.reshape(10, 1)

# extract the test data features
test_f1 = x_test[:, 0]
test_f2 = x_test[:, 1]

test_f1 = test_f1.reshape(10, 1)
test_f2 = test_f1.reshape(10, 1)

# predict
y_pred = w1 * test_f1 + w2 * test_f2

predictions = []

for val in y_pred:
    if (val > 1):
        predictions.append(1)
    else:
        predictions.append(-1)


















