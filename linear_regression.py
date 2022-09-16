"""
linear regression

Assumption is
1. given data point i (x, y)
2. relationship between y is linear => Y = aX + b
3. model training is to minimize the Y_l - Y_p  (Y_l : ground truth, Y_p : predicted y)
   initialize a = a_0, b = b_0
4. optimization is done with stochastic gradient descent (SGD)
   a = a - lambda * derivative(Y_l - Y_p)
   lambda = 0.001
   derivative (i.e. slope) = (y_l - y_p)/(x_l - x_p)  # TODO : how derivative is calculated? slope of two points

5. Do 3-4 until the error (Y_l - Y_p) is converged (i.e. not changing)
6. Lastly, found a and b is the trained model's parameters

"""


import numpy as np
import matplotlib.pyplot as plt

class LinearRegressor:  # because it is class, no need ()? only for init value?
    def __init__(self):
        self.m = 0
        self.b = 0


    def _mean(self, X):
        return np.mean(X)


    def sumOfSquare(self, e):
        SS = np.sum(e**2)
        return SS


    def SS(self, Y, Y_hat): # sum of loss
        return sum((Y - Y_hat) ** 2)


    # The most commonly used loss function for Linear Regression is Least Squared Error,
    # and its cost function is also known as Mean Squared Error(MSE).
    def loss(self, Y_l, Y_p):
        l = Y_l - Y_p


    def calc_deri(self, x1, y1, x2, y2):
        deri = (y2 - y1)/(x2 - x1)


    # in linear regression, fit() is finding m & b
    def fit(self, X, Y):  # X is x value, Y is y value

        # update m, b
        self.m = sum((X - np.mean(X)) * (Y - np.mean(Y))) / sum((X - np.mean(X)) ** 2)  # TODO : I am not sure this is correct
        # self.m = self.m - alpha * (sum( (y_pred - y) * x )) # I think, this might be correct
        self.b = np.mean(Y) - self.m * np.mean(X)  # y = m*x + b  , thus b = y - m*x


    # linear regression's Coefficient R2  # TODO : what is R2 coefficient
    # R2 = (coefficient of determination) R2  provides a measure of  how well the observed outcome are replicated by model
    # R2 = 1 - (SSres / SStot)
    # SStot = sum ( (Y - mean(Y))** 2 )
    # SSres = sum ( (Y - Y_predict) ** 2 )
    def coef(self, Y, Y_hat):
        return 1 - self.SS(Y, Y_hat) / self.SS(Y, np.mean(Y))


    def predict(self, X):
        return self.m * X + self.b


def get_data():
    X = np.linspace(0, 10, 10)
    print(' X : ', X)
    return X


def init_model(X):
    m, b = 3, -2
    Y = m * X + b + 0.1 * np.random.randn(X.shape[0])

    lr = LinearRegressor()

    return m, b, lr



def main():

    # get data
    X = get_data()

    # split data into train and test
    #train_data, test_data = split_train_test(X)


    # train

   # X = np.linspace(0, 10, 10)
    #m, b = 3, -2
    #Y = m * X + b + 0.1 * np.random.randn(X.shape[0])
    #print(np.random.randn(X.shape[0]))

    lr = LinearRegressor()
    X = np.linspace(0, 10, 10)
    m, b = 3, -2
    Y = m * X + b + 0.1 * np.random.randn(X.shape[0])
    lr.fit(X, Y)

    Y_hat = lr.predict(X)
    R2 = lr.coef(Y, Y_hat)

    print(lr.m, lr.b)
    print(R2)

    # visualization
    m, b = np.polyfit(X, Y, 1)

    plt.plot(X, Y, 'yo', X, m * X + b, '--k')
    plt.show()


if __name__ == "__main__":
    main()