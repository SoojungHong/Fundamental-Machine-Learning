"""

logistic regression is to classify one of two classes

data : x, y (given x, class y) - y can be , for example, 0 or 1

train : given x, probability value, if probability value is > threshold, classify as 1, if not 0

# TODO : algorithm of logistic regression

Very good explanation :
https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc

Python implementation of Logistic Regression :
https://github.com/SSaishruthi/LogisticRegression_Vectorized_Implementation/blob/master/Logistic_Regression.ipynb

"""

import numpy as np


class LogisticRegressor:

    def loss(self, h, y):
        # log likelihood of any (x, y) pair is y*log(pred_y) + (1-y)*log(1-pred_y)
        # WRONG --> loss should be sum of loss then mean, return -(y * np.log(h) + (1 - y) * np.log(1 - h))
        m = len(y)
        J = (1.0 / m) * np.sum(-y * np.log(h) - (1.0 - y) * np.log(1.0 - h))


    def gradient(self, h, y, x):
        m = len(y)
        print('x : ', x)
        return (1/m) * np.sum( np.dot((h-y),x))  # h : (10 X 1), y : (10 X 1) , x : (10, 2)


    def sigmoid(self, y):
        s = 1/(1+ np.exp(-y))
        return s


    def fit(self, x, y):
        # initial w and b : Z = w * x + b
        theta = np.random.rand(x.shape[1]) # I need only x's column dimension (i.e. num of features) parameter
        b = 1 # no need because all parameters are in theta array
        lr = 0.001
        iter = 10
        cost = []

        for i in range(iter):

            Y = np.dot(x, theta) # x is (10 x 2) , theta is [1 x 2]
            # calculate Y_pred = sigmoid(Z)
            h = self.sigmoid(Y)

            # calculate loss between Y and Y_pred
            log_loss = self.loss(h,y)
            cost.append(log_loss)

            # calculate the derivative of L (loss) with respect to w (theta)
            theta_grad = self.gradient(h, y, x)

            # calculate the derivative of L (loss) with respect to b
            # TODO ==> This is done in gradient calculation with theta vector which contains all parameters

            # update parameters
            theta = theta - (lr * theta_grad)

        return theta


    def predict(self, a, theta):
        #y = self.theta * a + self.b
        pred = np.dot(a, theta) # a : (2 x 2) , theta : (2 x 1)
        # No need this for prediction score = self.sigmoid(pred)
        pred[pred > 0.5] = True
        pred[pred <= 0.5] = False
        print('pred : ', pred)


def get_data():

    x1 = np.random.randn(5,2) + 5
    x2 = np.random.randn(5,2) - 5
    X = np.concatenate([x1, x2])
    print(X)
    y = np.random.randn(10)

    # or
    y = np.concatenate([np.ones(5), np.zeros(5)], axis=0)
    print(y)

    return X, y



def main():

    # 1. get data
    X, y = get_data()

    # TODO : split data with train and test
    model = LogisticRegressor()


    # 2. fit model
    theta = model.fit(X, y)
    #       2.1 prediction function (theta, b) and Sigmoid function to calculate the prediction score
    #       2.2 given predicted score, calculate the loss (log loss)
    #       2.3 with loss, calculate the gradient (with respect to theta) and update the parameter 'theta'
    #       Do training 2.1 - 2.3 with iteration N

    # 3. predict model
    #test_data = 8 # example
    test_data = np.random.randn(5, 2)
    model.predict(test_data, theta)




if __name__ == "__main__":
    main()












