"""

reference : https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc

github : https://github.com/SSaishruthi/LogisticRegression_Vectorized_Implementation/blob/master/Logistic_Regression.ipynb

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b


def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result


def model_optimize(w, b, X, Y):
    m = X.shape[0]

    # Prediction
    final_result = sigmoid_activation(np.dot(w, X.T) + b)   # why transpose? To make same dimension for dot product
    Y_T = Y.T
    cost = (-1 / m) * (np.sum((Y_T * np.log(final_result)) + ((1 - Y_T) * (np.log(1 - final_result)))))
    #

    # Gradient calculation
    dw = (1 / m) * (np.dot(X.T, (final_result - Y.T).T))
    db = (1 / m) * (np.sum(final_result - Y.T))

    grads = {"dw": dw, "db": db}

    return grads, cost


# this is training part (not prediction)
def model_predict(w, b, X, Y, learning_rate, no_iterations):
    costs = []
    for i in range(no_iterations):
        #
        grads, cost = model_optimize(w, b, X, Y)
        #
        dw = grads["dw"]
        db = grads["db"]
        # weight update
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        #

        if (i % 100 == 0):
            costs.append(cost)
            # print("Cost after %i iteration is %f" %(i, cost))

    # final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}

    return coeff, gradient, costs


def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1

    return y_pred



def main():
    """
    data preparation

    """
    X_tr_arr = np.random.randn(20, 5)  # data points 100, feature 5
    X_ts_arr = np.random.randn(10, 5)

    y_tr_arr = np.concatenate([np.zeros(10), np.ones(10)])
    y_ts_arr = np.concatenate([np.ones(5), np.zeros(5)])

    """
    feature initialization
    """
    # Get number of features
    n_features = X_tr_arr.shape[1]
    print('Number of Features', n_features)
    w, b = weightInitialization(n_features)

    """
    calculate Sigmoid value, Log loss, gradient 
    """
    # Gradient Descent
    coeff, gradient, costs = model_predict(w, b, X_tr_arr, y_tr_arr, learning_rate=0.0001, no_iterations=4500)
    # Final prediction
    w = coeff["w"]
    b = coeff["b"]
    print('Optimized weights', w)
    print('Optimized intercept', b)

    """
    prediction
    """
    #
    final_train_pred = sigmoid_activation(np.dot(w, X_tr_arr.T) + b)
    final_test_pred = sigmoid_activation(np.dot(w, X_ts_arr.T) + b)
    #
    m_tr = X_tr_arr.shape[0]
    m_ts = X_ts_arr.shape[0]
    #
    y_tr_pred = predict(final_train_pred, m_tr)
    print('Training Accuracy', accuracy_score(y_tr_pred.T, y_tr_arr))
    #
    y_ts_pred = predict(final_test_pred, m_ts)
    print('Test Accuracy', accuracy_score(y_ts_pred.T, y_ts_arr))

    plt.plot(costs) # x-axis : number of datapoints, y-axis : cost value
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title('Cost reduction over time')
    plt.show()


if __name__ == "__main__":
    main()



