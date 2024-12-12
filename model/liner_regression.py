import numpy as np

class linear_regression:

    def __init__(self,lr = 0.001,iterations = 1000):
        self.lr = lr
        self.iternations = iterations
        self.w = None
        self.b = None

    def fit(self,X,y):
        # m - number of samples
        # n - number of features
        m,n = X.shape()
        self.w = np.zeros(n)
        self.b = 0

        for i in range(self.iternations):
            y_pred = np.dot(X,self.w) + self.b
            dw = np.dot((X.T,y_pred - y))/m
            db = np.sum((y_pred - y))/m

            self.w -= self.lr*dw
            self.b -= self.lr*db

    def predict(self,X):
        y_pred = np.dot(X,self.w) + self.b
        return y_pred