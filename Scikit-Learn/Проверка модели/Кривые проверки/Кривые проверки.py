# Validation curve

import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


# создание данных для обучения модели
def make_data(N, err=1.0, rseed=1):
    # создание случайных выборок данных
    rng = np.random.RandomState(rseed)
    X = rng.rand(N, 1) ** 2
    y = 10 - 1. / (X.ravel() + 0.1)
    if err > 0:
        y += err * rng.randn(N)
    return X, y


X, y = make_data(40)

X_test = np.linspace(-0.1, 1.1, 500)[:, None]

plt.scatter(X.ravel(), y, color='black')
axis = plt.axis()
for degree in [1, 3, 5]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test, label='degree = {0}'.format(degree))
print(plt.xlim(-0.1, 1.0))
print(plt.ylim(-2, 12))
print(plt.legend(loc='best'))


from sklearn.model_selection import validation_curve

degree = np.arange(0, 21)
train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          'polynomialfeatures__degree',
                                          degree, cv=7)
print(()).plot(degree, np.median(train_score, 1), color='blue',
               label='training score')  # оценка обучения
print(plt.plot(degree, np.median(val_score, 1), color='red',
               label='validation score'))  # оценка проверки
print(plt.legend(loc='best'))
print(plt.ylim(0, 1))
print(plt.xlabel('degree'))  # степень
print(plt.ylabel('score'))  # оценка

print(plt.scatter(X.ravel(), y))
lim = plt.axis()
y_test = PolynomialRegression(3).fit(X, y).predict(X_test)
print(plt.plot(X_test.ravel(), y_test))
print(plt.axis(lim))
