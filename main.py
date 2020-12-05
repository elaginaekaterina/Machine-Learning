import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# Выбор класса модели
from sklearn.linear_model import LinearRegression

iris = sns.load_dataset('Iris')
print(iris.head())

sns.set()
sns.pairplot(iris, hue='species', height=1.5);
X_iris = iris.drop('species', axis=1)
X_iris.shape
Y_iris = iris['species']
Y_iris.shape
# Пример обучения с учителем: простая регрессия

rng = np.random.RandomState(42)
x = 10 * rng.rand(50)
y = 2 * x - 1 + rng.randn(50)
plt.scatter(x, y)
plt.show()


# Выбор гиперпараметров модели
model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
# model
# Формирование из данных матриц признаков и целевого вектора
X = x[:, np.newaxis]
print(X.shape)

# Обучение модели на наших данных
model.fit(X, y)
print(model.coef_)  # угловой коэффициент
print(model.intercept_)  # точка пересечения с осью координат
# Предсказание меток для новых данных
xfit = np.linspace(-1, 11)  # преобразуем матрицу признаков к виду [n_samples, n_faetures]
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
# Вывод результатов
plt.scatter(x, y)  # исходные данные
plt.plot(xfit, yfit)  # обученная модель
