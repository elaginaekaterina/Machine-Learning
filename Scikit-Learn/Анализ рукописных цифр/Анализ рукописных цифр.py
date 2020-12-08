import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
from sklearn.datasets import load_digits
digits = load_digits()
digits.images.shape # трехмерный массив 1797 выборок, каждая состоить из сетки 8х8

fig, axes = plt.subplots(10, 10, figsize = (8, 8),
                         subplot_kw = {'xticks':[], 'yticks':[]},
                         gridspec_kw = dict(hspace = 0.1, wspace = 0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary',
              interpolation = 'nearest')
    ax.text(0.05, 0.05, str(digits.target[i]),
            transform = ax.transAxes, color = 'green')

X = digits.data
print(X.shape) # представляем массив пиксело длиной 64 элемента
y = digits.target
print(y.shape) # Итого получили 1797 выборок и 64 признака


# 1. Обучение без учителя: понижение размерности

# Преобразуем данные в двумерный вид
from sklearn.manifold import Isomap # Алгоритм обучения на базе многообразий
iso = Isomap(n_components = 2) # Понижение количества измерений до 2
iso.fit(digits.data)
data_projected = iso.transform(digits.data)
print(data_projected.shape)
# Посторим график данных
plt.scatter(data_projected[:, 0], data_projected[:, 1],
            c = digits.target, edgecolors = 'none', alpha = 0.5,
            cmap = plt.cm.get_cmap("Spectral", 10))
plt.colorbar(label = 'digit label', ticks = range(10))
plt.clim(-0.5, 9.5);


# 2. Классификация цифр

#разбиваем данные на обучающую и контрольную последовательности

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state = 0)
# После обучаем Гауссову наивную байесовскую модель
from  sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(Xtrain, ytrain)
y_model = model.predict(Xtest)
# оценивем точноть, сравнив значения контрольной последовательности с предсказанной
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model)) # 0.8333333333333334 точность более 80%
# построение матрицы различий, демонстрирующую
# частоты ошибочных классификаций

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, y_model)
sns.heatmap(mat, square = True, annot = True, cbar = False)
plt.xlabel('predicted value') # прогнозируемое значение
plt.ylabel('true value')      # настоящее значение

# Построение графика входных данных для оценки характеристик модели
# красный цвет - ошибки, зеленый - правильные метки
fig, axes = plt.subplots(10, 10, figsize = (8, 8),
                         subplot_kw = {'xticks':[], 'yticks':[]},
                         gridspec_kw = dict(hspace = 0.1, wspace = 0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap = 'binary',
              interpolation = 'nearest')
    ax.text(0.05, 0.05, str(y_model[i]),
            transform = ax.transAxes,
            color = 'green' if (ytest[i] == y_model[i]) else 'red')
plt.show()
