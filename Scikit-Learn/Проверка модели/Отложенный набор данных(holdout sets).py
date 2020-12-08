# Отложенные данные(Holdout sets)
import seaborn as sns

sns.set()

from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 1)
from sklearn.model_selection import train_test_split
# Разделяем данные: по 50% в каждом из наборов
X1, X2, y1, y2 = train_test_split(X, y, random_state = 0,
                                  train_size = 0.5)
# обучаем модель на одном из наборов данных
model.fit(X1, y1)
# оцениваем работу модели на другом наборе
from sklearn.metrics import accuracy_score
y2_model = model.predict(X2)
print(accuracy_score(y2, y2_model)) # 0.9066666666666666 точность 90%
