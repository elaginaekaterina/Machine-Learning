# Обучение без учителя:
# кластеризация набора данных

import seaborn as sns

iris = sns.load_dataset('Iris')
print(iris.head())

sns.set()
sns.pairplot(iris, hue='species', height=1.5);
X_iris = iris.drop('species', axis=1)
X_iris.shape
Y_iris = iris['species']
Y_iris.shape

from sklearn.mixture import GaussianMixture # 1. Выбираем класс модели
model = GaussianMixture(n_components = 3, covariance_type = 'full') # 2. Создание экземпляра модели с гиперпараметрами
model.fit(X_iris) # 3. Обучение модели
y_gmm = model.predict(X_iris) # 4. Определение метки кластеров
iris['cluster'] = y_gmm # Добавление столбца 'cluster'
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", data = iris, hue = 'species', col = 'cluster', fit_reg = False)
