# Пример обучения без учителя:
# понижение размерности набора данных
# (используя метод главных компонент)
import seaborn as sns

iris = sns.load_dataset('Iris')
print(iris.head())

sns.set()
sns.pairplot(iris, hue='species', height=1.5);
X_iris = iris.drop('species', axis=1)
X_iris.shape
Y_iris = iris['species']
Y_iris.shape

from sklearn.decomposition import PCA # 1. выбираем класс модели
model = PCA(n_components=2) # 2. Создание экземпляра с гиперпараметрами
model.fit(X_iris) # 3. Обучение модели
X_2D = model.transform(X_iris) # 4. Преобразование данных в двумерные

#График результатов
iris['PCA1'] = X_2D[:, 0]
iris['PCA2'] = X_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue = 'species', data = iris, fit_reg = False)
