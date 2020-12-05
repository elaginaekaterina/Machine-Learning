# Обучение с учителем: классификация набора данных
import seaborn as sns


# Делим данные на обучающий и тестовый последовательности

iris = sns.load_dataset('Iris')
print(iris.head())

sns.set()
sns.pairplot(iris, hue='species', height=1.5);
X_iris = iris.drop('species', axis=1)
X_iris.shape
Y_iris = iris['species']
Y_iris.shape

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, Y_iris, random_state = 1)

from sklearn.naive_bayes import GaussianNB # 1. Выбираем класс модели
model = GaussianNB() # 2. Создаем экземпляр модели
model.fit(Xtrain, ytrain) # 3. Обучение модели на данных
y_model = model.predict(Xtest) # 4. Предсказание значений для новых данных

#Выяснение истинности предсказанных меток
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest, y_model)) # 0.9736842105263158 точность 97%
