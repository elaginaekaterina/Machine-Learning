import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns
sns.set()

from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)
print(plt.scatter(X[:, 0],X[:, 1], c=y, s=50, cmap='autumn'))

xfit = np.linspace(-1, 3.5)
print(plt.scatter(X[:, 0],X[:, 1], c=y, s=50, cmap='autumn'))
print(plt.plot([0.6],[2.1],'x', color='red', markeredgewidth=2, markersize=10))

for m, b in [(1, 0.65),(0.5,1.6),(-0.2,2.9)]:
    print(plt.plot(xfit, m*xfit+b, '-k'))
print(plt.xlim(-1, 3.5))

xfit = np.linspace(-1, 3.5)
print(plt.scatter(X[:, 0],X[:, 1], c=y, s=50, cmap='autumn'))
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m*xfit+b
    print(plt.plot(xfit, yfit, '-k'))
    print(plt.fill_between(xfit, yfit-d, yfit+d, egecolor='none',
                     color='#AAAAAA', alpha=0.4))
print(plt.xlim(-1, 3.5))

# Аппроксимация методом опорных векторов
from sklearn.svm import SVC # Классификатор на основе метода опорных векторов
model = SVC(kernel='linear', C=1E10)
model.fit(X,y)

def plot_svc_decision_function(model, ax=None,plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

# Создание координатной сетки для оценки модели
x = np.linspace(xlim[0], xlim[1], 30)
y = np.linspace(ylim[0], ylim[1], 30)
Y, x = np.meshgrid(y,x)
xy = np.vstack([X.ravel(),Y.ravel()]).T
P = model.decision_function(xy).reshape(X.shape)

# Границы принятия решений и отступы
ax.contour(X, Y, P, colors='k',
           levels = [-1, 0, 1], alpha = 0.5,
           linestyles = ['--', '-', '--'])

# Опорне векторы
if plot_support:
    ax.scatter(model.support_vectors_[:,0],
               model.support_vectors_[:,1],
               s = 300, linewidth = 1, facecolors = 'none')
ax.set_xlim(xlim)
ax.set_ylim(ylim)

print(plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn'))
print(plot_svc_decision_function(model));

model.support_vectors_

def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.60)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C = 1E10)
    model.fit(X, y)

    ax = ax or plt.gca()
    ax.scatter(X[:,0], X[:, 1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N={0}'.format(N))

from sklearn.datasets.samples_generator import make_circles
X, y = make_circles(100, factor=.1, noise=.1)

clf = SVC(kernel='linear').fit(X,y)

plt.scatter(X[:,0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False);
r = np.exp(-(X**2).sum(1))

# SVM с использованием ядерного преобразования
clf = SVC(kernel='rbf', C=1E6)
clf.fit(X,y)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],
                  s=300, lw=1, facecolors='none');
