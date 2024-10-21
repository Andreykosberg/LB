import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import mpl_toolkits.mplot3d
from sklearn import datasets
from sklearn.cluster import KMeans

import scipy.stats as stats

sns.set()

cols = ["SEQN","age_group","RIDAGEYR","RIAGENDR","PAQ605","BMXBMI","LBXGLU","DIQ010","LBXGLT","LBXIN"]

df =  pd.read_csv('D:\\Учеба\\Нейронки\\ЛАБЫ\\Андрей\\ЛБ 2\\NHANES_age_prediction.csv',
                     delimiter=';',

                     )
print(df.shape)
print(df.head(10))
print(df.info())
print(df.describe())
print("_______")
print("Уникальные значения столбца 'age_group'")
print(df['age_group'].unique())
print("_______")
print("Количество значений в группах:")
print(df['age_group'].value_counts())
print("_______")

sns.violinplot(y='age_group', x='RIDAGEYR', data=df, inner='quartile')
plt.suptitle("Зависимость от возраста")
plt.show()

sns.violinplot(y='age_group', x='RIAGENDR', data=df, inner='quartile')
plt.suptitle("Зависимость от пола")
plt.show()

sns.violinplot(y='age_group', x='PAQ605', data=df, inner='quartile')
plt.suptitle("Зависимость от спорта")
plt.show()

sns.violinplot(y='age_group', x='BMXBMI', data=df, inner='quartile')
plt.suptitle("Зависимость от ИМТ")
plt.show()

sns.violinplot(y='age_group', x='LBXGLU', data=df, inner='quartile')
plt.suptitle("Зависимость от диабета")
plt.show()

sns.violinplot(y='age_group', x='DIQ010', data=df, inner='quartile')
plt.show()

sns.violinplot(y='age_group', x='LBXGLT', data=df, inner='quartile')
plt.suptitle("Зависимость от устных показаний ответчика")
plt.show()

sns.violinplot(y='age_group', x='LBXIN', data=df, inner='quartile')
plt.suptitle("Зависимость от уровня инсулина")
plt.show()

sns.pairplot(df, hue='age_group', markers='+')
plt.show()

plt.figure(figsize=(7,5))
sns.heatmap(df.corr(numeric_only = True), annot=True)
plt.show()

#Построение модели
print("___________\nПостроение модели\n___________")
X = df.drop(['age_group'], axis=1)
y = df['age_group']
print(f'X shape: {X.shape} | y shape: {y.shape} ')

y_mapped = y.map({'Adult': 0, 'Senior': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.30, shuffle=False)

stats.ttest_ind (a=y_train, b=y_test)

X_train, X_test, y_train, y_test = train_test_split(X, y_mapped, test_size=0.20, random_state=15, stratify=y_mapped)

stats.ttest_ind (a=y_train, b=y_test)

# создаем лист для тех моделей, которые будем изучать
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC(gamma='auto')))

results = []
model_names = []
for name, model in models:
  kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
  cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
  results.append(cv_results)
  model_names.append(name)
  print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#Обучение конкретной модели
sk_lda = LinearDiscriminantAnalysis(solver='eigen')

sk_lda.fit(X_train, y_train)
sk_lda_pred_res = sk_lda.predict(X_test)
sk_transformed = sk_lda.transform(X_train)
sk_lda_accuracy = accuracy_score(y_test, sk_lda_pred_res)

print(f'sk LDA accuracy: {sk_lda_accuracy}')
print(f'sk LDA prediction: {sk_lda_pred_res}')
print('sk transformed features', sk_transformed[:5].T, sep='\n')

from mlxtend.plotting import plot_decision_regions

# Создадим целевой вектор y_2d
# Преобразуем классы к классам 0, 1, 2 соответственно
#y_s = y_train.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
y_s = y_train
X_2d= X_train[['RIDAGEYR',	'BMXBMI']].values
#'RIAGENDR','PAQ605','LBXGLU','DIQ010','LBXGLT','LBXIN'
y_2d = y_s.values


sk_lda1 = LinearDiscriminantAnalysis(solver='eigen')
sk_lda1.fit(X_2d, y_2d)

plt.title('LDA surface with original features')
plot_decision_regions(
    	X=X_2d,
    	y=y_2d,
    	clf=sk_lda1)

plt.show()


df = df["age_group"].map({'Adult': 0, 'Senior': 1})

Xs = X_test[['RIDAGEYR',	'BMXBMI']].values
ys = df.values

sk_lda2 = LinearDiscriminantAnalysis(solver='eigen')
X1_lda = sk_lda2.fit(X, ys).transform(X)

Xs = X1_lda

X1_lda_train, X1_lda_test, y1_train, y1_test = train_test_split(X1_lda, ys, random_state=0)

sk_lda2.fit(X1_lda_train, y1_train)

plt.title('LDA surface with transformed features')
plot_decision_regions(X=X1_lda, y=ys, clf=sk_lda2)

predicted = sk_lda2.predict(X1_lda_test)

from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition

## Accuray e AUC
'''
Теперь смотрим метрики.
НА ТЕСТОВОМ ДАТАСЕТЕ
'''
accuracy = metrics.accuracy_score(y1_test, predicted)#Оценим точность классификации.

'''
Получим результат
'''

## Precision e Recall
recall = metrics.recall_score(y1_test, predicted, average="weighted")
precision = metrics.precision_score(y1_test, predicted, average="weighted")
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y1_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

# Часть 2

