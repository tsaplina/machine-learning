"""
Created on Tue May 28 22:31:56 2019

@author: daria
"""
'''
Вычислите TF-IDF-признаки для всех текстов. Обратите внимание, что в этом задании 
мы предлагаем вам вычислить TF-IDF по всем данным. При таком подходе получается, 
что признаки на обучающем множестве используют информацию из тестовой выборки — 
но такая ситуация вполне законна, поскольку мы не используем значения целевой 
переменной из теста. На практике нередко встречаются ситуации, когда признаки 
объектов тестовой выборки известны на момент обучения, и поэтому можно ими 
пользоваться при обучении алгоритма.
Подберите минимальный лучший параметр C из множества 
[10^-5, 10^-4, ... 10^4, 10^5] для SVM с линейным ядром (kernel='linear') 
при помощи кросс-валидации по 5 блокам. Укажите параметр random_state=241 и 
для SVM, и для KFold. В качестве меры качества используйте долю верных ответов 
(accuracy).
Обучите SVM по всей выборке с оптимальным параметром C, найденным на предыдущем 
шаге.
Найдите 10 слов с наибольшим абсолютным значением веса (веса хранятся в поле 
coef_ у svm.SVC). Они являются ответом на это задание. Укажите эти слова через 
запятую или пробел, в нижнем регистре, в лексикографическом порядке.
'''

#%%
import numpy as np
import pandas
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import KFold

#%%
df = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
x = df.data
y = df.target
#%%
vectorizer = TfidfVectorizer()
vectorizer.fit_transform(x)
#%%
val = KFold(n_splits=5, shuffle=True, random_state=241)
model = SVC(kernel='linear', random_state=241)
#%%
grid = {'C': np.power(10.0, np.arange(-5, 6))}
gs = GridSearchCV(model, grid, scoring='accuracy', cv=val)
gs.fit(vectorizer.transform(x), y)
C = gs.best_params_['C']
#%%
model = SVC(kernel='linear', random_state=241, C=C)
model.fit(vectorizer.transform(x), y)
words = vectorizer.get_feature_names()
coef = pandas.DataFrame(model.coef_.data, model.coef_.indices)
top_words = coef[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])
#%%
np.sort(top_words)