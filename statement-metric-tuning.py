#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 25 22:51:43 2019

@author: daria
"""
'''
Приведите признаки в выборке к одному масштабу при помощи функции sklearn.preprocessing.scale.
Переберите разные варианты параметра метрики p по сетке от 1 до 10 с таким шагом, 
чтобы всего было протестировано 200 вариантов (используйте функцию numpy.linspace). 
Используйте KNeighborsRegressor с n_neighbors=5 и weights='distance' — данный параметр 
добавляет в алгоритм веса, зависящие от расстояния до ближайших соседей. В качестве 
метрики качества используйте среднеквадратичную ошибку (параметр scoring='mean_squared_error' 
у cross_val_score; при использовании библиотеки scikit-learn версии 0.18.1 и выше необходимо 
указывать scoring='neg_mean_squared_error'). Качество оценивайте, как и в предыдущем задании, 
с помощью кросс-валидации по 5 блокам с random_state = 42, не забудьте включить перемешивание 
выборки (shuffle=True).
Определите, при каком p качество на кросс-валидации оказалось оптимальным. Обратите внимание, 
что cross_val_score возвращает массив показателей качества по блокам; необходимо максимизировать
среднее этих показателей. Это значение параметра и будет ответом на задачу.
'''
#%%
import pandas
from numpy import linspace
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import datasets
from sklearn.preprocessing import scale

#%%
data = datasets.load_boston()

#%%
X = scale(data.data)
y = data.target

#%%
kf = KFold(n_splits = 5, shuffle=True, random_state = 42)
scores = list()
for p in linspace(1, 10, 200):
    model = KNeighborsRegressor(n_neighbors = 5, weights='distance',p=p)
    scores.append(cross_val_score(model, X, y, cv = kf, scoring = 'neg_mean_squared_error'))
pandas.DataFrame(scores,linspace(1, 10, 200)).max(axis = 1).sort_values(ascending = False).head(1)