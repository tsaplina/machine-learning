"""
Created on Thu May 30 14:12:01 2019

@author: daria
"""
'''
Загрузите файл classification.csv. В нем записаны истинные классы объектов выборки 
(колонка true) и ответы некоторого классификатора (колонка pred).
Заполните таблицу ошибок классификации. Для этого подсчитайте величины TP, FP, 
FN и TN согласно их определениям. Например, FP — это количество объектов, имеющих 
класс 0, но отнесенных алгоритмом к классу 1. Ответ в данном вопросе — четыре числа 
через пробел.
Посчитайте основные метрики качества классификатора:
    Accuracy (доля верно угаданных) — sklearn.metrics.accuracy_score
    Precision (точность) — sklearn.metrics.precision_score
    Recall (полнота) — sklearn.metrics.recall_score
    F-мера — sklearn.metrics.f1_score
Имеется четыре обученных классификатора. В файле scores.csv записаны истинные классы 
и значения степени принадлежности положительному классу для каждого классификатора 
на некоторой выборке:
    для логистической регрессии — вероятность положительного класса (колонка score_logreg),
    для SVM — отступ от разделяющей поверхности (колонка score_svm),
    для метрического алгоритма — взвешенная сумма классов соседей (колонка score_knn),
    для решающего дерева — доля положительных объектов в листе (колонка score_tree).
Посчитайте площадь под ROC-кривой для каждого классификатора. Какой классификатор 
имеет наибольшее значение метрики AUC-ROC (укажите название столбца)? Воспользуйтесь 
функцией sklearn.metrics.roc_auc_score.
Какой классификатор достигает наибольшей точности (Precision) при полноте (Recall) 
не менее 70% ? 
'''
#%%
import pandas
import sklearn.metrics

#%%
data = pandas.read_csv('classification.csv')
data['true'][0]
data['pred'][0]
#%%
tp = 0
fp = 0
fn = 0
tn = 0
for i in range(0,len(data)):
    if data['true'][i] == 1 and data['pred'][i] == 1: tp += 1
    if data['true'][i] == 0 and data['pred'][i] == 0: tn += 1
    if data['true'][i] == 0 and data['pred'][i] == 1: fp += 1
    if data['true'][i] == 1 and data['pred'][i] == 0: fn += 1
#%%
tp,fp,fn,tn
#%%
sklearn.metrics.accuracy_score(data['true'], data['pred'])
sklearn.metrics.precision_score(data['true'], data['pred'])
sklearn.metrics.recall_score(data['true'], data['pred'])
sklearn.metrics.f1_score(data['true'], data['pred'])
print(sklearn.metrics.classification_report(data['true'], data['pred']))
#%%
data = pandas.read_csv('scores.csv')
scores = {}
for x in data.columns[1:]:
    scores[x] = sklearn.metrics.roc_auc_score(data['true'], data[x])
scores
#%%
scores = {}
for x in data.columns[1:]:
    curve = sklearn.metrics.precision_recall_curve(data['true'], data[x])
    df = pandas.DataFrame({'precision': curve[0], 'recall': curve[1]})
    scores[x] = df[df['recall'] >= 0.7]['precision'].max()
