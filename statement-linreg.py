"""
Created on Thu May 30 21:42:05 2019

@author: daria
"""
'''
Проведите предобработку:
Приведите тексты к нижнему регистру (text.lower()).
Замените все, кроме букв и цифр, на пробелы — это облегчит дальнейшее разделение 
текста на слова. Для такой замены в строке text подходит следующий вызов: 
re.sub('[^a-zA-Z0-9]', ' ', text). Также можно воспользоваться методом replace 
у DataFrame, чтобы сразу преобразовать все тексты.
Примените TfidfVectorizer для преобразования текстов в векторы признаков. 
Оставьте только те слова, которые встречаются хотя бы в 5 объектах (параметр 
min_df у TfidfVectorizer).
Замените пропуски в столбцах LocationNormalized и ContractTime на специальную 
строку 'nan'. Код для этого был приведен выше.
Примените DictVectorizer для получения one-hot-кодирования признаков 
LocationNormalized и ContractTime.
Объедините все полученные признаки в одну матрицу "объекты-признаки". 
Обратите внимание, что матрицы для текстов и категориальных признаков являются 
разреженными. Для объединения их столбцов нужно воспользоваться функцией 
scipy.sparse.hstack.
Обучите гребневую регрессию с параметрами alpha=1 и random_state=241. 
Целевая переменная записана в столбце SalaryNormalized.
Постройте прогнозы для двух примеров из файла salary-test-mini.csv. 
Значения полученных прогнозов являются ответом на задание. Укажите их через пробел.
'''
# %%
import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

# %%
data = pandas.read_csv('salary-train.csv')
data['FullDescription'] = data['FullDescription'].map(lambda text: text.lower())
data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
#%%
vectorizer = TfidfVectorizer(min_df = 5)
X_train = vectorizer.fit_transform(data['FullDescription'])
data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)
dictVectorizer = DictVectorizer()
X_train_one_hot = dictVectorizer.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_train = hstack([X_train, X_train_one_hot])
y_train = data['SalaryNormalized']
#%%
model = Ridge(alpha=1, random_state = 241)
model.fit(X_train, y_train)
#%%
data = pandas.read_csv('data/salary-test-mini.csv')
data['FullDescription'] = data['FullDescription'].map(lambda text: text.lower())
data['FullDescription'] = data['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)
data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)
X_test = vectorizer.transform(data['FullDescription'])
X_test_one_hot = dictVectorizer.transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test, X_test_one_hot])
y_test = model.predict(X_test)
y_test