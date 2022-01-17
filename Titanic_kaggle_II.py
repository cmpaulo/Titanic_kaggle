from numpy.random import f
import pandas as pd
from pandas_profiling.profile_report import ProfileReport
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")
print(' ')

def change_str(value):
    if value == 'female':
        return 1
    else:
        return 0

def change_embarked(value):
    if value == 'S':
        return 1
    elif value == 'C':
        return 2
    else:
        return 3

# train['Sex_binario'] = train['Sex'].map(change_str)
change = {'female':1 , 'male': 0}
train['Sex_binario'] = train['Sex'].map(change)

print(train['Sex_binario'])
# train['Embarket_num'] = train['Embarked'].map(change_embarked)

# test['Sex_binario'] = test['Sex'].map(change_str)
# test['Embarket_num'] = test['Embarked'].map(change_embarked)

exit()
## ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
##        'Ticket', 'Fare', 'Cabin', 'Embarked']
# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

#Escolhendo as melhores features com Kbest
train = train.fillna(-1)
test = test.fillna(-1)
# teste de variaveis
# import pandas_profiling as pdff
# profile = pdff.ProfileReport(train)
# rejected_features= list(profile.get_rejected_variables()) 
# print(rejected_features)
# X_drop= train.drop(rejected_features,axis=1)
# X_drop.shape
# print(X_drop)

# train.set_index(train['PassengerId'],inplace=True)
# test.set_index(test['PassengerId'], inplace=True)

features = train.drop(['PassengerId','Ticket','Embarked','Cabin', 'Name' ,'Sex','Survived'], 1)
test_sel = test.drop(['PassengerId','Ticket','Embarked','Cabin', 'Name' ,'Sex'], 1)

#  teste de features
labels = train['Survived']
features_list = list(features.keys())
k_best_features = SelectKBest(k='all')
k_best_features.fit_transform(features, labels)
k_best_features_scores = k_best_features.scores_
raw_pairs = zip(features_list, k_best_features_scores)
ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))

k_best_features_final = dict(ordered_pairs)
best_features = k_best_features_final.keys()
print ('')
print ("Best features:")
print (k_best_features_final)
print (best_features)
# seleciona melhores features
features = train.loc[:,['Sex_binario', 'Pclass', 'Fare', 'Embarket_num', 'Parch', 'SibSp']]
X_test = test.loc[:,['Sex_binario', 'Pclass', 'Fare', 'Embarket_num', 'Parch', 'SibSp']]
print(X_test.head())
# Normalizando os dados de entrada(features)
# Gerando o novo padrão
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(features)  # Normalizando os dados de entrada(treinamento)
X_test_scale  = scaler.fit_transform(X_test)
labels_scale  = scaler.fit_transform(labels.values.reshape(-1, 1))

# treinamento usando regressão linear
lr = linear_model.LinearRegression()
lr.fit(X_train_scale, labels_scale)
pred = lr.predict(X_test_scale)
import numpy as np
p = []
for i in range(len(pred)):
    if pred[i][0] > 0.5:
        p.append(1)
    else:
        p.append(0)

surv  = pd.Series(np.array(p,dtype=np.int), index=test['PassengerId'].values,name='Survived')
print(surv)

# surv.to_csv('model_II.csv', header=True)

