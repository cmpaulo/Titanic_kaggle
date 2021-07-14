import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

from sklearn.ensemble import RandomForestClassifier
modelo = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
variaveis = ['Sex_binario','Age']

def change_str(value):
    if value == 'female':
        return 1
    else:
        return 0
train['Sex_binario'] = train['Sex'].map(change_str)

# ntrain = train.loc[:,variaveis].dropna()
X = train[variaveis].fillna(-1)
y = train['Survived']
modelo.fit(X,y)

# conjunto de test
test['Sex_binario'] = test['Sex'].map(change_str)
X_prev = test[variaveis].fillna(-1)

p = modelo.predict(X_prev)

surv = pd.Series(p, index=test['PassengerId'], name='Survived')
surv.to_csv('model_I.csv', header=True)
