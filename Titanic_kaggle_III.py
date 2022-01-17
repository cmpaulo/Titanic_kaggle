import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model

# read a files with pandas
test = pd.read_csv("test.csv")
train = pd.read_csv("train.csv")

# using fuction of pandas map.
change = {'female': 1 , 'male': 0}
train['Sex_binario'] = train['Sex'].map(change)

embarked = {'S': 1 , 'C': 2, 'Q': 3}
train['Embarket_num'] = train['Embarked'].map(embarked)

test['Sex_binario'] = test['Sex'].map(change)
test['Embarket_num'] = test['Embarked'].map(embarked)

#chose the features with SeleckKBest
# Machine learning without Analyse Exploratoring Data
# quick fillna with -1 values
train = train.fillna(-1)
# test = train.loc[440:891,:].fillna(-1)
test = test.fillna(-1)

features = train.drop(['PassengerId','Ticket','Embarked','Cabin', 'Name' ,'Sex','Survived'], 1)

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

# select the features 'Pclass', 'Fare', 'Embarket_num', 'Parch', 'SibSp'
features = train.loc[:,['Sex_binario','Pclass', 'Fare', 'Embarket_num']]
X_test = test.loc[:,['Sex_binario', 'Pclass', 'Fare', 'Embarket_num']]

# Normalized data of features
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(features)  # Normalized data (train)
X_test_scale  = scaler.fit_transform(X_test)
labels_scale  = scaler.fit_transform(labels.values.reshape(-1, 1))
# training linear model.
lr = linear_model.LinearRegression()
lr.fit(X_train_scale, labels_scale)
pred = lr.predict(X_test_scale)
# the pred is a probability, and 0.5 or less is dead, and upper 0.5 is survived.
import numpy as np
p = []
for i in range(len(pred)):
    if pred[i][0] > 0.5:
        p.append(1)
    else:
        p.append(0)

surv  = pd.Series(np.array(p,dtype=np.int64), index=test['PassengerId'].values,name='Survived')

# surv.to_csv('model_III.csv', header=True)

