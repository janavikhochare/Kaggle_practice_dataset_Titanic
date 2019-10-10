import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("train.csv")
#data.info()
dict_of_title = {
    "Dr": "Officer",
    "Rev": "Officer",
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Jonkheer": "Royal",
    "Lady" : "Royal",
    "Don": "Royal",
    "Sir" : "Royal",
    "the Countess":"Royal"
}

def make_titles(data):
    data["Title"] = data["Name"].map(lambda s:s.split(",")[1].split(".")[0].strip())
    data["Title"] = data["Title"].map(dict_of_title)
    return data

data = make_titles(data)

data['Family Size']=data['SibSp']+data['Parch']

p = {1:'1st',2:'2nd',3:'3rd'}
data['Pclass'] = data['Pclass'].map(p)

df_new_onehot = data[['Sex','Embarked','Pclass','Title']]

c = ['Sex', 'Embarked','Pclass','Title']

for i in range(len(c)):
    a1 = pd.get_dummies(df_new_onehot, c[i])

data = data.join(a1)

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

df1 =data[['Age','Fare']]
df1 = normalize(df1)
data= data.drop(['Embarked','Sex','Ticket','Cabin','Name','Age','SibSp','Parch','Fare','Pclass','Title'],axis=1)

data =data.join(df1)
data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
data.info()

features = data.select_dtypes(include=[np.number])
print(features.dtypes)
corr =features.corr()
print(corr['Survived'].sort_values(ascending=False))
print(corr)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=1, square=True)
plt.show()


#on test data now
test=pd.read_csv("test.csv")
test = make_titles(test)

test['Family Size']=test['SibSp']+test['Parch']
p = {1:'1st',2:'2nd',3:'3rd'}
test['Pclass'] = test['Pclass'].map(p)

df_new_onehot1 = test[['Sex','Embarked','Pclass','Title']]

c = ['Sex', 'Embarked','Pclass','Title']

for i in range(len(c)):
    a2 = pd.get_dummies(df_new_onehot1, c[i])

test = test.join(a1)
df2 =test[['Age','Fare']]

df2 = normalize(df2)
test= test.drop(['Embarked','Sex','Ticket','Cabin','Name','Age','SibSp','Parch','Fare','Pclass','Title'],axis=1)

test =test.join(df1)
#test = test[~test.isin([np.nan, np.inf, -np.inf]).any(1)]
test.fillna(test['Fare'].mean(), inplace=True)
test.fillna(test['Age'].mean(), inplace=True)
print("TEST ======================================")
test.info()

train = data
X_train = pd.DataFrame(train.drop(['Survived'], axis=1))
Y_train = pd.DataFrame(train['Survived'])

X_test = pd.DataFrame(test)


# from sklearn.ensemble import RandomForestClassifier
# random_forest = RandomForestClassifier(n_estimators=100)
# #random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, max_features='sqrt', min_samples_split=5)
#
# random_forest.fit(X_train, Y_train)
#
#
# Y_pred_1 = random_forest.predict(X_test)

import lightgbm as lgb
from sklearn.metrics import mean_squared_error

lgb_train = lgb.Dataset(X_train, Y_train)
#lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train)

params = {'task':'train', 'boosting_type':'gbdt', 'objective':'regression',
              'metric': {'rmse'}, 'num_leaves': 8, 'learning_rate': 0.05,
              'feature_fraction': 0.8, 'max_depth': 7, 'verbose': 0,
              'num_boost_round':25000, #'early_stopping_rounds':100,
           'nthread':-1}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=200)
                #early_stopping_rounds=5)

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
print(y_pred)
pred= pd.DataFrame(y_pred)
sub = pd.read_csv("gender_submission.csv")
sub['Survived']=pred
print(sub.head())
sub.to_csv('gender_submission.csv')
#print(random_forest.score(X_train, Y_train))