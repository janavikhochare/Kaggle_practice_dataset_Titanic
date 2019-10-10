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

#data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]


#data.info()
data.fillna(data['Age'].mean(), inplace=True)
data.fillna(data['Fare'].mean(), inplace=True)
#data.info()

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
#test =test.fillna(test.median(), inplace=True)
print(test.info())
print("TEST ======================================")
test.info()

train = data
X_train = pd.DataFrame(train.drop(['Survived'], axis=1))
Y_train = pd.DataFrame(train['Survived'])

X_test = pd.DataFrame(test)



# Random Forests

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier


# random_forest = RandomForestClassifier(criterion='gini',
#                              n_estimators=1000,
#                              min_samples_split=10,
#                              min_samples_leaf=1,
#                              max_features='auto',
#                              oob_score=True,
#                              random_state=1,
#                              n_jobs=-1)

seed= 42
random_forest =RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=5, min_samples_split=2,
                           min_samples_leaf=1, max_features='auto',bootstrap=False, oob_score=False,
                           n_jobs=1, random_state=seed,verbose=0)

random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
Y_pred= Y_pred.astype(int)
print(Y_pred)

result_train = random_forest.score(X_train, Y_train)
result_val = cross_val_score(random_forest,X_train, Y_train, cv=5).mean()

print('taring score = %s , while validation score = %s' %(result_train , result_val))

#pred= pd.DataFrame(Y_pred)


test1=pd.read_csv("test.csv")

submission = pd.DataFrame({'PassengerID': test1['PassengerId'],
                           'Survived': Y_pred
                           })
submission.to_csv('Titanic_Submission.csv', index=False)
print("done")