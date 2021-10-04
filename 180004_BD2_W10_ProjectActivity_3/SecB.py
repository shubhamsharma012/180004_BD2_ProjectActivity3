#!/usr/bin/env python
# coding: utf-8

# In[21]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# ### Reading and Exploring data

# In[22]:


train_df = pd.read_csv(r"Titanic.csv")



# In[23]:


print(train_df.columns.values)


# In[24]:


# preview the data
train_df.head()


# In[25]:


train_df.tail()


# ### Data Analysis

# In[26]:


train_df.info()
print('_'*40)



# In[27]:


train_df.describe()


# In[28]:


train_df.describe(include=['O'])


# ### Analyse by pivot table

# In[29]:


train_df[['pclass','survived']].groupby(['pclass'],as_index=False).mean()


# In[30]:


train_df[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[31]:


train_df[["sex", "survived"]].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[32]:


train_df[["sibsp", "survived"]].groupby(['sibsp'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[33]:


train_df[["parch", "survived"]].groupby(['parch'], as_index=False).mean().sort_values(by='parch',ascending=True)


# In[34]:


train_df[['fare']].describe()


# ### Analyse by visualising Data

# In[35]:


g = sns.FacetGrid(train_df, col='pclass')
g.map(plt.hist,'fare', bins=10)


# In[36]:


g = sns.FacetGrid(train_df, col='survived')
g.map(plt.hist, 'age', bins=20)


# In[38]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='survived', row='pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();


# In[40]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'pclass', 'survived','sex', palette='deep')
grid.add_legend()


# In[42]:


grid = sns.FacetGrid(train_df, row='sex', col='survived', size=2.2, aspect=1.6)
grid.map(plt.hist,'age',bins=20)
grid.add_legend();


# In[43]:


grid = sns.FacetGrid(train_df, row='embarked', col='survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'sex', 'fare', alpha=.5, ci=None)
grid.add_legend()


# ### Data wrangling

# In[44]:


print("Before", train_df.shape)

train_df = train_df.drop(['ticket', 'cabin'], axis=1)


"After", train_df.shape


# ### Creating new feature extracting from existing

# In[46]:


for dataset in train_df:
    dataset['Title'] = dataset.name.str.extract(' ([A-Za-z]+)\.', expand=False)


pd.crosstab(train_df['Title'], train_df['sex'])


# In[47]:


for dataset in train_df:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'survived']].groupby(['Title'], as_index=False).mean()


# In[48]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


# In[49]:


train_df['Title'].value_counts()


# In[50]:


print(train_df.head())


# In[51]:


train_df = train_df.drop(['name', 'passengerId'], axis=1)
train_df.shape


# In[52]:


train_df.shape
train_df.head()


# ### Converting a categorical feature

# In[53]:


for dataset in train_df:
    dataset['sex'] = dataset['sex'].map( {'female': 1, 'male': 0}).astype(int)

print(train_df.head())




# ### Completing a numerical continuous feature

# In[55]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='pclass', col='sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend()


# In[56]:


guess_ages = np.zeros((2,3))
guess_ages


# In[61]:


for dataset in train_df:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['sex'] == i) & (dataset['pclass'] == j+1)]['age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.age.isnull()) & (dataset.sex == i) & (dataset.pclass == j+1),'age'] = guess_ages[i,j]

    dataset['age'] = dataset['age'].astype(int)

train_df.head()


# In[62]:


train_df['AgeBand'] = pd.cut(train_df['age'], 5)
train_df[['AgeBand', 'survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[63]:


for dataset in combine:    
    dataset.loc[ dataset['age'] <= 16, 'age'] = 0
    dataset.loc[(dataset['age'] > 16) & (dataset['age'] <= 32), 'age'] = 1
    dataset.loc[(dataset['age'] > 32) & (dataset['age'] <= 48), 'age'] = 2
    dataset.loc[(dataset['age'] > 48) & (dataset['age'] <= 64), 'age'] = 3
    dataset.loc[ dataset['age'] > 64, 'age']
train_df.head()


# In[64]:


train_df = train_df.drop(['AgeBand'], axis=1)
train_df.head()



# In[65]:


train_df.head()




# ### Create new feature combining existing features

# In[67]:


for dataset in train_df:
    dataset['FamilySize'] = dataset['sibsp'] + dataset['parch'] + 1


# In[68]:


train_df.head()
train_df[['FamilySize', 'survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='survived', ascending=False)


# In[69]:


train_df.info()


# In[70]:


train_df['FamilySize'].value_counts()


# In[71]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1


# In[72]:


train_df.head()
train_df[['IsAlone', 'survived']].groupby(['IsAlone'], as_index=False).mean()


# In[73]:


dropped_one = train_df['parch']
dropped_two = train_df['sibsp']
dropped_three = train_df['FamilySize']
dropped_one




# In[79]:




train_df.head()


# In[82]:


for dataset in train_df:
    dataset['age*Class'] = dataset.age * dataset.pclass

train_df.loc[:, ['age*Class', 'age', 'pclass']].head(10)


# In[83]:


train_df['age*Class'].value_counts()


# ### Completing a categorical feature

# In[84]:


freq_port = train_df['embarked'].dropna().mode()[0]
freq_port


# In[85]:


for dataset in train_df:
    dataset['embarked'] = dataset['embarked'].fillna(freq_port)
    
train_df[['embarked', 'survived']].groupby(['embarked'], as_index=False).mean().sort_values(by='survived', ascending=False)


# ### Converting categorical feature to numeric

# In[86]:


for dataset in train_df:
    dataset['embarked'] = dataset['embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# In[87]:


train_df['fare'].fillna(train_df['fare'].dropna().median(), inplace=True)
train_df.head()


# In[88]:


train_df['FareBand'] = pd.qcut(train_df['fare'], 4)
train_df[['FareBand', 'survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[89]:


for dataset in train_df:
    dataset.loc[ dataset['fare'] <= 7.91, 'fare'] = 0
    dataset.loc[(dataset['fare'] > 7.91) & (dataset['fare'] <= 14.454), 'fare'] = 1
    dataset.loc[(dataset['fare'] > 14.454) & (dataset['fare'] <= 31), 'fare']   = 2
    dataset.loc[ dataset['fare'] > 31, 'fare'] = 3
    dataset['fare'] = dataset['fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)

    
train_df.head(10)


# In[90]:


train_df.head(10)


# In[91]:


copy_df=train_df.copy()



# ### Encoding into numeric feature

# In[92]:


from sklearn.preprocessing import OneHotEncoder


# In[94]:


train_Embarked = copy_df["embarked"].values.reshape(-1,1)



# In[95]:


onehot_encoder = OneHotEncoder(sparse=False)
train_OneHotEncoded = onehot_encoder.fit_transform(train_Embarked)



# In[96]:


copy_df["EmbarkedS"] = train_OneHotEncoded[:,0]
copy_df["EmbarkedC"] = train_OneHotEncoded[:,1]
copy_df["EmbarkedQ"] = train_OneHotEncoded[:,2]



# In[97]:


copy_df.head()


# In[98]:



# In[99]:


train_df.head()


# In[100]:





# ### Creating and Training a model

# In[101]:


X_trainTest = copy_df.drop(copy_df.columns[[0,5]],axis=1)
Y_trainTest = copy_df["survived"]
X_trainTest.head()


# In[102]:




# In[103]:


logReg = LogisticRegression()
logReg.fit(X_trainTest,Y_trainTest)
acc = logReg.score(X_trainTest,Y_trainTest)
acc


# In[105]:


X_train = train_df.drop("survived", axis=1)
Y_train = train_df["survived"]
X_test  = train_df.drop("passengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
X_train.head()


# In[106]:


X_test.head()


# In[110]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[116]:


logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[118]:


svcTest = SVC()
svcTest.fit(X_trainTest, Y_trainTest)
acc_svcTest = round(svcTest.score(X_trainTest, Y_trainTest)*100,2)
acc_svcTest


# In[119]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[120]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[121]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[122]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[123]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[124]:


random_forestTest = RandomForestClassifier(n_estimators=100)
random_forestTest.fit(X_trainTest, Y_trainTest)
acc_random_forestTest = round(random_forestTest.score(X_trainTest, Y_trainTest) * 100, 2)
acc_random_forestTest


# In[125]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[135]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 
              'Stochastic Gradient Decent',  
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian,
              acc_sgd,  acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




