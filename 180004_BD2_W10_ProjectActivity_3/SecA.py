#!/usr/bin/env python
# coding: utf-8

#  Project 3 Predicting the Survival of Titanic Passengers
# 
#  Section: A (Exploratory Data Analysis)

# In[84]:


# Importing all necessary libraries
#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[85]:


# Reading and exploring the dataset
titanic_df=pd.read_csv(r"Titanic.csv")



# In[86]:


titanic_df.head


# In[87]:


titanic_df.tail


# In[88]:


titanic_df.sample(5)


# In[89]:


titanic_df.describe


# In[90]:


titanic_df.info()


# In[91]:


print("The missing values details of features:")
total = titanic_df.isnull().sum().sort_values(ascending=False)
percent_1 = titanic_df.isnull().sum()/titanic_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)


# In[92]:


titanic_df.columns.values


# In[93]:


print("Numerical Features: Age (Continuous), Fare (Continuous), SibSp (Discrete), Parch (Discrete)")
print("Categorical Features: Survived, Sex, Embarked, Pclass")
print("Alphanumeric Features: Ticket, Cabin")


# ### Question 1.) Find out the overall chance of survival for a Titanic passenger.

# In[94]:


print("Total number of passengers survived are",titanic_df['survived'].value_counts()[1])
print("Percentage passengers survived are",titanic_df['survived'].value_counts(normalize=True)[1]*100)


# ### Question 2.) Find out the chance of survival for a Titanic passenger based on their sex and plot it.

# In[95]:


sns.barplot(x="sex", y="survived", data=titanic_df)
print("Percentage of females who survived is", titanic_df["survived"][titanic_df["sex"] == 'female'].value_counts(normalize = True)[1]*100)
print("Percentage of males who survived is", titanic_df["survived"][titanic_df["sex"] == 'male'].value_counts(normalize = True)[1]*100)


# ### Question 3.) Find out the chance of survival for a Titanic passenger by traveling class wise and plot it.

# In[96]:


sns.barplot(x="pclass", y="survived", data=titanic_df)
print("Percentage of Pclass 1 who survived is", titanic_df["survived"][titanic_df["pclass"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass 2 who survived is", titanic_df["survived"][titanic_df["pclass"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of Pclass 3 who survived:", titanic_df["survived"][titanic_df["pclass"] == 3].value_counts(normalize = True)[1]*100)


# ### Question 4.) Find out the average age for a Titanic passenger who survived by passenger class and sex. 

# In[97]:


fig = plt.figure(figsize=(12,5))
fig.add_subplot(121)
plt.title('TRAIN - Age/Sex per Passenger Class')
sns.barplot(data=titanic_df, x='pclass',y='age',hue='sex')


# In[98]:


meanAgeTrnMale = round(titanic_df[(titanic_df['sex'] == "male")]['age'].groupby(titanic_df['pclass']).mean(),2)
meanAgeTrnFeMale = round(titanic_df[(titanic_df['sex'] == "female")]['age'].groupby(titanic_df['pclass']).mean(),2)


print('\n\t\tMEAN AGE PER SEX PER PCLASS')
print(pd.concat([meanAgeTrnMale, meanAgeTrnFeMale], axis = 1,keys= ['Male','Female']))


# ### Question 5.) Find out the chance of survival for a Titanic passenger based on number of siblings the passenger had on the ship and plot it.

# In[99]:


sns.barplot(x="sibsp", y="survived", data=titanic_df)
print("Percentage of SibSp 0 who survived is", titanic_df["survived"][titanic_df["sibsp"] == 0].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp 1 who survived is", titanic_df["survived"][titanic_df["sibsp"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of SibSp 2 who survived is", titanic_df["survived"][titanic_df["sibsp"] == 2].value_counts(normalize = True)[1]*100)


# ### Question 6.) Find out the chance of survival for a Titanic passenger based on number of parents/children the passenger had on the ship and plot it.

# In[100]:


sns.barplot(x="parch", y="survived", data=titanic_df)
plt.show()
print("Percentage of parch 0 who survived is", titanic_df["survived"][titanic_df["parch"] == 0].value_counts(normalize = True)[1]*100)
print("Percentage of parch 1 who survived is", titanic_df["survived"][titanic_df["parch"] == 1].value_counts(normalize = True)[1]*100)
print("Percentage of parch 2 who survived is", titanic_df["survived"][titanic_df["parch"] == 2].value_counts(normalize = True)[1]*100)
print("Percentage of parch 3 who survived is", titanic_df["survived"][titanic_df["parch"] == 3].value_counts(normalize = True)[1]*100)


# ### Question 7.) Plot out the variation of survival and death amongst passengers of different age.

# In[101]:


titanic_df["age"] = titanic_df["age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
titanic_df['agegroup'] = pd.cut(titanic_df['age'], bins, labels = labels)
sns.barplot(x="agegroup", y="survived", data=titanic_df)
plt.show()


# In[102]:


g = sns.FacetGrid(titanic_df, col='survived')
g.map(plt.hist, 'age', bins=20)


# ### Question 8.) Plot out the variation of survival and death with age amongst passengers of different passenger classes.

# In[103]:


grid = sns.FacetGrid(titanic_df, col='survived', row='pclass', size=3, aspect=2)
grid.map(plt.hist, 'age', alpha=.5, bins=20)
grid.add_legend();


# ### Question 9.) Find out the survival probability for a Titanic passenger based on title from the name of passenger.

# In[104]:


#create a combined group of both datasets
combine = [titanic_df, test_df]

#extract a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(titanic_df['Title'],titanic_df['sex'])


# In[105]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

titanic_df[['Title', 'survived']].groupby(['Title'], as_index=False).mean()


# ### Question 10.) What conclusions are you derived from the analysis?

# In[106]:


print("Following are the conclusions which I have made: ")
print("Females have a much higher chance of survival than males.")
print("People with higher socioeconomic class had a higher rate of survival")
print("People with no siblings or spouses were less to likely to survive than those with one or two")
print("People traveling alone are less likely to survive than those with 1-3 parents or children.")
print("Babies are more likely to survive than any other age group.")
print("People with a recorded Cabin number are, in fact, more likely to survive.")

