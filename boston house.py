#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Data description

#The Boston data frame has 506 rows and 14 columns.

#This data frame contains the following columns:

#crim
#per capita crime rate by town.

#zn
#proportion of residential land zoned for lots over 25,000 sq.ft.

#indus
#proportion of non-retail business acres per town.

#chas
#Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

#nox
#nitrogen oxides concentration (parts per 10 million).

#rm
#average number of rooms per dwelling.

#age
#proportion of owner-occupied units built prior to 1940.

#dis
#weighted mean of distances to five Boston employment centres.

#rad
#index of accessibility to radial highways.

#tax
#full-value property-tax rate per \$10,000.

#ptratio
#pupil-teacher ratio by town.

#black
#1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

#lstat
#lower status of the population (percent).

#medv
#median value of owner-occupied homes in \$1000s.


# In[2]:


df = pd.read_csv('boston.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[18]:


df.drop(['Unnamed: 0'], axis=1)


# In[19]:


df.drop(['Unnamed: 0'], axis=1)


# In[21]:


df.head()


# In[24]:


df.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[25]:


df.head()


# In[26]:


df.describe()


# In[28]:


#exploring the relationship between other variables and the median value of the house
sns.lmplot('age', 'medv', df)
plt.xlabel('age of the house')
plt.ylabel('median value of owner-occupied homes in $1000s')
plt.show()


# ##### The plot above shows a negative relationship between the price of the house and its age. older houses tend to be of lower value 

# In[29]:


sns.lmplot('rm', 'medv', df)
plt.xlabel('average number of rooms per dwelling')
plt.ylabel('median value of owner-occupied homes in $1000s')
plt.show()


# ##### Contrary to the age-medv realtionship, the age-average number of room per dwelling showed a positive relationship. An increase in the average number of rooms per dwelling shows a corresponding increase in median value of owner-occupied nomes in $1000s

# In[30]:


sns.lmplot('ptratio', 'medv', df)
plt.xlabel('pupil-teacher ratio')
plt.ylabel('median value of owner-occupied homes in $1000s')
plt.show()


# ##### similar to the age-medv relationship, ptratio-medv also shows a negative correlation. As the ptratio increases, medv decreases. That is, if there are fewer teachers, the medv increases and vice vera.

# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[37]:


X = df.drop(['medv'], axis = 1)
Y = df['medv']
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2,random_state = 0)


# In[34]:


x_train.shape, y_train.shape


# In[38]:


x_test.shape, y_train.shape


# In[40]:


linear_regression = LinearRegression()
linear_regression.fit(x_train,y_train)


# In[41]:


y_pred = linear_regression.predict(x_test)
y_pred


# In[42]:


#Comparing the actual value of y (y_test) to the predicted value of y (y_pred)
df_y = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
df_y.sample(10)


# #### we can tell that there are slight variations between the actual value of y and predicted value

# In[43]:


#measuring the extent of correctness of the machine learning model
print('Training score ; ', linear_regression.score(x_train, y_train))


# #### training score (R Square) measures the goodness of fit of the linear regression. The training score value showed that the line of fit captures sufficient amount of variance points in the data. That is, very few outliers.

# In[46]:


#now that we know the Rsquare for the train data, let find the Rsquare for the test data.
from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print('Testing score: ', score)


# #### the r2 score for the test data differ r2 score of the training data but not too widely. However, another portion of the data could be trained and test using the k-fold for better fitting

# In[58]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
def linear_model(x_train, y_train):
    print('Linear Regression')
    linear_regression = LinearRegression()
    linear_regression.fit(x_train,y_train)
    
    return linear_regression
        


# In[60]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
def lasso(x_train, y_train):
    print('Lasso Regression')
    lasso_regression = LassoRegression(alpha=0.8, max_iter=10000)
    lasso_regression.fit(x_train,y_train)
    
    return lasso_regression


# In[61]:



def build_and_train_model(df, target_name, reg_fn):
    X = df.drop(['target_name'], axis = 1)
    Y = df['target_name']
    x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3,random_state = 0)
    model = reg_fn(x_train, y_train)
    score = score.model(x_train, y_train)
    print('Training Score: ', score)
    
    y_pred = model.predict(x_test)
    r_score = r2_score(y_test, y_pred)
    print('Testing score: ', r_score)
    
    return {'model': model,
            'x_train': x_train, 'x_test': x_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred': y_pred
           }


# In[ ]:


linear_regression = build_and_train_model(df,'price', lasso_regression)
lasso_regression = build_and_train_model(df,'price', lasso_regression)


# #### Lasso and SGD would be appropriate for data with over 100k entries.

# In[ ]:




