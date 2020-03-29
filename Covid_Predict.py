#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import pickle


# In[2]:


df = pd.read_csv('covid19.csv')


# In[8]:


y = df['Outcome']
x = df[['P_age','P_gender','Fever','Running_Nose','Travel_History','Coughing','Difficulty breathing']]


# In[39]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.85, random_state=42)


# In[40]:


tr = LogisticRegression().fit(X_train,y_train)


# In[41]:


tr.score(X_test,y_test)


# In[42]:


nav = MultinomialNB().fit(X_train,y_train)


# In[47]:


nav.score(X_test,y_test)


# In[66]:


nav.predict_proba([[30,1,1,1,1,1,1]])


# In[67]:





# In[69]:


pickle.dump(nav, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

