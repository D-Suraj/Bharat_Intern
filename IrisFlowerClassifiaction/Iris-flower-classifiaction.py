#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv('iris.csv')


# In[3]:


X = df.drop('species', axis=1)
y = df['species']


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)


# In[6]:


y_pred = classifier.predict(X_test)


# In[7]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[ ]:
