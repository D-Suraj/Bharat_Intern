#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[9]:


data = pd.read_csv("winequality.csv")

print(data.head())


# In[10]:


data.info()


# In[11]:


data.describe().T


# In[12]:


data.isnull().sum()


# In[13]:


data.hist(bins=20, figsize=(10, 10))
plt.show()


# In[16]:


plt.bar(data['quality'], data['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[20]:


numeric_data = data.select_dtypes(include=['number'])

plt.figure(figsize=(12, 12))
sb.heatmap(numeric_data.corr() > 0.7, annot=True, cbar=False)
plt.show()


# In[21]:


X = data[['fixed acidity', 'volatile acidity', 'citric acid', 'alcohol']].values
y = data['quality'].values


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer




# In[32]:


pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('classifier', LogisticRegression(max_iter=10000))
])

pipeline.fit(X_train, y_train)


# In[38]:


# Now, you can use the model to make predictions on test data (X_test)
y_pred = pipeline.predict(X_test)


# In[39]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[40]:


print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[41]:


conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)


# In[ ]:
