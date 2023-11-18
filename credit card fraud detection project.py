#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


df=pd.read_csv('creditcard.csv')
df


# In[3]:


df.isnull().sum()


# In[6]:


df.info()


# In[7]:


non_fraud=len(df[df.Class==0])


# In[8]:


fraud=len(df[df.Class==1])


# In[9]:


df['Class'].value_counts()


# In[19]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[20]:


df['Normalized_amount']=scaler.fit_transform(df['Amount'].values.reshape(-1,1))


# In[25]:


df.drop(['Amount'],inplace=True ,axis=1)


# In[26]:


df.describe()


# In[27]:


x=df.drop(['Class'],axis=1)
y=df.Class


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


# In[29]:


len(x_train)


# In[30]:


len(y_test)


# In[31]:


from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()


# In[32]:


reg.fit(x_train,y_train)


# In[33]:


reg.predict(x_test)


# In[34]:


reg.score(x_train,y_train)


# In[35]:


reg.score(x_test,y_test)


# In[ ]:




