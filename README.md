# Iris-Flower-classification-
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:


df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\IRIS.csv')
df.head()


# In[3]:


df.describe()


# In[4]:


import seaborn as sns
sns.pairplot(df , hue="species")



# In[5]:


x=pd.DataFrame(df,columns=["sepal_length","sepal_width","petal_length","petal_width"]).values
y=df.species.values.reshape(-1,1)


# 

# In[6]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
model_LR=LogisticRegression()
model_LR.fit(X_train,y_train)


# In[8]:


prediction= model_LR.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction)*100)


# In[9]:


from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))


# In[12]:


df.columns


# In[11]:


model_LR.predict([[5.7,2.8,4.1,1.3]])


# In[ ]:




