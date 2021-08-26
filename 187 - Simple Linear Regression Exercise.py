#!/usr/bin/env python
# coding: utf-8

# # Simple linear regression - exercise

# You are given a real estate dataset. 
# 
# Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.
# 
# The data is located in the file: 'real_estate_price_size.csv'. 
# 
# You are expected to create a simple linear regression (similar to the one in the lecture), using the new data.
# 
# In this exercise, the dependent variable is 'price', while the independent variables is 'size'.
# 
# Good luck!

# ## Import the relevant libraries

# In[3]:


import numpy as np #multidimensional arrays
import pandas as pd #format data into columns and rows
import matplotlib.pyplot as plt #2d visualization
import statsmodels.api as sm #summaries
import seaborn #nice graphs
seaborn.set()


# ## Load the data

# In[4]:


data = pd.read_csv('real_estate_price_size.csv')


# In[22]:


data


# In[20]:


data.head()


# In[21]:


data.describe()


# ## Create the regression

# ### Declare the dependent and the independent variables

# In[8]:


y = data['price']
x1 = data['size']


# ### Explore the data

# In[9]:


plt.scatter(x1,y)
plt.xlabel('Size',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.show()


# ### Regression itself

# In[10]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# ### Plot the regression line on the initial scatter

# In[12]:


plt.scatter(x1,y)
yhat = 223.1787*x1 + 101900
fig = plt.plot(x1, yhat, lw=4, c='red', label='regression line')
plt.xlabel('Size', fontsize=20)
plt.ylabel('Price', fontsize=20)
plt.show()

