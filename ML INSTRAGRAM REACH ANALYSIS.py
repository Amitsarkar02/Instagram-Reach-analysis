#!/usr/bin/env python
# coding: utf-8

# In[91]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR


# In[4]:


df = pd.read_csv("Instagram data.csv",encoding = 'latin1')


# In[5]:


df.head()


# In[7]:


df.isnull().sum()


# In[9]:


df.info()


# # EDA

# In[15]:


plt.figure(figsize = (10,8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.distplot(df['From Home'])
plt.show()


# In[18]:


plt.figure(figsize = (10,8))
plt.title("Distribution of Impressions From Hashtags")
sns.distplot(df['From Hashtags'])
plt.show()


# **So from this above two distplot we can infer that impressions from hashtag is create more impressions from home**

# In[20]:


plt.figure(figsize = (10,8))
plt.title("Distribution of Impressions From Explore")
sns.distplot(df['From Explore'])
plt.show()


# In[41]:


Home = df['From Home'].sum()
Hashtag = df['From Hashtags'].sum()
Explore = df['From Explore'].sum()
Other = df['From Other'].sum()


labels = ['From Home','From Hashtags','From Explore','From Other']
values = [Home,Hashtag,Explore,Other]

fig = px.pie(df,values = values, names=labels,title = 'percentage of impressions',hole=0.5)
fig.show()


# In[56]:



figure = px.scatter(data_frame = df, x="Impressions",

                    y="Likes", size="Likes", trendline="ols", 

                    title = "Relationship Between Likes and Impressions")

figure.show()


# **Now you can see a linear Relationship Between likes and impressions**

# In[57]:



figure = px.scatter(data_frame = df, x="Impressions",

                    y="Comments", size="Comments", trendline="ols", 

                    title = "Relationship Between Comments and Impressions")

figure.show()


# **It looks like the number of comments we get on a post doesn’t affect its reach.**

# In[59]:



figure = px.scatter(data_frame = df, x="Impressions",

                    y="Shares", size="Shares", trendline="ols", 

                    title = "Relationship Between Shares and Impressions")

figure.show()


# **A more number of shares will result in a higher reach, but shares don’t affect the reach of a post as much as likes do**

# In[61]:


figure = px.scatter(data_frame = df, x="Impressions",

                    y="Saves", size="Saves", trendline="ols", 

                    title = "Relationship Between Saves and Impressions")

figure.show()


# **So there a liner Regression between no of times you save post and reach on instagram**

# In[62]:


correlation = df.corr()
print(correlation["Impressions"].sort_values(ascending=False))


# **So we can say that the more you explore more you got impressions**

# # Analysize conversion Rate

# In[63]:


conversions_Rate = (df["Follows"].sum() / df['Profile Visits'].sum()) * 100


# In[64]:


print(conversions_Rate)


# In[65]:


figure = px.scatter(data_frame = df, x="Profile Visits",

                    y="Follows",  trendline="ols", 

                    title = "Relationship Between Profile Visits and Follows")

figure.show()


# **There is linear Relationship between Profile Visits and Follows**

# # Instagram Reach Prediction Model

# In[80]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # **linear Regression**

# In[76]:


x = np.array(df[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(df["Impressions"])


# In[78]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)


# In[81]:


lr = LinearRegression()
lr.fit(x_train,y_train)


# In[82]:


lr.score(x_test,y_test)


# # **Decision Tree**

# In[84]:


DTRmodel = DecisionTreeRegressor(max_depth=3,random_state=0)
DTRmodel.fit(x_train,y_train)
y_pred = DTRmodel.predict(x_test)


# In[85]:


DTRmodel.score(x_test,y_test)


# # **Random Forest**

# In[89]:


rf = RandomForestRegressor(max_depth=2, random_state=0)
rf.fit(x_train,y_train)
y_pred1 = rf.predict(x_test)


# In[90]:


rf.score(x_test,y_test)


# # **Support vector Regression**

# In[92]:


svr=SVR(C=50000.0, max_iter=500)

svr.fit(x_train, y_train)


# In[93]:


svr.score(x_test,y_test)


# In[95]:


# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
lr.predict(features)


# In[97]:


features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
svr.predict(features)


# In[98]:


features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
rf.predict(features)


# # CONCLUSIONS

# 1. So we can see that Linear Regressions is best fit for the mode with a acccuracy of 87 %
# 2. similarily RandomForest also best fit for the model with accuracy of 81 % 

# In[ ]:




