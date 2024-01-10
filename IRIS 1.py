#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


df = pd.read_csv('C:/Users/saroh/OneDrive/Desktop/iriss/Iris (1).csv')
df


# In[5]:


dimension= df.shape
print("Dimension of the dataset is:",dimension)
df.describe()


# In[6]:


df.info()


# In[7]:


Types_of_species=df.value_counts("Species")
print("Number of species and number of flowers in each species")
Types_of_species


# In[15]:


plt.hist(df['SepalLengthCm'],color='grey')
plt.title("Histogram of Sepal length vs Frequency")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")


# In[17]:


plt.hist(df['SepalWidthCm'],color = 'silver')
plt.title("barplot")
plt.xlabel("Sepal Width")
plt.ylabel("frequency")


# In[20]:


plt.hist(df['PetalLengthCm'],color='grey')
plt.title("histogram")
plt.ylabel("Petal Length")
plt.xlabel("frequency")


# In[21]:


plt.hist(df['PetalWidthCm'],color='silver')
plt.title("histogram")
plt.xlabel("Petal width")
plt.ylabel("frequency")


# In[28]:


#we can define the species with help of their SEPAL LENGTH
sb.swarmplot(x='Species',y='SepalLengthCm',data=df)


# In[ ]:


# Now we plot the point of SPECIES with respect to their respective Measurments


# In[27]:


sb.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df, )
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()


# In[28]:


sb.scatterplot(x='PetalLengthCm', y='PetalWidthCm', hue='Species', data=df, )
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()


# In[29]:


sb.scatterplot(x='SepalLengthCm', y='PetalWidthCm', hue='Species', data=df, )
plt.legend(bbox_to_anchor=(1, 1), loc=3) 
plt.show()


# In[10]:


sb.pairplot(df,hue='Species', height=6)


# In[31]:


df


# In[32]:


encoded =pd.get_dummies(df)
encoded


# In[34]:


sb.heatmap(encoded.corr(), annot = True)
plt.show


# In[44]:


from sklearn.model_selection import train_test_split
x = df.drop(columns = ['Species'])
y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.70)
print("Random data of species in y_test")
print(y_test)
print("\n")
print("Random data of species in x_test")
print(x_test)


# In[48]:


from sklearn.tree import DecisionTreeClassifier
id3 = DecisionTreeClassifier(criterion = 'entropy')
#fitiing the data
k = id3.fit(x_train,y_train)
#predict the data
y_predict = id3.predict(x_test)
print(y_predict)


# In[49]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_predict,y_test)
plt.figure(figsize=(10,5))
sb.heatmap(cm,annot = True)
plt.xlabel('predict-x')
plt.ylabel('actual-y')
plt.show()


# In[50]:


print(classification_report(y_predict,y_test))
print('accuracy-score', accuracy_score(y_predict,y_test))
print('Model score',id3.score(x_test,y_test))


# In[ ]:




