
# coding: utf-8

# In[1]:


import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd


# In[2]:


data = pd.read_csv("Manual-Data/Training.csv")


# In[3]:


data.head()


# In[4]:


data.columns


# In[6]:


len(data.columns)




len(data['prognosis'].unique())


df = pd.DataFrame(data)


df.head()



cols = df.columns



cols = cols[:-1]


cols



len(cols)


x = df[cols]
y = df['prognosis']
# print x[:5]
# print y[:5]




with open('Manual-Data/Training.csv') as f:
    reader = csv.reader(f)
    i = next(reader)
    rest = [row for row in reader]
column_headings = i



for ix in i:
    ix = ix.replace('_', ' ')
    # print ix



#import seaborn as sns
#get_ipython().magic(u'matplotlib inline')



x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)



mnb = MultinomialNB()
mnb = mnb.fit(x_train, y_train)




mnb.score(x_test, y_test)




print("cross result========")
scores = cross_val_score(mnb, x_test, y_test, cv=3)
#print (scores)
#print (scores.mean())




test_data = pd.read_csv("Manual-Data/Testing.csv")




test_data.head()


testx = test_data[cols]
testy = test_data['prognosis']


# In[25]:


mnb.score(testx, testy)


# In[26]:


# In[27]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)


# In[28]:


#print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt = dt.fit(x_train, y_train)
#print ("Acurracy: ", clf_dt.score(x_test,y_test))


# In[29]:


#print ("cross result========")
scores = cross_val_score(dt, x_test, y_test, cv=3)
#print (scores)
#print (scores.mean())


# In[30]:


#print ("Acurracy on the actual test data: ", clf_dt.score(testx,testy))


# In[31]:


#get_ipython().magic(u'matplotlib inline')

importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
#print("Feature ranking:")


# In[32]:


features = cols


# In[33]:


for f in range(5):
    print("%d. feature %d - %s (%f)" %
          (f + 1, indices[f], features[indices[f]], importances[indices[f]]))


# In[34]:


feature_dict = {}
for i, f in enumerate(features):
    feature_dict[f] = i


# In[35]:


feature_dict['hip_joint_pain']


# In[36]:


sample_x = [i/79 if i == 79 else i*0 for i in range(len(features))]


# In[37]:


len(sample_x)


# In[38]:


sample_x = np.array(sample_x).reshape(1, len(sample_x))


# In[39]:


# print dt.predict(sample_x)


# In[40]:

decision_tree_pkl_filename = 'decision_tree_classifier.pickle'
# Open the file to save as pkl file
decision_tree_model_pkl = open(decision_tree_pkl_filename, 'wb')
pickle.dump(dt, decision_tree_model_pkl)
# Close the pickle instances
decision_tree_model_pkl.close()
# print dt.predict_proba(sample_x)
