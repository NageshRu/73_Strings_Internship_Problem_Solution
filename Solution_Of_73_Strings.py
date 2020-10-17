#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd              #It is used for getting a excel file from computer/users file
import matplotlib.pyplot as plt  #It is used for visualization.
import nltk                      #VADER model is available in this package so that's why imported.
from nltk.sentiment.vader import SentimentIntensityAnalyzer  
import numpy as np


# In[59]:


#load the data
train_data=pd.read_excel(r'N:\Machine Learning\Sentiment Analysis\Train_dataset_of_Customer_Mobile_Review.xlsx') #get a users training dataset


# In[60]:


train_data #display the testing dataset


# In[61]:


nltk.download('vader_lexicon')   #download the VADER model


# In[62]:


sid=SentimentIntensityAnalyzer()  #It takes in string and returns a dictionary of scores of positive,negative,neutral and compound.


# In[63]:


train_data['Customer_Review'].value_counts()


# In[86]:


# REMOVE NaN VALUES AND EMPTY STRINGS:
train_data.dropna(inplace=True)

blanks=[] #start with an empty list

for i in train_data.itertuples():
    if type(i)==str:            
        if i.isspace():        
            blanks.append(i) 
    
train_data.drop(blanks,inplace=True)

#Storing the data in df variable with column header 
df=pd.DataFrame(train_data,columns=['Customer_Review'])


#import data from removed extra string and spaces from sentences
df.to_excel(r'N:\Machine Learning\Sentiment Analysis\Data_Clean_File.xlsx',index=False,header=True)


# In[65]:


train_data['Customer_Review'] #Display test_data


# In[66]:


#Adding score and labels in the test_data
train_data['Scores']=train_data['Customer_Review'].apply(lambda Customer_Review:sid.polarity_scores(Customer_Review))

#Seperate the compound value of the sentence
train_data['Compound']=train_data['Scores'].apply(lambda score_dict:score_dict['compound'])

#Calculating which sentences is positive,neutral and negative sentences
train_data['Compound_Score']=train_data['Compound'].apply(lambda c: 'Neutral' if c==0 else( 'Positive' if c>0 else 'Negative'))

#Display the data
train_data


# In[67]:


#Storing the data in df variable with column header 
df=pd.DataFrame(train_data,columns=['Mobile_Name','Rating','Customer_Review','Scores','Compound','Compound_Score'])


# In[68]:


#import data in Solution.xlsx file
df.to_excel(r'N:\Machine Learning\Sentiment Analysis\Solution.xlsx',index=False,header=True)


# In[83]:


#Visualization of the data(Positive,Negative,Neutral)

train_data.Compound_Score.value_counts().plot(kind='pie',autopct='%1.0f%%', colors=["yellow", "red", "green"])
plt.savefig('N:\Machine Learning\Sentiment Analysis\PieChart_Of_Result.png',dpi=300,bbox_inches='tight')


# In[84]:


#Display the Rating data into bar chart 
train_data.Rating.value_counts().plot(kind='bar')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.savefig('N:\Machine Learning\Sentiment Analysis\BarChart_Of_Result.png',dpi=300,bbox_inches='tight')


# In[85]:


#Display the data into scatter plot
plt.scatter(train_data['Rating'],train_data['Compound'],marker='^')
plt.xlabel('Rating')
plt.ylabel('Compound')
plt.savefig('N:\Machine Learning\Sentiment Analysis\ScatterPlot_Of_Result.png',dpi=300,bbox_inches='tight')


# In[ ]:




