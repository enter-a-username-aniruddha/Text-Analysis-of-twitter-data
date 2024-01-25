#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp=spacy.load('en_core_web_sm')


# In[2]:


df=pd.read_csv('tweets.csv',encoding='ISO-8859-1')
df


# In[3]:


df.loc[0,'text']


# In[4]:


df.loc[512,'text']


# In[5]:


#pre process tweets
def preprocess(text):
    #remove unicode characters
    text=re.sub(r'<U\+[A-Z0-9]+>|<ed>','',text)
    #remove newline and rawstring characters
    text=re.sub(r'\n|\r','',text)
    
    return text


# In[6]:


df['text']=df['text'].apply(preprocess)


# In[7]:


df.head()


# In[8]:


# to find mentions and their counts
def mentions(text):
    mentions=re.findall('@\w+',text)
    
    return len(mentions)


# In[9]:


df['mentions_count']=df['text'].apply(mentions)


# In[10]:


df.head()


# In[11]:


df['mentions_count'].describe()


# In[12]:


# hashtags and its count
def hashtags(text):
    hashtags=re.findall('#\w+',text)
    
    return len(hashtags)


# In[13]:


df['hashtags_count']=df['text'].apply(hashtags)
df.head()


# In[14]:


df['hashtags_count'].describe()


# In[15]:


# title of a person with count
def title(text):
    count=re.findall('Mr\.|Mrs.\|Dr.\|Miss\s*',text)
    
    return len(count)


# In[16]:


df['title']=df['text'].apply(title)


# In[17]:


df.head()


# In[18]:


# list comprehension to count number of words
df['word_count']=[len(i.split()) for i in df['text']]


# In[19]:


df.head()


# In[20]:


df['word_count'].describe()


# In[21]:


#list comprehension for character count
df['character_count']=[len(i) for i in df['text']]
df.head()


# In[22]:


df['character_count'].describe()


# In[23]:


#Average word length
def avg_word_len(text):
    #variable to store word length
    word_lens=0
    
    #iterating 
    for token in text.split():
        word_lens+=len(token)
    #number of words in tweet
    word_count=text.split()
    #return avg length
    return word_lens/len(word_count)


# In[24]:


df['avg_word_len']=df['text'].apply(avg_word_len)
df.head()


# In[25]:


df['avg_word_len'].describe()


# In[26]:


#function to count number of stop words
def stopwords(text):
    
    #create a spacy object
    doc=nlp(text)
    #cariable to store
    count=0
    for token in doc:
        if token.is_stop==True:
            count+=1
    return count


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




