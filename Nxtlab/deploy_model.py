#!/usr/bin/env python
# coding: utf-8

# In[78]:



import warnings
warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np

import string
#import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


import re


# from gensim.models import Word2Vec
# from gensim.models import KeyedVectors
#import pickle

# from tqdm import tqdm
# import os

from collections import Counter

from sklearn.preprocessing import OneHotEncoder



#import matplotlib.pyplot as plt

from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
import streamlit as st


# In[80]:


#get_ipython().system('pip install protobuf==3.20.*')


# In[79]:


st.markdown("<h1 style='text-align: center; color: red;'>NXTLABS</h1>", unsafe_allow_html=True)
uploaded_data = st.file_uploader('test_data')


# In[66]:


X_train = pd.read_csv('X_train.csv')


# In[12]:

if uploaded_data is not None:
    st.subheader('Input:\n')
    data = pd.read_csv(uploaded_data)
    data = data.drop(data.columns[:2],axis = 1)
    data['Nationality'] = data['Nationality'].str.lower() 
    data['DistributionChannel'] = data['DistributionChannel'].str.replace(' ','_')
    data['DistributionChannel'] = data['DistributionChannel'].str.replace('/','_')
    data['DistributionChannel'] = data['DistributionChannel'].str.lower()
    data['MarketSegment'] = data['MarketSegment'].str.replace(' ','_')
    data['MarketSegment'] = data['MarketSegment'].str.replace('/','_')
    data['MarketSegment'] = data['MarketSegment'].str.lower()
    data['Age'].fillna(46.0,inplace = True)
    st.write(data)

    # In[67]:


    lis = ['Nationality','DistributionChannel','MarketSegment']
    dic = dict()
    for i in lis:
        vectorizer = CountVectorizer()
        vectorizer.fit(X_train[i].values)
        dic[i] = vectorizer.transform(data[i].values)
        #print(dic[i].shape)


    # In[58]:


    #dic


    # In[68]:


    from sklearn.preprocessing import Normalizer
    lisn = ['Age', 'DaysSinceCreation', 'AverageLeadTime',
        'LodgingRevenue', 'OtherRevenue', 'BookingsCanceled',
        'BookingsNoShowed', 'BookingsCheckedIn', 'PersonsNights',
        'RoomNights', 'DaysSinceLastStay', 'DaysSinceFirstStay', 'SRHighFloor',
        'SRLowFloor', 'SRAccessibleRoom', 'SRMediumFloor', 'SRBathtub',
        'SRShower', 'SRCrib', 'SRKingSizeBed', 'SRTwinBed',
        'SRNearElevator', 'SRAwayFromElevator', 'SRNoAlcoholInMiniBar',
        'SRQuietRoom']
    for i in lisn:
        normalizer = Normalizer()
        normalizer.fit(X_train[i].values.reshape(-1,1))
        dic[i] = normalizer.transform(data[i].values.reshape(-1,1))
        #print(dic[i].shape)


    # In[69]:


    #(dic['Nationality'],dic['DistributionChannel'],dic['MarketSegment'],dic['Age'],dic['DaysSinceCreation'],dic['AverageLeadTime'],dic['LodgingRevenue'],dic['OtherRevenue'],dic['BookingsCanceled'],dic['BookingsNoShowed'],dic['BookingsCheckedIn'],dic['PersonsNights'],dic['RoomNights'],dic['DaysSinceLastStay'],dic['DaysSinceFirstStay'],dic['SRHighFloor'],dic['SRLowFloor'],dic['SRAccessibleRoom'],dic['SRMediumFloor'],dic['SRBathtub'],dic['SRShower'],dic['SRCrib'],dic['SRKingSizeBed'],dic['SRTwinBed'],dic['SRNearElevator'],dic['SRAwayFromElevator'],dic['SRNoAlcoholInMiniBar'],dic['SRQuietRoom'])
            


    # In[70]:


    from scipy.sparse import hstack
    query_data = hstack((dic['Nationality'],dic['DistributionChannel'],dic['MarketSegment'],dic['Age'],dic['DaysSinceCreation'],dic['AverageLeadTime'],dic['LodgingRevenue'],dic['OtherRevenue'],dic['BookingsCanceled'],dic['BookingsNoShowed'],dic['BookingsCheckedIn'],dic['PersonsNights'],dic['RoomNights'],dic['DaysSinceLastStay'],dic['DaysSinceFirstStay'],dic['SRHighFloor'],dic['SRLowFloor'],dic['SRAccessibleRoom'],dic['SRMediumFloor'],dic['SRBathtub'],dic['SRShower'],dic['SRCrib'],dic['SRKingSizeBed'],dic['SRTwinBed'],dic['SRNearElevator'],dic['SRAwayFromElevator'],dic['SRNoAlcoholInMiniBar'],dic['SRQuietRoom'])).tocsr()


    # In[71]:


    #query_data.shape


    # In[72]:


    
    import keras

    model = keras.models.load_model(r'C:\Users\goldr\Desktop\Nxtlab')


    # In[76]:

    st.subheader('Output:\n')
    st.write('Raw output\n')
    st.write(model.predict(query_data))
    st.write('\nWith threshold = 0.5')
    st.write([1 if x>0.5 else 0 for x in model.predict(query_data)])


    # In[ ]:




