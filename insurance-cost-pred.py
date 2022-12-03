#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pycaret
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl


# In[8]:


model=pkl.load(open('final_model.pkl','rb'))


# In[9]:


st.title('---------- Medical Insurance Cost Prediction ---------')


# In[11]:


age = st.number_input('Age', min_value=1, max_value=100, value=25)
sex = st.selectbox('Sex', ['male', 'female'])
bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
if st.checkbox('Smoker'):
    smoker = 'yes'
else:
    smoker = 'no'
region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])
output=""
if st.button("Predict"):
    result = model.predict([[age,sex,bmi,children,smoker,region]])
    st.success ('The output of the above is {}'.format(result))  

