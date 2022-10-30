#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle


# In[2]:


app=Flask(__name__)


# In[3]:


model=pickle.load(open('model/model.pkl','rb'))


# In[4]:

@app.route('/')
def home():
    return render_template('index.html')


# In[5]:

@app.route('/predict',methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    features= [np.array(int_features)]
    prediction=model.predict(features)
    output=prediction[0]
    return render_template('index.html',prediction_text='Pass or fail  {}'.format(output))


# In[6]:


if __name__ == "__main__":
    app.run()


# In[ ]:




