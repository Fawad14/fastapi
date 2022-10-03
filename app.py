# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 02:49:58 2022

@author: fawad
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Maintenance import Maintenance
import numpy as np
import pickle
import pandas as pd
import keras 
from keras.models import load_model
# 2. Create the app object
app = FastAPI()
classifier=load_model('model.h5')

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Krish Youtube Channel': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_maintenance(data:Maintenance):
    data = data.dict()
    Temperature=data['TEMPRATURE']
    Voltage=data['VOLTAGE']
    Current=data['current']
    Power=data['POWER']
   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[Temperature,Voltage,Current,Power]])
    if(prediction[0]>0.5):
            prediction="Maintenance Required"
          
    else:
            prediction="Maintenance isn't required"
       
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload