import json
from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)


model = load_model(r"\Tensorflow Models\irrigation.h5")

@app.route("/")
def index():
    return "Atleat server is working !!! Hello Lets ML!"


# define a predict function as an endpoint
@app.route("/calc", methods=["GET","POST"])
def predict():

    data = []

    params = request.json

    if (params == None):
        #params = [ 0.08611226061567466, 0.08156113102349655, 0.08850692662904062, 0.22089688050165154,  0.2149282761974228]
        answer = "No Data Recieved"

    
    if (params != None):

        for i in params:
            data.append(i)

        df = pd.DataFrame([data], columns=["Light", "Moisture 1", "Moisture 2", "Temp", "Humidity"])
        properties = list(df.columns.values)
        x = df[properties]
        #print(x)         # dataframe is correctly formed
        
        prediction = (model.predict(x))
        answer = round(prediction[0][0], 0)


    # return a response in json format
    return str(answer),201        # tried returning x , confirmed x is a df

# start the flask app, allow remote connections
app.run(debug=True, host='127.0.1.2', port= 4444)
