import json
from flask import Flask, request, jsonify
import pandas as pd

import csv
import keras
from tensorflow.keras.models import load_model
import numpy as np
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications.imagenet_utils import (
    preprocess_input,
    decode_predictions,
)

min_val = 135
max_val = 600


def RangeScaling(val):
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    else:
        return val


# ------- LOAD irrigationModel -------


irrigationModel = load_model(
    r"/home/jetbot/Desktop/server/Python/Tensorflow irrigationModels/irrigation.h5"
)
diseaseModel = load_model(
    r"/home/jetbot/Desktop/server/Python/Tensorflow Models/DiseaseIdentification.h5"
)

# ------- LIST ASSIGNMENT -------
min_max_values = []

# ------ READING MinMaxValue Csv Created During Normalization --------
with open(r"CSVs/Target/minMaxVals.csv", "rt") as f:
    data = csv.reader(f)
    for row in data:
        min_max_values.append(row[0])
# ------ Converting to Float --------

for i in range(2, len(min_max_values)):
    min_max_values[i] = float(min_max_values[i])


# ------ FLASK APP START ------

app = Flask(__name__)


@app.route("/")
def index():
    return "Welcome TO AIFA backend!"


# define a predict function as an endpoint
@app.route("/calc", methods=["GET", "POST"])
def predict():
    global irrigationModel
    paramIndex = [6, 4, 5, 2, 3]

    data = []
    parList = []

    params = request.json
    parDict = params[0]
    parListValues = list(parDict.values())

    for i in paramIndex:
        parList.append(parListValues[i])

    if params == None:
        answer = "No Data Recieved"
        return str(answer), 201

    elif params != None:

        # ------ Converting to Float --------
        for i in parList:
            data.append(float(i))

        data[1] = RangeScaling(data[1])
        data[2] = RangeScaling(data[2])

        # ------ Normalizing The Input To The Network --------
        data[0] = (data[0] - (min_max_values[2])) / (
            (min_max_values[3]) - (min_max_values[2])
        )
        data[1] = (data[1] - (min_max_values[4])) / (
            (min_max_values[5]) - (min_max_values[4])
        )
        data[2] = (data[2] - (min_max_values[6])) / (
            (min_max_values[7]) - (min_max_values[6])
        )
        data[3] = (data[3] - (min_max_values[8])) / (
            (min_max_values[9]) - (min_max_values[8])
        )
        data[4] = (data[4] - (min_max_values[10])) / (
            (min_max_values[11]) - (min_max_values[10])
        )

        # ------ Creating A Dataframe Object --------
        df = pd.DataFrame(
            [data], columns=["Light", "Moisture 1", "Moisture 2", "Temp", "Humidity"]
        )
        properties = list(df.columns.values)
        x = df[properties]

        # ------ Passing The inputData To Tensorflow irrigationModel --------
        prediction = irrigationModel.predict(x)
        answer = round(prediction[0][0], 0)

        # ------ Returning The Answer --------
        return str(answer), 201


@app.route("/disease", methods=["GET", "POST"])

def predict():
    # ------ PreProcessing --------
    global diseaseModel

    message = request.get_json(force=True)
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    
    img = image.resize((224, 224))

    # Preprocessing the image
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode="tf")

    preds = diseaseModel.predict(x)
    # Process your result for human
    pred_proba = "{:.3f}".format(np.amax(preds))  # Max probability
    pred_class = decode_predictions(preds, top=1)  # ImageNet Decode

    result = str(pred_class[0][0][1])  # Convert to string
    result = result.replace("_", " ").capitalize()
    print(result)
    return jsonify(result=result, probability=pred_proba), 201


# start the flask app, allow remote connections
app.run(debug=True, host="0.0.0.0", port=4444)
