import csv
import os
import pandas as pd
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

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

model =tf.keras.models.load_model(r'Tensorflow Models\\PlantDNet.h5',compile=False)

irrigationModel = tf.keras.models.load_model(r"Tensorflow Models\\irrigation.h5")

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

# Define a flask app
app = Flask(__name__)


def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);
        disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                         'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                         'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                         'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                         'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
        a = preds[0]
        ind=np.argmax(a)
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        return result
    return None

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


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 4448), app)
    http_server.serve_forever()
    app.run()
