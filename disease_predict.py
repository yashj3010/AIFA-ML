import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

diseaseModel = load_model(
    r"Tensorflow Models\DiseaseIdentification.h5"
)

def model_predict(img):
    global diseaseModel
    img = img.resize((224, 224))

    # Preprocessing the image
    x = img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')

    preds = diseaseModel.predict(x)
    print(preds)
    return preds
    
print(" * Loading Keras model...")

@app.route("/")
def index():
    return "Welcome TO AIFA Disease BackEnd!"

@app.route("/predict", methods=["POST"])
def predict():
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

app.run(debug=True, host="0.0.0.0", port=4445)