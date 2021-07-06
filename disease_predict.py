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

def get_model():
    global model
    model = load_model('DiseaseIdentification.h5')
    #model._make_predict_function()
    print(" * Model loaded!")

def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds
    
print(" * Loading Keras model...")
get_model()

@app.route("/predict", methods=["POST"])
def predict():
    global model
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    preds = model_predict(img, model)
    #processed_image = preprocess_image(image, target_size=(224, 224))
    # Process your result for human
    pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
    pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

    result = str(pred_class[0][0][1])               # Convert to string
    result = result.replace('_', ' ').capitalize()

    #prediction = model.predict(processed_image).tolist()

    return jsonify(result=result, probability=pred_proba)
    