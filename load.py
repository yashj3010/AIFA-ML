import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

data = [0.086112261,0.081561131,0.088506927,0.220896881,0.214928276] 
  
df = pd.DataFrame([data], columns = ["Light","Moisture 1","Moisture 2","Temp","Humidity"]) 

properties = list(df.columns.values)
X = df[properties]
print(X)

new_model = tf.keras.models.load_model(r"/content/Only moisture--model.h5")

acc = (new_model.predict(X))
x = round(acc[0][0], 0)
print(x)