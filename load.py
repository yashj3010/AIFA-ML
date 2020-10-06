import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

lightMin = 0
lightMax = 991.3385

moisture1Min = 135
moisture1Max = 600

moisture2Min = 166.889
moisture2Max = 600

tempMin = 21.3167
tempMax = 57.3231

humidityMin = 9
humidityMax = 47.0833

#data = [0.086112261,0.081561131,0.088506927,0.220896881,0.214928276]
data = [85.37037037037038,172.9259259259259,205.22222222222226,29.27037037037037,17.185185185185187]

data[0] = ((data[0] - lightMin) / (lightMax - lightMin))
data[1] = ((data[1] - moisture1Min) / (moisture1Max - moisture1Min))
data[2] = ((data[2] - moisture2Min) / (moisture2Max - moisture2Min))
data[3] = ((data[3] - tempMin) / (tempMax - tempMin))
data[4] = ((data[4] - humidityMin) / (humidityMax - humidityMin))

print(data)

df = pd.DataFrame([data], columns = ["Light","Moisture 1","Moisture 2","Temp","Humidity"]) 
properties = list(df.columns.values)
X = df[properties]
#print(X)

new_model = tf.keras.models.load_model(r"/content/Only moisture--model.h5")

acc = (new_model.predict(X))
x = round(acc[0][0], 0)
print(x)