# ------- IMPORTING LIBRARIES ---------
import tensorflow as tf
import csv
import pandas as pd

# ------- LIST ASSIGNMENT -------
min_max_values = []

# ------ READING MinMaxValue Csv Created During Normalization --------
with open(r'CSVs\\Target\\minMaxVals.csv','rt')as f:
  data = csv.reader(f)
  for row in data:
    min_max_values.append(row[0])

# ------ Converting to Float --------
for i in range(2,len(min_max_values)):
  min_max_values[i] = float(min_max_values[i])

# ------ This Round Of Data Input --------
#dataAfterNormalization = [0.086112261,0.081561131,0.088506927,0.220896881,0.214928276]
data = [85.37037037037038,172.9259259259259,205.22222222222226,29.27037037037037,17.185185185185187]

# ------ Normalizing The Input To The Network --------

data[0] = ((data[0] - (min_max_values[2])) / ((min_max_values[3])- (min_max_values[2])))
data[1] = ((data[1] - (min_max_values[4])) / ((min_max_values[5])- (min_max_values[4])))
data[2] = ((data[2] - (min_max_values[6])) / ((min_max_values[7])- (min_max_values[6])))
data[3] = ((data[3] - (min_max_values[8])) / ((min_max_values[9])- (min_max_values[8])))
data[4] = ((data[4] - (min_max_values[10])) /((min_max_values[11]) -( min_max_values[10])))

# ------ Creating A Dataframe Object --------
df = pd.DataFrame([data], columns = ["Light","Moisture 1","Moisture 2","Temp","Humidity"]) 
properties = list(df.columns.values)
inputData = df[properties]


# ------ Load The H5 Tensorflow Model --------
new_model = tf.keras.models.load_model(r"Tensorflow Models\\Only moisture--model.h5")

# ------ Passing The inputData To Tensorflow Model --------
prediction = (new_model.predict(inputData))
answer = round(prediction[0][0], 0)

print(answer)