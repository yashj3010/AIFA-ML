
# ------- IMPORTING LIBRARIES ---------
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ------ READING Dataset Into A DataFrame Object --------
df = pd.read_csv(r"CSVs\\Target\\Final.csv")

# ------ PREPARING Dataset For Feeding To Network --------
properties = list(df.columns.values)
properties.remove('Target')
properties.remove('TimeStamp')
X = df[properties]
y = df['Target']

# ------ Splitting Data --------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ------ Neural Network Model --------
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="adam",loss="BinaryCrossentropy", metrics=["accuracy"])

# ------ TRAINING The Model --------
model.fit(X_train,y_train, epochs=17, verbose=0)
val_loss, val_acc = model.evaluate(X_test,y_test)

# ------ SAVING The Model --------
model.save(r"Tensorflow Models\\Only moisture--model.h5")