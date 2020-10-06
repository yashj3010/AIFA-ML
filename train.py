import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv(r"/content/Final.csv")

#df['TimeStamp'] = pd.Categorical(df['TimeStamp'])
#df['TimeStamp'] = df.TimeStamp.cat.codes

properties = list(df.columns.values)
properties.remove('Target')
properties.remove('TimeStamp')
X = df[properties]
y = df['Target']

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train)
#checkpoint = ModelCheckpoint(r'{epoch:02d}-model-{accuracy:03f}.h5', verbose=0, monitor='accuracy',save_best_only=True, mode='auto')
#for i in range():
model.compile(optimizer="adam",loss="BinaryCrossentropy", metrics=["accuracy"])
model.fit(X_train,y_train, epochs=17, verbose=0)
val_loss, val_acc = model.evaluate(X_test,y_test)
print(val_acc)
model.save(f"Only moisture--model.h5")