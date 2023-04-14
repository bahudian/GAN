from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
import io
import os
import requests
import numpy as np
from sklearn import metrics

df = pd.read_csv('mlb2022.csv')

# Drop rows with missing values
df = df.dropna()

# Define columns to use
COLS_USED = ['Age','G','PA','AB','R','H','2B','3B','HR','RBI','SB','CS','BB','SO','BA','OBP','SLG']
COLS_TRAIN = ['Age','G','PA','AB','H','2B','3B','HR','RBI','SB','CS','BB','SO','BA','OBP','SLG'] 

# Extract columns for training and testing
df_train = df[COLS_USED]

# Split into training and test sets
train, test = train_test_split(df_train, test_size=0.3)

# Split training and test sets into input and output data
x_train = train.drop(['R'], axis=1)
y_train = train['R']
x_test = test.drop(['R'], axis=1)
y_test = test['R']

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_dim=x_train.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Define early stopping callback
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=500, verbose=2, mode='auto',
        restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=1000, batch_size=32, callbacks=[monitor], verbose=2)


pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Final score (RMSE): {}".format(score))

