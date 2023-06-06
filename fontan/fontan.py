import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import io
import os
import requests
from sklearn import metrics


df = pd.read_csv('fontan_data.csv')
print(df.columns)

# Convert VENTDOM to categorical

df["VENTDOM"] = df["VENTDOM"].astype('category').cat.codes

# Drop rows with missing values
df.dropna()

# Define columns to use

COLS_USED = ['VENTDOM', 'COMP_AGE', 'FONTAN_AGE', 'NUMASSOC', 'NUMPROC', 'FENESTRN', 'NUMSURG', 'NUM_CATH', 'NUMSTRKE', 'SEIZURE', 'NUMTHRM', 'PLE', 'NUMARRHY', 'echobsa', 'echosv', 'dpdt', 'tde', 'tda', 'BNP']

COLS_TRAIN = ['VENTDOM', 'COMP_AGE', 'NUMASSOC', 'NUMPROC', 'FENESTRN', 'NUMSURG', 'NUM_CATH', 'NUMSTRKE', 'SEIZURE', 'NUMTHRM', 'PLE', 'NUMARRHY', 'echobsa', 'echosv', 'dpdt', 'tde', 'tda', 'BNP']

# Extract columns for training and testing
df_train = df[COLS_USED]

# Split into training and test sets
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
    df.drop("FONTAN_AGE", axis=1),
    df["FONTAN_AGE"],
    test_size=0.20,
    #shuffle=False,
    random_state=42,
)

# Create dataframe versions for tabular GAN
df_x_test, df_y_test = df_x_test.reset_index(drop=True), \
  df_y_test.reset_index(drop=True)
df_y_train = pd.DataFrame(df_y_train)
df_y_test = pd.DataFrame(df_y_test)

# Pandas to Numpy
x_train = df_x_train.values
x_test = df_x_test.values
y_train = df_y_train.values
y_test = df_y_test.values

# Define the neural network model
model = Sequential([
    Dense(64, activation='relu', input_dim=x_train.shape[1]), Dropout(0.3),
    Dense(32, activation='relu'), Dropout(0.3),
    Dense(16, activation='relu'), Dropout(0.3),
    Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='rmsprop')

# Define early stopping callback
monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3,
        patience=500, mode='auto',
        restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=1000, batch_size=32, callbacks=[monitor])


pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Final score (RMSE): {}".format(score))

# Generate new patients

from tabgan.sampler import GANGenerator


gen_x, gen_y = GANGenerator(gen_x_times=1.1, cat_cols=None,
           bot_filter_quantile=0.001, top_filter_quantile=0.999, \
              is_post_process=True,
           adversarial_model_params={
               "metrics": "rmse", "max_depth": 2, "max_bin": 100,
               "learning_rate": 0.02, "random_state": \
                42, "n_estimators": 500,
           }, pregeneration_frac=2, only_generated_data=False,\
           gan_params = {"batch_size": 500, "patience": 25, \
          "epochs" : 500,}).generate_data_pipe(df_x_train, df_y_train,\
          df_x_test, deep_copy=True, only_adversarial=False, \
          use_adversarial=True)

gen_x
gen_y





