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
df_x_train, df_x_test, df_y_train, df_y_test = train_test_split(
    df.drop("R", axis=1),
    df["R"],
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

# Split into training and test sets
#train, test = train_test_split(df_train, test_size=0.3)

# Split training and test sets into input and output data
#x_train = train.drop(['R'], axis=1)
#y_train = train['R']
#x_test = test.drop(['R'], axis=1)
#y_test = test['R']

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
        patience=5, verbose=1, mode='auto',
        restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          epochs=1000, batch_size=32, callbacks=[monitor], verbose=2)


pred = model.predict(x_test)
score = np.sqrt(metrics.mean_squared_error(pred,y_test))
print("Final score (RMSE): {}".format(score))

# Generate new players

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
          "epochs" : 500,}).generate_data_pipe(df_x_train,df_y_train,\
          df_x_test, deep_copy=True, only_adversarial=False, \
          use_adversarial=True)

gen_x
