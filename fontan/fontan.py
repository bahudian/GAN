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

df["VENTDOM"] = df["VENTDOM"].astype('category').cat.codes

COMP_AGE_MIN = df["COMP_AGE"].min()
COMP_AGE_MAX = df["COMP_AGE"].max()

FONTAN_AGE_MIN = df["FONTAN_AGE"].min()
FONTAN_AGE_MAX = df["FONTAN_AGE"].max()

echobsa_MIN = df["echobsa"].min()
echobsa_MAX = df["echobsa"].max()

echosv_MIN = df["echosv"].min()
echosv_MAX = df["echosv"].max()

dpdt_MIN = df["dpdt"].min()
dpdt_MAX = df["dpdt"].max()

tde_MIN = df["tde"].min()
tde_MAX = df["tde"].max()

tda_MIN = df["tda"].min()
tda_MAX = df["tda"].max()

BNP_MIN = df["BNP"].min()
BNP_MAX = df["BNP"].max()


df_COMP_AGE = pd.cut(df['COMP_AGE'], bins = np.linspace(COMP_AGE_MIN, COMP_AGE_MAX, 21), labels=False)
df_FONTAN_AGE = pd.cut(df['FONTAN_AGE'], bins = np.linspace(FONTAN_AGE_MIN, FONTAN_AGE_MAX, 21), labels=False)
df_echobsa = pd.cut(df['echobsa'], bins = np.linspace(echobsa_MIN, echobsa_MAX, 21), labels = False)
df_echosv = pd.cut(df['echosv'], bins = np.linspace(echosv_MIN, echosv_MAX, 21), labels = False)
df_dpdt = pd.cut(df['dpdt'], bins = np.linspace(dpdt_MIN, dpdt_MAX, 21), labels = False)
df_tde = pd.cut(df['tde'], bins = np.linspace(tde_MIN, tde_MAX, 21), labels=False)
df_tda = pd.cut(df['tda'], bins = np.linspace(tda_MIN, tda_MAX, 21), labels=False)
df_BNP = pd.cut(df['BNP'], bins = np.linspace(BNP_MIN, BNP_MAX, 21), labels=False)

df.drop(["COMP_AGE", "FONTAN_AGE", "echobsa", "echosv", "dpdt", "tde", "tda", "BNP"], axis=1, inplace=True)

df = pd.concat([df, df_COMP_AGE, df_FONTAN_AGE, df_echobsa, df_echosv, df_dpdt, df_tde, df_tda, df_BNP], axis=1)

df[df.columns] = PowerTransformer(method='yeo-johnson', standardize=True, copy=True).fit_transform(df[df.columns])

df.dropna()


