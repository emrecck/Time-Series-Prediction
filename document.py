# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 18:47:01 2021

@author: Emre ÇİÇEK
"""

# KÜTÜPHANELERİN İMPORT EDİLMESİ

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters




register_matplotlib_converters()
sns.set(style='whitegrid',palette='muted',font_scale=1.5)

rcParams['figure.figsize'] = 22,10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# VERİ DOSYASININ OKUNMASI

df = pd.read_csv("london_merged.csv",parse_dates=['timestamp'],index_col="timestamp")

#%%
print(df.head())
print(df.columns)

# AY HAFTA GÜN VERİLERİNİN ÇEKİLMESİ

df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['month'] = df.index.month

print(df.head())

#%% 

# TÜM VERİLERİN GÖSTERİMİ

sns.lineplot(x=df.index,y='cnt', data=df)

#%%

# VERİLERİN AYLIK OLARAK GÖSTERİLMESİ

df_by_month = df.resample('M').sum()

sns.lineplot(x=df_by_month.index,y='cnt', data = df_by_month)


#%% VERİLERİN SAATLİK GÖSTERİMİ

sns.pointplot(data = df, x= 'hour', y='cnt')

#%% TATİLDE TALEP EDİLEN BİSİKLET SAYISI SAATLİK
 
sns.pointplot(data = df, x= 'hour', y='cnt',hue='is_holiday')


#%% HAFTALIK TALEP EDİLEN BİSİKLET SAYISI

sns.pointplot(data=df,x='day_of_week',y='cnt')


#%% SPLITTING DATA ( VERİYİ EĞİTİM VE TEST VERİLERİSİ OLARAK İKİYE AYIRMA )

train_size = int(len(df)*0.9)               # VERİNİN YÜZDE 90'INI İFADE EDER
test_size = len(df) - train_size            # GERİ KALAN YÜZDE 10 U İFADE EDER

train, test = df.iloc[0:train_size],df.iloc[train_size:len(df)]     # VERİLERİ AYIRMA 

print(train.shape,test.shape)


#%% PREPROCESS

from sklearn.preprocessing import RobustScaler

f_columns = ['t1','t2','hum','wind_speed']

f_transformer = RobustScaler()

cnt_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())

cnt_transformer = cnt_transformer.fit(train[['cnt']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['cnt'] = cnt_transformer.transform(train[['cnt']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['cnt'] = cnt_transformer.transform(test[['cnt']])



#%% CREATE DATASET

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    
    for i in range(len(X) - time_steps):
        v = X.iloc[i: (i + time_steps)].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i + time_steps]) 
    return np.array(Xs), np.array(ys)

#%% 

TIME_STEPS = 24

X_train, y_train = create_dataset(train, train.cnt, time_steps = TIME_STEPS)
X_test, y_test = create_dataset(test, test.cnt, time_steps= TIME_STEPS)

#[samples, time_steps, n_features]


print(X_train.shape,y_train.shape)

#%%

print(X_test.shape,y_test.shape)

print(X_test[0][0].shape)

#%% BIDRECTIONAL LSTM MODEL

# both past-future

model = keras.Sequential()
model.add(
    keras.layers.Bidirectional(
        keras.layers.LSTM(                                              # LSTM KATMANI , 128 NÖRON İNPUT_SHAPE =  24,13
            units = 128,
            input_shape = (X_train.shape[1], X_train.shape[2])
            )
        )
    )
model.add(
    keras.layers.Dropout(
        rate = 0.2                                                      # DROPOUT KATMANI , ORAN = %20
        )
    )
model.add(
    keras.layers.Dense(                                                 # DENSE KATMANI , 1 NÖRON         
        units=1
        )
    )

opt = Adam(lr=0.0004)
model.compile(loss = 'mean_squared_error', optimizer = opt)         # MSE LOSS FONKSİYONU VE ADAM OPTİMİZER TERCİH EDİLMİŞTİR.

#%%

# MODELİN EĞİTİLMESİ

es = EarlyStopping(monitor='val_loss', mode='min', patience=20,
                   verbose=0)

history = model.fit(
    X_train,y_train,
    epochs = 30,                # EĞİTİM 30 EPOCH SÜRECEKTİR
    callbacks=[es],
    batch_size = 32,            # BATCH_SİZE'IMIZ 32 DİR
    validation_split = 0.1,     # VALİDATİON VERİSİ OLARAK YÜZDE 10 LUK KISIM AYRILMIŞTIR
    shuffle = False             # VERİLERİN SHUFFLE YAPILMASINI İSTEMİYORUZ.
    )



#%%  LOSS VE VALİDATİON LOSS BİLGİLERİNİ PLOTLAMA

plt.plot(history.history['loss'], label = 'train')
plt.plot(history.history['val_loss'], label = 'validation')
plt.legend()

#%% MODELİN TEST VERİLERİYLE TAHMİN YAPMASI 

y_pred = model.predict(X_test)


#%%

y_train_inv = cnt_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cnt_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = cnt_transformer.inverse_transform(y_pred)



#%%

plt.plot(y_test_inv.flatten(), marker = '.', label = 'true')
plt.plot(y_pred_inv.flatten(), 'r', marker = '.', label = 'predicted')
plt.legend()










