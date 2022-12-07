import tensorflow as tf
from keras import Sequential, layers
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

df = pd.read_csv('./csvfile/featrue.csv')
# print(df)
dataset = df.values
# print(dataset)
# print(dataset.ndim)
x = dataset[:, 0:8]        # 資料
y = dataset[:, 8]          # 標籤
# print(x)
# print(y)
mm_scale = preprocessing.MinMaxScaler()
x_scale = mm_scale.fit_transform(x)
# print(x_scale)
lb = LabelEncoder()
y = lb.fit_transform(y)               # 將label類別轉換為數字型態(如:0、1、2.....)
# print(y)

x_train, x_val_and_test, y_train, y_val_and_test = train_test_split(x_scale, y, test_size=0.3)
# print(x_train, y_train)
# train_test_split()功能只能將資料拆分成兩分，test_size->第二份資料拆成30%
# x_train:訓練集；x_val_and_test:驗證與測試集
x_val, x_test, y_val, y_test = train_test_split(x_val_and_test, y_val_and_test, test_size=0.5)
# 再使用train_test_split()將 x_val_and_test 拆分成> x_val:驗證集；x_test:測試集，將兩份資料拆成對半
print(x_train.shape, x_val.shape, x_test.shape, y_train.shape, y_val.shape, y_test.shape)
model = Sequential([
    layers.Dense(256, activation=tf.nn.relu, input_shape=(8,)),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(3, activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train,
          y_train,
          batch_size=100, epochs=100,
          shuffle=30, validation_data="x_val, y_val")

model.evaluate(x_test, y_test)
