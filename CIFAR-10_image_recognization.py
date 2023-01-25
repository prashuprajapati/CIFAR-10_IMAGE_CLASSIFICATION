import tensorflow as tf

# Display the version
print(tf.__version__)	

# other imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

cifar100=tf.keras.datasets.cifar100
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)  
# number of classes
num_class=len(set(y_train))

# calculate total number of classes
# for output layer
print("number of classes:", num_class)

# Build the model using the functional API
# input layer
i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)

# Hidden layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

# last hidden layer i.e.. output layer
x = Dense(num_class, activation='softmax')(x)

model = Model(i, x)

# compile the model 
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# fit the model
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10)

# predict the model 
ypred=model.predict(x_test)

# visulization data
import random as rd 
index_r=10000
index=rd.randint(1,index_r)
j=np.argmax(ypred[index])
b=[]
for i in range(100):
  b.append(0)
b[j]=1
plt.imshow(x_test[index])