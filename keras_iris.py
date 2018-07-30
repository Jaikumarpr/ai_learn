from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# import data
dataset = pd.read_csv("dataset/iris/iris.csv")

x = dataset.iloc[:, 1:5].values
y = dataset.iloc[:, 5].values

# encode labels

le = LabelEncoder()
transformed_y = le.fit_transform(y)

encoded_y = np_utils.to_categorical(transformed_y, num_classes=3)

print(encoded_y)

# feature scaling
stdSclr = StandardScaler()

x = stdSclr.fit_transform(x)


# split the data in to test and train

x_train, x_test, y_train, y_test = train_test_split(x, encoded_y, test_size=0.2,
                                                    random_state=0)


print(x_train)

# create type of neural model

model = Sequential()

# input & first hidden layer
model.add(Dense(80, activation='relu', input_dim=4))
model.add(Dropout(0.5))

# second hidden layers

model.add(Dense(90, activation='relu'))
model.add(Dropout(0.5))

# third hidden layers

model.add(Dense(80, activation='tanh'))
model.add(Dropout(0.5))


# final layers

model.add(Dense(3, activation='softmax'))

# compile the models - binary classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
training = model.fit(x_train, y_train, batch_size=5, epochs=20, verbose=0)

loss, accuracy = model.evaluate(x_test, y_test, batch_size=2)

print("\n%s: %.2f%%" % (model.metrics_names[1], accuracy * 100))

model.save('kerasmodel.h5')

# plot_model(model, to_file='model.png')

# summarize history for accuracy
plt.subplots_adjust(hspace=0.5, wspace=0.7)
plt.subplot(211)
plt.plot(training.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)

# # summarize history for loss
plt.subplot(212)
plt.plot(training.history['loss'], color='r')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid(True)

plt.show()
