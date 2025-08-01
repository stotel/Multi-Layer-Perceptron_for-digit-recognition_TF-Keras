import random

import keras.src.layers
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
import seaborn as sn

from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from tensorflow.python.keras.models import save_model

import plotter

SEED_VALUE = 1

# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(X_test.shape)

# plotter.show_images_from_dataset(X_train,4,4)

# reshape datasets from 3d to 2d
X_train = X_train.reshape((X_train.shape[0], 28 * 28))
X_train = X_train.astype("float32") / 255

print('single input shape')
print(X_train.shape[1])

X_test = X_test.reshape((X_test.shape[0], 28 * 28))
X_test = X_test.astype("float32") / 255


# convert labels to one-hot vectors
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = tf.keras.models.Sequential()

model.add(keras.src.layers.Input((X_train.shape[1],)))
model.add(keras.src.layers.Dense(128,"relu",True))
model.add(keras.src.layers.Dense(128,"relu",True))
model.add(keras.src.layers.Dense(10,"softmax",True))

model.summary()

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

training_results = model.fit(X_train,
                             y_train,
                             epochs=21,
                             batch_size=64,
                             validation_split=0.2);

plotter.show_results(training_results)

#preds for test
predictions = model.predict(X_test)

#one prediction
predicted_labels = [np.argmax(i) for i in predictions]

#one-hot to number
y_test_integer_labels = tf.argmax(y_test, axis=1)

cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)

# plot confusion matrix
plt.figure(figsize=[10, 10])
sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 14})
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()

#save model
save_model_decision = input("save model?(y/n)")
if save_model_decision == "y":
    name = input("input filename (model by default)")
    if name == "":
        model.save("Models/model.keras")
    else:
        model.save("Models/"+name+".keras")
print("finished executing")