import os
import matplotlib.pyplot as plt
import tensorflow as tf

new_model = tf.keras.models.load_model('os.getcwd/' + 'test.h5')

print(new_model.keys())
