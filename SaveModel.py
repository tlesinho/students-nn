import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tf.reset_default_graph()

# Re-initialize our two variables
# h_est = tf.Variable(h_est2, name='hor_estimate2')
# v_est = tf.Variable(v_est2, name='ver_estimate2')

x = tf.Variable(tf.zeros(395, 5))
y = tf.Variable(tf.zeros(395, 1))

# Create a builder
builder = tf.saved_model.builder.SavedModelBuilder('/Users/tles/PycharmProjects/Fyp/newmodel/saved_model')

# Add graph and variables to builder and save
with tf.Session() as sess:
    sess.run(x.initializer)
    sess.run(y.initializer)
    builder.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.TRAINING],
                                       signature_def_map=None,
                                       assets_collection=None)
builder.save()
