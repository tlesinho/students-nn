import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


# Reading the dataset
def read_dataset():
    df = pd.read_csv("students-data2.csv", usecols=(0, 2, 3, 4, 5, 6))
    # print(len(df.columns))
    X = df[df.columns[0:5]].values
    y = df[df.columns[5]]
    print(X.shape)
    print(y.shape)
    # Encode the dependent variable
    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)
    Y = one_hot_encode(y)

    return (X, Y)


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


X, Y = read_dataset()
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=20, random_state=2)

learning_rate = 0.001
training_epochs = 300
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
print("n_dim", n_dim)
n_class = 5
model_path = "/Users/tles/PycharmProjects/Fyp/newmodel/saved_model.cktp"

# Define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 256
n_hidden_2 = 256
n_hidden_3 = 256
n_hidden_4 = 256

n_hidden_5 = 256
n_hidden_6 = 512
n_hidden_7 = 512

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])


# Define the model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    #
    # # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)

    # Hidden layer with RELU activation
    # layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    # layer_5 = tf.nn.sigmoid(layer_5)
    #
    # # Hidden layer with RELU activation
    # layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    # layer_6 = tf.nn.sigmoid(layer_6)
    #
    # # Hidden layer with RELU activation
    # layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    # layer_7 = tf.nn.sigmoid(layer_7)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


# Define the weights and the biases for each layer

weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    # 'h5': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5])),
    # 'h6': tf.Variable(tf.truncated_normal([n_hidden_5, n_hidden_6])),
    # 'h7': tf.Variable(tf.truncated_normal([n_hidden_6, n_hidden_7])),

    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class]))
}
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    # 'b5': tf.Variable(tf.truncated_normal([n_hidden_5])),
    # 'b6': tf.Variable(tf.truncated_normal([n_hidden_6])),
    # 'b7': tf.Variable(tf.truncated_normal([n_hidden_7])),

    'out': tf.Variable(tf.truncated_normal([n_class]))
}

# Initialize all the variables


saver = tf.train.Saver()

# Call your model defined
y = multilayer_perceptron(x, weights, biases)

# Define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Calculate the cost and the accuracy for each epoch

mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy: ", (sess.run(accuracy, feed_dict={x: test_x, y_: test_y})))
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)

    plt.plot(epoch + 1, cost, 'co')

    print('epoch : ', epoch, ' - ', 'cost: ', cost, " - MSE: ", mse_, "- Train Accuracy: ", accuracy)

save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % save_path)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = sess.run(accuracy, feed_dict={x: test_x, y_: test_y})

print("Test Accuracy: ", test_accuracy)
# Print the final mean square error
pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
final_mse = sess.run(mse)
print("MSE: %.4f" % sess.run(mse))

# Print the final accuracy
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Lr=%f, epoch=%d, mse=%.4f, acc=%.4f' % (learning_rate, training_epochs, final_mse, test_accuracy))
plt.tight_layout()
plt.savefig('./MLP-TF14-test.png', dpi=200)


tf.reset_default_graph()

# Re-initialize our two variables
# h_est = tf.Variable(h_est2, name='hor_estimate2')
# v_est = tf.Variable(v_est2, name='ver_estimate2')

x1 = tf.Variable(tf.zeros(395, 5), name="inputs")
y1 = tf.Variable(tf.zeros(395, 1), name="output")

# Create a builder
builder = tf.saved_model.builder.SavedModelBuilder('/Users/tles/PycharmProjects/Fyp/newmodel/saved_model')

# Add graph and variables to builder and save
with tf.Session() as sess:
    sess.run(x1.initializer)
    sess.run(y1.initializer)

    builder.add_meta_graph_and_variables(sess,
                                       [tf.saved_model.tag_constants.SERVING],
                                       signature_def_map=None,
                                       assets_collection=None)
builder.save()


# plt.show()
