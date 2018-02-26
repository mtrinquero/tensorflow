# Mark Trinquero
# Tensorflow - Neural Network
# Overview: neural network for recognition of handwritten numerical images (0,1,2,3,4,5,6,7,8,9)
# Data Source: uses the MNIST data set of 70,000 labeled images from US census handwriting samples
# Works Cited: adapted from the book Neural Networks, A Visual Introduction 
# https://www.tensorflow.org/tutorials/layers
# https://www.tensorflow.org/programmers_guide/datasets

import numpy as np 
import tensorflow as tf

# download and read MNIST handwriting data
# http://yann.lecun.com/exdb/mnist/
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# define general parameters
learning_rate = 0.0001
batch_size = 100
update_step = 10

# define network parameters: 
## 3 hidden layers with 500 nodes per layer
## output = 10 nodes (one for each possible digit 0-9)
## input= 2d images from MNIST dataset (input images are 28pixels x 28 pixels = 784pixels)
layer1_nodes = 500
layer2_nodes = 500
layer3_nodes = 500
output_nodes = 10
input_nodes = 784

# setup placeholders for the network's input layer and target output layer (tensors)
# datatype= float32, length=None (will be the num images in a batch), width=784 (input images are 28pixels x 28 pixels = 784)
network_input = tf.placeholder(tf.float32, [None, input_nodes])
target_output = tf.placeholder(tf.float32, [None, output_nodes])

# creating the network model (3 layers) - define the parameters that the network will adjust as it trains (weights, biases, values)
# layer 1 setup - intially assign edge weights using a random normal variable with shape [input_nodes, layer1_nodes] 
# corresponding to the num of adjustable params (or edge weights) between the network_input and the first hidden layer_1
layer_1 = tf.Variable(tf.random_normal([input_nodes, layer1_nodes]))
layer_1_bias = tf.Variable(tf.random_normal([layer1_nodes]))
# layer 2 setup - update tf.random_normal to adjust shape as needed to match where we are in the network
layer_2 = tf.Variable(tf.random_normal([layer1_nodes, layer2_nodes]))
layer_2_bias = tf.Variable(tf.random_normal([layer2_nodes]))
# layer 3 setup - update tf.random_normal to adjust shape as needed to match where we are in the network
layer_3 = tf.Variable(tf.random_normal([layer2_nodes, layer3_nodes]))
layer_3_bias = tf.Variable(tf.random_normal([layer3_nodes]))
# output layer setup
out_layer = tf.Variable(tf.random_normal([layer3_nodes, output_nodes]))
out_layer_bias = tf.Variable(tf.random_normal([output_nodes]))


# configure the FeedForward calculations needed to move inputs thru the network
# utilizes the ReLu activation operation, and matmul matrix multiplication operation
# ReLu Activation: https://www.tensorflow.org/api_docs/python/tf/nn/relu
# MatMul Operation: https://www.tensorflow.org/api_docs/python/tf/matmul (Matrix A x Matrix B)
l1_output = tf.nn.relu(tf.matmul(network_input, layer_1) + layer_1_bias)    #calculates the output of every node in the first hidden layer
l2_output = tf.nn.relu(tf.matmul(l1_output, layer_2) + layer_2_bias)        #calculates the output of every node in the second hidden layer
l3_output = tf.nn.relu(tf.matmul(l2_output, layer_3) + layer_3_bias)        #calculates the output of every node in the third hidden layer

# calculate the raw final output of the network then use tf's Softmax activation operation to calculate the 0-1 probabilities from the final output
ntwk_output_1 = tf.matmul(l3_output, out_layer) + out_layer_bias    #the un-scaled output of the network
ntwk_output_2 = tf.nn.softmax(ntwk_output_1)                        #the scaled output of the network

# Define Training Elements for the network
# cf --> Cost Function (computes the cost, or loss/error, of the network)
cf = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=ntwk_output_1, labels=target_output))
# ts --> Training Step (the size of step, or distance, that the network takes towards minimizing the cost function cf)
ts = tf.train.GradientDescentOptimizer(learning_rate).minimize(cf)
# cp --> Correct Predictions (evaluates which predictions of the network are correct) is an array of True/False statements
cp = tf.equal(tf.argmax(ntwk_output_2, 1), tf.argmax(target_output, 1))
# acc --> Average number of correct predictions (Trues) from above 
acc = tf.reduce_mean(tf.cast(cp, tf.float32))

# Create a tensorflow Session - setup a loop that trains the network and prints updates along the way
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    num_epochs = 10   #'epoch' = defined as when an entire training set goes forward and backward thru the network
    for epoch in range(num_epochs):
        total_cost = 0  #setup variable to track the total cost, updated each time an epoch is completed
        for _ in range(int(mnist.train.num_examples / batch_size)):
            batch_x, batch_y = mnist.train.next_batch(batch_size)    #batch_x stores training examples, and batch_y stores the corresponding target for each example
            t, c = sess.run([ts, cf], feed_dict={network_input: batch_x, target_output: batch_y})   #t--> training step, c--> cost function    
            total_cost += c   #update the total cost with each iteration
        print('Epoch #', epoch, 'completed out of', num_epochs, 'with loss:', total_cost)
    # output the accuracy of the network once all the epochs are complete
    print('Final Network Accuracy:', acc.eval({network_input: mnist.test.images,target_output: mnist.test.labels}))
# session will close once all evaluations are complete


# Note: test inside tensorflow virtualenv 