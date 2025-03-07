from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
import numpy as np 

# Load MNIST Data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# Accuracy Calculation
def compute_accuracy(v_xs,v_ys):
	global prediction # change prediction to global variable
	y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob: 1}) # feed training data
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob: 1})
	return result

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1) ## similar to normal distribution
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME') # x : whole image W : weight stride : how large for one step
	#Must have strides[0] = strides[3] = 1
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784]) # 1*784 input nodes
ys = tf.placeholder(tf.float32,[None,10]) # 10 output nodes
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1]) #[-1:ignore dimension 28,28,1 : black and white]

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # nonlinearize -> output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1) # output size 14*14*32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # nonlinearize -> output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2) #output size 7*7*64 

## func1 layer ##
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) # 3d to 2d
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# [n_samples,7,7,64] ->> [n_samples,7*7*64]
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

## func2 layer ##
# Loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Add ops to save and restore all the variables.
saver = tf.train.Saver(tf.all_variables())

#important step
sess = tf.Session()

# Don't initialize variables when restoring data
sess.run(tf.initialize_all_variables())

for i in range(2000):
	batch_xs, batch_ys = mnist.train.next_batch(100) # fetch 100 samples from database
	sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
	if i % 50 == 0:
		print(compute_accuracy(mnist.test.images, mnist.test.labels))

# Save the variables to disk.
save_path = saver.save(sess,"model.ckpt")
print("Model save in file: ", save_path)