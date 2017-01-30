import tensorflow as tf 
import numpy as np 
import cv2

def convolutional(input):
	#Read input mnist image
	input_image_old= cv2.imread(input,0)
	input_image = cv2.resize(input_image_old,(28,28), interpolation = cv2.INTER_LINEAR)

	def weight_variable(shape):
		initial = tf.truncated_normal(shape,stddev=0.1) ## similar to normal distribution
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1,shape=shape)
		return tf.Variable(initial)

	def conv2d(x,W):
		# stride [1,x_movement, y_movement]  # Padding : Valid Padding , Same Padding
		return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding = 'SAME') # x : whole image W : weight stride : how large for one step
		#Must have strides[0] = strides[3] = 1
	def max_pool_2x2(x):
		return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

	#define placeholder for inputs to network
	xs = tf.placeholder(tf.float32,[None,784]) #28*28
	ys = tf.placeholder(tf.float32,[None,10]) # every sample has ten outputs
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
	# Cost function
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver(tf.all_variables())

	#important step
	sess = tf.Session()

	# Restore variables from disk
	saver.restore(sess,"model.ckpt")
	print("Model restore.")	

	input_image = input_image / 255.0
	input_image = np.reshape(input_image,(1,784))

	# Predict the input image number
	y_pre = sess.run(prediction,feed_dict={xs:input_image,keep_prob: 1}) # feed training data
	yp =  np.argmax(y_pre,1)
	return yp

if __name__ == '__main__':
	pre = convolutional('3.png')
	s_pre = 'predicted label is : '
	print(s_pre + str(pre))