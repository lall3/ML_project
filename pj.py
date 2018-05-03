#tensor project
import tensorflow as tf
import numpy as np
import math
import csv


#gloabls
training_images=[]
training_labels=[]

#number of neurons on each level
neurons=[ 128, 64, 16]

#learning rate
learning_rate= 0.05

size=280


data_read=[]
data_len=0
cutoff=0

#weight and bias initialization functions
weight_init = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=1)
bias_initializer= tf.zeros_initializer()

def read_images(filename):
	#ret val
	images=[]
	file = open(filename, "r")
	ctr = 0
	curr_image=[]


	for line in file:
		# 28 lines per digit
		if ctr == 28 :
			ctr = 0
			images.append(curr_image)
			curr_image=[]

		ctr+=1

		for chr in list(line):
			if chr == "#" or chr == '+' :
				curr_image.append(1.0)
			else:
				curr_image.append(0.0)	

	images.append(curr_image)
	return images

#
# returns values that are stored in file
#
def read_vals(filename):
	values=[]
	with open(filename, 'r') as file:
		for line in file:
			values.append(int(list(line)[0]))

	return values


	

#cr5eates datas set from labels and images
def build_train(labels , imgs):
	dataset= tf.data.Dataset.from_tensor_slices( images, labels )
	#one can shuffle also
	return dataset


def work():
	#place holders are input

	global neurons, training_images, training_labels, weight_init, bias_initializer, size, data_len, cutoff

	epochs= 1

	Input1= tf.placeholder(dtype= tf.float32 , shape=[None, 6], name='Intput1') #check
	Target= tf.placeholder(dtype= tf.float32 , shape=[None], name= 'Target')

	x = tf.Variable(1, dtype='float')




	#layer weights
	w_layer1= tf.Variable( weight_init([6, neurons[0]]) )
	b_layer1= tf.Variable( bias_initializer(neurons[0]) )

	w_layer2= tf.Variable( weight_init([ neurons[0], neurons[1]]) )
	b_layer2= tf.Variable( bias_initializer(neurons[1]) )

	w_layer3= tf.Variable( weight_init([ neurons[1], neurons[2]]) )
	b_layer3= tf.Variable( bias_initializer(neurons[2]) )

	w_layer_o= tf.Variable( weight_init([ neurons[2], 1]) )
	b_layer_o= tf.Variable( bias_initializer(1) )

	'''w_layer4= tf.Variable( weight_init(size, neurons[0]) )
	b_layer4= tf.Variable( bias_initializer(neurons[0]) )'''

	layer1 = tf.nn.relu(tf.add(tf.matmul(Input1, w_layer1), b_layer1))

	layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, w_layer2), b_layer2))

	layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, w_layer3), b_layer3))

	layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, w_layer_o), b_layer_o))

	output = tf.convert_to_tensor(layer4, dtype=tf.float32)

	cost = tf.reduce_mean(tf.squared_difference(Target, output))

	optimizer = tf.train.AdamOptimizer().minimize(cost)



	#testImages = read_images('testimages')
	#testVals = read_vals("testlabels")

	counter = 0
	acc = []
	#total = 0 
	#test_l = len(testImages)
	cutoff = data_len-100	

	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		for i in range(10):
			for k in range(cutoff):
				a=[]
				a.append(data_read[k])
				b=[]
				b.append(data_read[k+1][3])
				#print(a,b)
				err, waste = sess.run([cost, optimizer], feed_dict={Input1: a, Target: b})
				
			

			counter = 0
			'''
			for k in range(cutoff, data_len-3):
				a=[]
				a.append(data_read[k])
				x = sess.run( [output], feed_dict={Input1: a} )

				predicted_output = (x[0].item(0)) #x is numpy.ndarray
				#abs_output = int(round(predicted_output))
				#print(type(predicted_output))
				#print(predicted_output, abs_output ,testVals[k])

				print(predicted_output, data_read[k+1][3] )'''
				
	
			#acc.append( float(counter*100) / test_l )



		for k in range(cutoff, data_len-3):
			a=[]
			a.append(data_read[k])
			x = sess.run( [output], feed_dict={Input1: a} )

			print(type(x))

			predicted_output = (x[0].item(0)) #x is numpy.ndarray
				

			print(x, data_read[k+1][3] )

				
















def main():
	global training_images,training_labels

	'''training_images =  read_images('trainingimages')
	print(len(training_images))
	training_labels = read_vals('traininglabels')'''

	work()




	
with open('data.csv', 'rb') as csvfile:
	read= csv.reader(csvfile)
	for row in read:
		data_len+=1
		data_read.append(row[1:]) 

cutoff = data_len-100	

value_data =[]

data_read =data_read [1:]
for row in data_read:
	t=[]

	for c in row:
		t.append(float(c))
	value_data.append(t)

data_read = value_data		


main()