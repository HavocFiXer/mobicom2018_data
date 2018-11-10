#/usr/bin/python
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import random as rd
import sys
import os
from sklearn.metrics import confusion_matrix

order=int(sys.argv[1])

# data: (38400, 30, 128, 4)
# activity label: (38400, 5)
# subject label: (38400, 40)
# 38400=40x960
# train number: 22*960=21120
# test  number: 18*960=17280

train_size=21120
test_size=38400
batch_size=128 # 128*135=17280
batch_number=200000
write_slot=500
category_size=6
sub_size=22
test_batch_number=(test_size-train_size)//batch_size

lr=0.0001
do_prob=1.0
cnn_do_prob=1.0

infilename='./data/databel.dat'
keyword='3cnn_s-1fc.2fc_3ad.cnn-bn.entropy.order%d'%(order)
savefolder='save'
os.system('mkdir %s'%(savefolder))

def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial=tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')

def max_pool(x):
	return tf.nn.max_pool(x, ksize=[1,1,2,1], strides=[1,1,2,1], padding='SAME')

def next_batch():
	global trainingset, traininglabel, trainingsub, batch_size
	spnumber=rd.sample(xrange(len(trainingset)), batch_size)
	batch_x=[trainingset[x] for x in spnumber]
	batch_y=[traininglabel[x] for x in spnumber]
	batch_s=[trainingsub[x] for x in spnumber]
	return batch_x, batch_y, batch_s

#read data and label
datafile=open(infilename,'rb')
datall=np.load(datafile) #(38400, 30, 128, 4)
labelall=np.load(datafile) #(38400, 6)
sub_labelall=np.load(datafile)[:, :sub_size] #(38400, 22)
datafile.close()
print 'Done with the reading.'
print datall.shape
print labelall.shape
print sub_labelall.shape

#training and testing
trainingset=datall[:train_size]
traininglabel=labelall[:train_size]
trainingsub=sub_labelall[:train_size]
print 'shape of train:', np.array(trainingset).shape, np.array(traininglabel).shape, np.array(trainingsub).shape
testingset=datall[train_size:]
testinglabel=labelall[train_size:]
print 'shape of test:', np.array(testingset).shape, np.array(testinglabel).shape

x=tf.placeholder(tf.float32, [None, 30, 128, 4])
y=tf.placeholder(tf.float32, [None, category_size])
s=tf.placeholder(tf.float32, [None, sub_size])
is_train=tf.placeholder(tf.bool)
keep_prob=tf.placeholder(tf.float32)
cnn_keep_prob=tf.placeholder(tf.float32)

# conv1
w_conv1=weight_variable([30, 4, 4, 64])
b_conv1=bias_variable([64])
c_conv1=conv2d(x, w_conv1)+b_conv1
n_conv1=tf.layers.batch_normalization(c_conv1, training=is_train)
h_conv1=tf.nn.relu(n_conv1)
d_conv1=tf.nn.dropout(h_conv1, cnn_keep_prob)
p_conv1=max_pool(d_conv1)

# conv2
w_conv2=weight_variable([1, 4, 64, 128])
b_conv2=bias_variable([128])
c_conv2=conv2d(p_conv1, w_conv2)+b_conv2
n_conv2=tf.layers.batch_normalization(c_conv2, training=is_train)
h_conv2=tf.nn.relu(n_conv2)
d_conv2=tf.nn.dropout(h_conv2, cnn_keep_prob)
p_conv2=max_pool(d_conv2)

# conv3
w_conv3=weight_variable([1, 4, 128, 256])
b_conv3=bias_variable([256])
c_conv3=conv2d(p_conv2, w_conv3)+b_conv3
n_conv3=tf.layers.batch_normalization(c_conv3, training=is_train)
h_conv3=tf.nn.relu(n_conv3)
d_conv3=tf.nn.dropout(h_conv3, cnn_keep_prob)
p_conv3=max_pool(d_conv3) #(None, 30, 16, 256)

# flatten
x_flat=tf.reshape(p_conv3, [-1, 30*16*256])

# fc1
w_fc1=weight_variable([30*16*256, 128])
b_fc1=bias_variable([128])
h_fc1=tf.nn.relu(tf.matmul(x_flat, w_fc1)+b_fc1)
d_fc1=tf.nn.dropout(h_fc1, keep_prob)

#fc2
#w_fc2=weight_variable([256, 64])
#b_fc2=bias_variable([64])
#h_fc2=tf.nn.relu(tf.matmul(d_fc1, w_fc2)+b_fc2)
#d_fc2=tf.nn.dropout(h_fc2, keep_prob)

# discriminator
w_dfc1=weight_variable([30*16*256, 128])
b_dfc1=bias_variable([128])
h_dfc1=tf.nn.relu(tf.matmul(x_flat, w_dfc1)+b_dfc1)
d_dfc1=tf.nn.dropout(h_dfc1, keep_prob)
w_sub=weight_variable([128, sub_size])
b_sub=bias_variable([sub_size])
y_sub=tf.matmul(d_dfc1, w_sub)+b_sub

var_sub=[w_dfc1, b_dfc1, w_sub, b_sub]
cs_sub=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_sub, labels=s))

# softmax
w_softmax=weight_variable([128, category_size])
b_softmax=bias_variable([category_size])
y_softmax=tf.matmul(d_fc1, w_softmax)+b_softmax

var_softmax=[w_conv1, b_conv1, w_conv2, b_conv2, w_conv3, b_conv3, w_fc1, b_fc1, w_softmax, b_softmax]
model_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_sub, labels=tf.nn.softmax(y_sub)))
entropy_model=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_softmax, labels=y))
cross_entropy=model_entropy+entropy_model#-cs_sub

#remember to update bn parameters
extra_update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
	ts_sub=tf.train.AdamOptimizer(learning_rate=lr).minimize(cs_sub, var_list=var_sub)
	train_step=tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy, var_list=var_softmax)

#accuracy
y_true=tf.argmax(y, 1)
y_pred=tf.argmax(y_softmax, 1)
correct_prediction=tf.equal(y_pred, y_true)
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#set memory to enough
config=tf.ConfigProto()
config.gpu_options.allow_growth=True

#session
sess=tf.Session(config=config)
init=tf.global_variables_initializer()
sess.run(init)

	

outfile=open('./%s/%s.rst'%(savefolder,keyword),'w')
test_sum=0.0
test_ce=0.0
for k in xrange(test_batch_number):
	result=sess.run([accuracy, model_entropy], feed_dict={x:testingset[k*batch_size: (k+1)*batch_size], y:testinglabel[k*batch_size: (k+1)*batch_size], keep_prob:1.0, cnn_keep_prob: 1.0, is_train: False})
	test_sum+=result[0]
	test_ce+=result[1]
test_sum/=test_batch_number
test_ce/=test_batch_number
print '%d, %f, %f'%(0, test_sum, test_ce)
outfile.write('%d, %f, %f\n'%(0, test_sum, test_ce))
outfile.flush()

for i in xrange(1, batch_number+1):
	for _ in xrange(3):
		batch_x, batch_y, batch_s=next_batch()
		sess.run(ts_sub, feed_dict={x:batch_x, y:batch_y, keep_prob: do_prob, cnn_keep_prob: cnn_do_prob, s:batch_s, is_train: True})
	sess.run(train_step, feed_dict={x:batch_x, y:batch_y, keep_prob: do_prob, cnn_keep_prob: cnn_do_prob, s:batch_s, is_train: True})
	if i%write_slot==0:
		test_sum=0.0
		test_ce=0.0
		#y_true_list=[]
		#y_pred_list=[]
		for k in xrange(test_batch_number):
			result=sess.run([accuracy, model_entropy], feed_dict={x:testingset[k*batch_size: (k+1)*batch_size], y:testinglabel[k*batch_size: (k+1)*batch_size], keep_prob:1.0, cnn_keep_prob: 1.0, is_train: False})
			test_sum+=result[0]
			test_ce+=result[1]
			#y_pred_true.append(result[1])
			#y_pred_list.append(result[2])
		test_sum/=test_batch_number
		test_ce/=test_batch_number
		print '%d, %f, %f'%(i, test_sum, test_ce)
		outfile.write('%d, %f, %f\n'%(i, test_sum, test_ce))
		outfile.flush()

outfile.close()
