import tensorflow as tf
import os
from model import *

if not os.path.exists('./logs'):
	os.makedirs('./logs')

map(os.unlink, (os.path.join( './logs',f) for f in os.listdir('./logs')) )


with tf.Session() as sess:
		print 'Building Graph...'
		net = HyperFace(sess,tf_record_file_path='aflw_train.tfrecords')
		print 'Graph Built!'
		sess.run(tf.global_variables_initializer())
		net.print_variables()
		# net.train()

