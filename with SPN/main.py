import tensorflow as tf
import os
from model import *

weights_path = '/Users/shashank/Tensorflow/SPN/weights/'
imgs_path = '/Users/shashank/Tensorflow/CSE252C-Hyperface/git/truth_data.npy'

if not os.path.exists('./logs'):
	os.makedirs('./logs')

map(os.unlink, (os.path.join( './logs',f) for f in os.listdir('./logs')) )


with tf.Session() as sess:
		print 'Building Graph...'
		model = Network(sess)
		print 'Graph Built!'
		sess.run(tf.global_variables_initializer())
		model.train()