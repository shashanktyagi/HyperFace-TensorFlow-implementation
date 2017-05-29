import tensorflow as tf
import os
from model import *


weights_path = '/Users/shashank/Tensorflow/SPN/weights/'
imgs_path = '/Users/shashank/Tensorflow/CSE252C-Hyperface/git/truth_data.npy'

if not os.path.exists('./logs'):
	os.makedirs('./logs')

if not os.path.exists('../../checkpoint'):
	os.makedirs('../../checkpoint')

if not os.path.exists('../../best_checkpoint'):
	os.makedirs('../../best_checkpoint')

map(os.unlink, (os.path.join( './logs',f) for f in os.listdir('./logs')) )

net = HyperFace(False,tf_record_file_path='../../aflw_train_small_new.tfrecords',model_save_path='../../checkpoint',best_model_save_path='../../best_checkpoint')

with tf.Session() as sess:
		print 'Building Graph...'
		net.build_network(sess)
		print 'Graph Built!'
		# net.print_variables()
		# net.load_weights(weights_path)
		net.train()

