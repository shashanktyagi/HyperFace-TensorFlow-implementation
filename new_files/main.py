import vis
import cv2
import tensorflow as tf
import os
import argparse
from skimage import io
from model import *
# import pdb
# pdb.set_trace()

if not os.path.exists('./logs'):
	os.makedirs('./logs')

map(os.unlink, (os.path.join( './logs',f) for f in os.listdir('./logs')) )

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-f','--forward_only',dest='forward_only',help='Test/Train Mode Flag',default=0,type=int)
	parser.add_argument('-b','--batchsize',dest='batch_size',help='Batch Size to calculate the number of iterations per epoch',default=32,type=int)
	parser.add_argument('-e','--n_epochs',dest='num_epochs',help='Number of Epochs for Training',default=10,type=int)
	parser.add_argument('-p','--model_path',dest='model_path',help='Enter the path for the model to use for testing',default=None,type=str)
	parser.add_argument('-t','--tf_record_path',dest='tf_record_file_path',help='Enter the path for the Tf Record File to use for training',default=None,type=str)	
	parser.add_argument('-i','--test_image_path',dest='test_image_path',help='Enter the test image path',default=None,type=str)	
	args = parser.parse_args()
	return args

with tf.Session() as sess:
	print "Parsing Argument"
	args = parse_args()
	print 'Building Graph...'
	net = HyperFace(sess,batch_size=args.batch_size,num_epochs=args.num_epochs,forward_only=args.forward_only)
	print 'Graph Built!'
	sess.run(tf.global_variables_initializer())
	if args.forward_only == 1:
		print "Loading Model"
		net.load_model(args.model_path)
		print "Start Testing"
		#img_raw = np.asarray()
		img_raw = np.asarray(cv2.imread(args.test_image_path))
		print img_raw.shape
		output_set = net.test_hyperface(img_raw)
		vis.vis_results(img_raw,output_set)
	else:
		filename_queue = tf.train.string_input_producer([args.tf_record_file_path], num_epochs=args.num_epochs)
		#net.train()
		print "Start Training"
	
		# net.train()

