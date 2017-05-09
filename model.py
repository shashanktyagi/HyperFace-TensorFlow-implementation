import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class HyperFace(object):

	def __init__(self, sess,tf_record_file_path=None):

		self.sess = sess
		self.batch_size = 2
		self.img_height = 227
		self.img_width = 227
		self.channel = 3

		self.num_epochs = 10

		# Hyperparameters
		self.weight_detect = 1
		self.weight_landmarks = 5
		self.weight_visibility = 0.5
		self.weight_pose = 5
		self.weight_gender = 2

		#tf_Record Paramters
		self.filename_queue = tf.train.string_input_producer([tf_record_file_path], num_epochs=self.num_epochs)

		#Spatial Transformer Input
		self.sp_input_width = 500
		self.sp_input_height = 500

		self.build_network()


	def build_network(self):

		self.X = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.channel], name='images')
		self.detection = tf.placeholder(tf.float32, [self.batch_size,2], name='detection')
		self.landmarks = tf.placeholder(tf.float32, [self.batch_size, 42], name='landmarks')
		self.visibility = tf.placeholder(tf.float32, [self.batch_size,21], name='visibility')
		self.pose = tf.placeholder(tf.float32, [self.batch_size,3], name='pose')
		self.gender = tf.placeholder(tf.float32, [self.batch_size,2], name='gender')
		
		self.X = self.load_from_tfRecord(self.filename_queue)
		
		net_output = self.network(self.X) # (out_detection, out_landmarks, out_visibility, out_pose, out_gender)

		loss_detection = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(net_output[0], self.detection))
		
		visibility_mask = tf.reshape(tf.tile(tf.expand_dims(self.visibility, axis=2), [1,1,2]), [self.batch_size, -1])
		loss_landmarks = tf.reduce_mean(tf.square(visibility_mask*(net_output[1] - self.landmarks)))
		
		loss_visibility = tf.reduce_mean(tf.square(net_output[2] - self.visibility))
		loss_pose = tf.reduce_mean(tf.square(net_output[3] - self.pose))
		loss_gender = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(net_output[4], self.gender))

		self.loss = self.weight_detect*loss_detection + self.weight_landmarks*loss_landmarks  \
					+ self.weight_visibility*loss_visibility + self.weight_pose*loss_pose  \
					+ self.weight_gender*loss_gender


	def train(self):
		
		optimizer = tf.train.AdamOptimizer().minimize(self.loss)

		writer = tf.summary.FileWriter('./logs', self.sess.graph)
		loss_summ = tf.summary.scalar('loss', self.loss)

	def network(self,inputs):

		with slim.arg_scope([slim.conv2d, slim.fully_connected],
							 activation_fn = tf.nn.relu,
							 weights_initializer = tf.truncated_normal_initializer(0.0, 0.01) ):
			
			conv1 = slim.conv2d(inputs, 96, [11,11], 4, padding= 'VALID', scope='conv1')
			max1 = slim.max_pool2d(conv1, [3,3], 2, padding= 'VALID', scope='max1')

			conv1a = slim.conv2d(max1, 256, [4,4], 4, padding= 'VALID', scope='conv1a')

			conv2 = slim.conv2d(max1, 256, [5,5], 1, scope='conv2')
			max2 = slim.max_pool2d(conv2, [3,3], 2, padding= 'VALID', scope='max2')
			conv3 = slim.conv2d(max2, 384, [3,3], 1, scope='conv3')

			conv3a = slim.conv2d(conv3, 256, [2,2], 2, padding= 'VALID', scope='conv3a')

			conv4 = slim.conv2d(conv3, 384, [3,3], 1, scope='conv4')
			conv5 = slim.conv2d(conv4, 256, [3,3], 1, scope='conv5')
			pool5 = slim.max_pool2d(conv5, [3,3], 2, padding= 'VALID', scope='pool5')

			concat_feat = tf.concat(3, [conv1a, conv3a, pool5])
			conv_all = slim.conv2d(concat_feat, 192, [1,1], 1, padding= 'VALID', scope='conv_all')
			
			shape = int(np.prod(conv_all.get_shape()[1:]))
			fc_full = slim.fully_connected(tf.reshape(conv_all, [-1, shape]), 3072, scope='fc_full')

			fc_detection = slim.fully_connected(fc_full, 512, scope='fc_detection')
			fc_landmarks = slim.fully_connected(fc_full, 512, scope='fc_landmarks')
			fc_visibility = slim.fully_connected(fc_full, 512, scope='fc_visibility')
			fc_pose = slim.fully_connected(fc_full, 512, scope='fc_pose')
			fc_gender = slim.fully_connected(fc_full, 512, scope='fc_gender')

			out_detection = slim.fully_connected(fc_detection, 2, scope='out_detection')
			out_landmarks = slim.fully_connected(fc_landmarks, 42, scope='out_landmarks')
			out_visibility = slim.fully_connected(fc_visibility, 21, scope='out_visibility')
			out_pose = slim.fully_connected(fc_pose, 3, scope='out_pose')
			out_gender = slim.fully_connected(fc_gender, 2, scope='out_gender')

		return [out_detection, out_landmarks, out_visibility, out_pose, out_gender]

	def load_from_tfRecord(self,filename_queue):
		
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		
		features = tf.parse_single_example(
			serialized_example,
			features={
				'image_raw':tf.FixedLenFeature([], tf.string),
				'width': tf.FixedLenFeature([], tf.int64),
				'height': tf.FixedLenFeature([], tf.int64)
			})
		
		image = tf.decode_raw(features['image_raw'], tf.float32)
		orig_height = tf.cast(features['height'], tf.int32)
		orig_width = tf.cast(features['width'], tf.int32)
		
		image_shape = tf.pack([orig_height,orig_width,3])
		image_tf = tf.reshape(image,image_shape)

		resized_image = tf.image.resize_image_with_crop_or_pad(image_tf,target_height=self.img_height,target_width=self.img_width)
		
		images = tf.train.shuffle_batch([resized_image],batch_size=self.batch_size,num_threads=1,capacity=50,min_after_dequeue=10)
		
		return images

	def print_variables(self):
		variables = slim.get_model_variables()
		print 'Model Variables:'
		for var in variables:
			print var.name, ' ', var.get_shape()


			

