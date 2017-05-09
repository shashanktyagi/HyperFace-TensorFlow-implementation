import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from spatial_transformer import transformer
from tqdm import tqdm
from pdb import set_trace as brk

class Network(object):

	def __init__(self, sess):

		self.sess = sess
		self.batch_size = 2
		self.img_height = 500
		self.img_width = 500
		self.out_height = 227
		self.out_width = 227
		self.channel = 3

		self.num_epochs = 10

		# Hyperparameters
		self.weight_detect = 1
		self.weight_landmarks = 5
		self.weight_visibility = 0.5
		self.weight_pose = 5
		self.weight_gender = 2

		self.build_network()


	def build_network(self):

		self.X = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.channel], name='images')
		self.detection = tf.placeholder(tf.float32, [self.batch_size,2], name='detection')
		self.landmarks = tf.placeholder(tf.float32, [self.batch_size, 42], name='landmarks')
		self.visibility = tf.placeholder(tf.float32, [self.batch_size,21], name='visibility')
		self.pose = tf.placeholder(tf.float32, [self.batch_size,3], name='pose')
		self.gender = tf.placeholder(tf.float32, [self.batch_size,2], name='gender')


		theta = self.localization_squeezenet(self.X)

		self.T_mat = tf.reshape(theta, [-1, 2,3])

		self.cropped = transformer(self.X, self.T_mat, [self.out_height, self.out_width])

		net_output = self.hyperface(self.cropped) # (out_detection, out_landmarks, out_visibility, out_pose, out_gender)


		loss_detection = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(net_output[0], self.detection))
		
		visibility_mask = tf.reshape(tf.tile(tf.expand_dims(self.visibility, axis=2), [1,1,2]), [self.batch_size, -1])
		loss_landmarks = tf.reduce_mean(tf.square(visibility_mask*(net_output[1] - self.landmarks)))
		
		loss_visibility = tf.reduce_mean(tf.square(net_output[2] - self.visibility))
		loss_pose = tf.reduce_mean(tf.square(net_output[3] - self.pose))
		loss_gender = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(net_output[4], self.gender))

		self.loss = self.weight_detect*loss_detection + self.weight_landmarks*loss_landmarks  \
					+ self.weight_visibility*loss_visibility + self.weight_pose*loss_pose  \
					+ self.weight_gender*loss_gender



	def get_transformation_matrix(self, theta):
		with tf.name_scope('T_matrix'):
			theta = tf.expand_dims(theta, 2)
			mat = tf.constant(np.repeat(np.array([[[1,0,0],[0,0,0],[0,1,0],[0,0,0],[0,1,0],[0,0,1]]]),
										 self.batch_size, axis=0), dtype=tf.float32)
			tr_matrix = tf.squeeze(tf.matmul(mat, theta))

		return tr_matrix



	def train(self):
		
		optimizer = tf.train.AdamOptimizer().minimize(self.loss)

		writer = tf.summary.FileWriter('./logs', self.sess.graph)
		loss_summ = tf.summary.scalar('loss', self.loss)
		img_summ = tf.summary.image('cropped_image', self.cropped)

		print self.sess.run(self.T_mat, feed_dict={self.X: np.random.randn(self.batch_size, self.img_height, self.img_width, self.channel)})



	def hyperface(self,inputs, reuse = False):

		if reuse:
			tf.get_variable_scope().reuse_variables()

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
			# transposed for weight loading from chainer model
			fc_full = slim.fully_connected(tf.reshape(tf.transpose(conv_all, [0,3,1,2]), [-1, shape]), 3072, scope='fc_full')

			fc_detection = slim.fully_connected(fc_full, 512, scope='fc_detection1')
			fc_landmarks = slim.fully_connected(fc_full, 512, scope='fc_landmarks1')
			fc_visibility = slim.fully_connected(fc_full, 512, scope='fc_visibility1')
			fc_pose = slim.fully_connected(fc_full, 512, scope='fc_pose1')
			fc_gender = slim.fully_connected(fc_full, 512, scope='fc_gender1')

			out_detection = slim.fully_connected(fc_detection, 2, scope='fc_detection2', activation_fn = None)
			out_landmarks = slim.fully_connected(fc_landmarks, 42, scope='fc_landmarks2', activation_fn = None)
			out_visibility = slim.fully_connected(fc_visibility, 21, scope='fc_visibility2', activation_fn = None)
			out_pose = slim.fully_connected(fc_pose, 3, scope='fc_pose2', activation_fn = None)
			out_gender = slim.fully_connected(fc_gender, 2, scope='fc_gender2', activation_fn = None)

		return [tf.nn.softmax(out_detection), out_landmarks, out_visibility, out_pose, tf.nn.softmax(out_gender)]



	def localization_VGG16(self,inputs):

		with tf.variable_scope('localization_network'):
			with slim.arg_scope([slim.conv2d, slim.fully_connected],
								 activation_fn = tf.nn.relu,
								 weights_initializer = tf.constant_initializer(0.0)):
				
				net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
				net = slim.max_pool2d(net, [2, 2], scope='pool1')
				net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
				net = slim.max_pool2d(net, [2, 2], scope='pool2')
				net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
				net = slim.max_pool2d(net, [2, 2], scope='pool3')
				net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
				net = slim.max_pool2d(net, [2, 2], scope='pool4')
				net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
				net = slim.max_pool2d(net, [2, 2], scope='pool5')
				shape = int(np.prod(net.get_shape()[1:]))

				net = slim.fully_connected(tf.reshape(net, [-1, shape]), 4096, scope='fc6')
				net = slim.fully_connected(net, 1024, scope='fc7')
				identity = np.array([[1., 0., 0.],
									[0., 1., 0.]])
				identity = identity.flatten()
				net = slim.fully_connected(net, 6, biases_initializer = tf.constant_initializer(identity) , scope='fc8')
			
		return net


	def localization_squeezenet(self, inputs):

		with tf.variable_scope('localization_network'):	
			with slim.arg_scope([slim.conv2d], activation_fn = tf.nn.relu,
									padding = 'SAME',
									weights_initializer = tf.constant_initializer(0.0)):

				conv1 = slim.conv2d(inputs, 64, [3,3], 2, padding = 'VALID', scope='conv1')
				pool1 = slim.max_pool2d(conv1, [2,2], 2, scope='pool1')
				fire2 = self.fire_module(pool1, 16, 64, scope = 'fire2')
				fire3 = self.fire_module(fire2, 16, 64, scope = 'fire3', res_connection=True)
				fire4 = self.fire_module(fire3, 32, 128, scope = 'fire4')
				pool4 = slim.max_pool2d(fire4, [2,2], 2, scope='pool4')
				fire5 = self.fire_module(pool4, 32, 128, scope = 'fire5', res_connection=True)
				fire6 = self.fire_module(fire5, 48, 192, scope = 'fire6')
				fire7 = self.fire_module(fire6, 48, 192, scope = 'fire7', res_connection=True)
				fire8 = self.fire_module(fire7, 64, 256, scope = 'fire8')
				pool8 = slim.max_pool2d(fire8, [2,2], 2, scope='pool8')
				fire9 = self.fire_module(pool8, 64, 256, scope = 'fire9', res_connection=True)
				conv10 = slim.conv2d(fire9, 128, [1,1], 1, scope='conv10')
				shape = int(np.prod(conv10.get_shape()[1:]))
				fc11 = slim.fully_connected(tf.reshape(conv10, [-1, shape]), 6, biases_initializer = tf.constant_initializer(np.array([[1., 0., 0.],
																										  [0., 1., 0.]])) , scope='fc11')
		return fc11


	def fire_module(self, inputs, s_channels, e_channels, scope, res_connection = False):
		with tf.variable_scope(scope):
			sq = self.squeeze(inputs, s_channels, 'squeeze')
			ex = self.expand(sq, e_channels, 'expand')
			if res_connection:
				ret = tf.nn.relu(tf.add(inputs,ex))
			else:
				ret = tf.nn.relu(ex)
		return ret


	def squeeze(self, inputs, channels, scope):
		with slim.arg_scope([slim.conv2d], activation_fn = None,
							padding = 'SAME',
							weights_initializer = tf.truncated_normal_initializer(0.0, 0.01)):
			sq = slim.conv2d(inputs, channels, [1,1], 1, scope = scope)
		return sq

	def expand(self, inputs, channels, scope):
		with slim.arg_scope([slim.conv2d], activation_fn = None,
							padding = 'SAME',
							weights_initializer = tf.truncated_normal_initializer(0.0, 0.01)):
			with tf.variable_scope(scope):
				e1x1 = slim.conv2d(inputs, channels, [1,1], 1, scope='e1x1')
				e3x3 = slim.conv2d(inputs, channels, [3,3], 1, scope='e3x3')
				expand = tf.concat(3, [e1x1, e3x3])
		
		return expand



	def predict(self, imgs_path):
		print 'Running inference...'
		np.set_printoptions(suppress=True)
		imgs = (np.load(imgs_path) - 127.5)/128.0
		shape = imgs.shape
		self.X = tf.placeholder(tf.float32, [shape[0], self.img_height, self.img_width, self.channel], name='images')
		pred = self.network(self.X, reuse = True)

		net_preds = self.sess.run(pred, feed_dict={self.X: imgs})

		print net_preds[-1]
		import matplotlib.pyplot as plt
		plt.imshow(imgs[-1]);plt.show()

		brk()




	def load_weights(self, path):
		variables = slim.get_model_variables()
		print 'Loading weights...'
		for var in tqdm(variables):
			if ('conv' in var.name) and ('weights' in var.name):
				self.sess.run(var.assign(np.load(path+var.name.split('/')[0]+'/W.npy').transpose((2,3,1,0))))
			elif ('fc' in var.name) and ('weights' in var.name):
				self.sess.run(var.assign(np.load(path+var.name.split('/')[0]+'/W.npy').T))
			elif 'biases' in var.name:
				self.sess.run(var.assign(np.load(path+var.name.split('/')[0]+'/b.npy')))
		print 'Weights loaded!!'

	def print_variables(self):
		variables = slim.get_model_variables()
		print 'Model Variables:\n'
		for var in variables:
			print var.name, ' ', var.get_shape()


			

