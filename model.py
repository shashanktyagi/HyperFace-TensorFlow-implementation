import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tqdm import tqdm
from pdb import set_trace as brk


class HyperFace(object):

	def __init__(self,load_model,tf_record_file_path=None,model_save_path=None,best_model_save_path=None,restore_model_path=None):

		self.batch_size = 2
		self.img_height = 227
		self.img_width = 227
		self.channel = 3

		self.num_epochs = 10

		# Hyperparameters  1,5,0.5,5,2
		self.weight_detect = 1      
		self.weight_landmarks = 0
		self.weight_visibility = 0
		self.weight_pose = 0
		self.weight_gender = 0

		#tf_Record Paramters
		self.tf_record_file_path = tf_record_file_path
		self.filename_queue = tf.train.string_input_producer([self.tf_record_file_path], num_epochs=self.num_epochs)
		self.images, self.labels = self.load_from_tfRecord(self.filename_queue)

		self.model_save_path = model_save_path
		self.best_model_save_path = best_model_save_path
		self.restore_model_path = restore_model_path

		self.save_after_steps = 200
		self.print_after_steps = 50
		self.load_model =  load_model

	def build_network(self, sess):

		self.sess = sess

		self.X = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.channel], name='images')
		self.detection = tf.placeholder(tf.int32, [self.batch_size], name='detection')
		# self.landmarks = tf.placeholder(tf.float32, [self.batch_size, 42], name='landmarks')
		# self.visibility = tf.placeholder(tf.float32, [self.batch_size,21], name='visibility')
		# self.pose = tf.placeholder(tf.float32, [self.batch_size,3], name='pose')
		# self.gender = tf.placeholder(tf.float32, [self.batch_size,2], name='gender')
		
		net_output = self.network(self.X) # (out_detection, out_landmarks, out_visibility, out_pose, out_gender)

		loss_detection = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_output, labels=tf.one_hot(self.detection, 2)))
		
		# visibility_mask = tf.reshape(tf.tile(tf.expand_dims(self.visibility, axis=2), [1,1,2]), [self.batch_size, -1])
		# loss_landmarks = tf.reduce_mean(tf.square(visibility_mask*(net_output[1] - self.landmarks)))
		
		# loss_visibility = tf.reduce_mean(tf.square(net_output[2] - self.visibility))
		# loss_pose = tf.reduce_mean(tf.square(net_output[3] - self.pose))
		# loss_gender = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(net_output[4], self.gender))

		# self.loss = self.weight_detect*loss_detection + self.weight_landmarks*loss_landmarks  \
		# 			+ self.weight_visibility*loss_visibility + self.weight_pose*loss_pose  \
		# 			+ self.weight_gender*loss_gender

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(net_output,1),tf.int32),self.detection),tf.float32))

		self.loss = loss_detection
		self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

	def train(self):
		
		
		if self.load_model:
			print "Restoring Model"
			ckpt = tf.train.get_checkpoint_state(self.model_save_path)
			if ckpt and ckpt.model_checkpoint_path:
				self.saver.restore(self.sess,ckpt.model_checkpoint_path)
		else:
			print "Initializing Model"
			self.sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))

		
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)

		writer = tf.summary.FileWriter('./logs', self.sess.graph)
		loss_summ = tf.summary.scalar('loss', self.loss)
		img_summ = tf.summary.image('images', self.images, max_outputs=5)
		label_summ = tf.summary.histogram('labels', self.detection)

		summ_op = tf.summary.merge_all()

		counter = 0
		best_loss = sys.maxint
		try:
			while not coord.should_stop():
				batch_imgs,batch_labels = self.sess.run([self.images,self.labels])
				batch_imgs = (batch_imgs - 127.5) / 128.0
				
				input_feed={self.X: batch_imgs, self.Y: batch_labels}
				
				_,loss, summ, accuracy = self.sess.run([optimizer, self.loss, summ_op, self.accuracy], self,input_feed)
				
				writer.add_summary(summ, counter)

				if (counter%self.save_after_steps == 0):
					self.saver.save(self.sess,self.model_save_path+'statefarm_model',global_step=int(counter),write_meta_graph=False)
				
				if batch_loss <= best_loss:
					best_loss = batch_loss
					self.best_saver.save(self.sess,self.best_model_save_path+'statefarm_best_model',global_step=int(counter),write_meta_graph=False)
					
				if counter%self.print_after_steps == 0:
					print "Iteration:{},Loss:{},Accuracy:{}".format(counter,batch_loss,accuracy)
				counter += 1

		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')
		finally:
			coord.request_stop()

		coord.join(threads)	





	def network(self,inputs,reuse=False):

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

			concat_feat = tf.concat([conv1a, conv3a, pool5],3)
			conv_all = slim.conv2d(concat_feat, 192, [1,1], 1, padding= 'VALID', scope='conv_all')
			
			shape = int(np.prod(conv_all.get_shape()[1:]))
			fc_full = slim.fully_connected(tf.reshape(conv_all, [-1, shape]), 3072, scope='fc_full')

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

		return out_detection #[tf.nn.softmax(out_detection), out_landmarks, out_visibility, out_pose, tf.nn.softmax(out_gender)]



	def predict(self, imgs_path):
		print 'Running inference...'
		np.set_printoptions(suppress=True)
		imgs = (np.load(imgs_path) - 127.5)/128.0
		shape = imgs.shape
		self.X = tf.placeholder(tf.float32, [shape[0], self.img_height, self.img_width, self.channel], name='images')
		pred = self.network(self.X, reuse = True)

		net_preds = self.sess.run(pred, feed_dict={self.X: imgs})

		print 'gender: \n', net_preds[-1]
		import matplotlib.pyplot as plt
		plt.imshow(imgs[-1]);plt.show()

		brk()

	def load_from_tfRecord(self,filename_queue):
		
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		
		features = tf.parse_single_example(
			serialized_example,
			features={
				'image_raw':tf.FixedLenFeature([], tf.string),
				'width': tf.FixedLenFeature([], tf.int64),
				'height': tf.FixedLenFeature([], tf.int64),
				'pos_locs':tf.FixedLenFeature([], tf.string),
				'neg_locs':tf.FixedLenFeature([], tf.string),
				'n_pos_locs':tf.FixedLenFeature([], tf.int64),
				'n_neg_locs':tf.FixedLenFeature([], tf.int64),

			})
		
		image = tf.decode_raw(features['image_raw'], tf.uint8)
		pos_locs = tf.decode_raw(features['pos_locs'], tf.float32)
		neg_locs = tf.decode_raw(features['neg_locs'], tf.float32)


		orig_height = tf.cast(features['height'], tf.int32)
		orig_width = tf.cast(features['width'], tf.int32)
		n_pos_locs = tf.cast(features['n_pos_locs'], tf.int32)
		n_neg_locs = tf.cast(features['n_neg_locs'], tf.int32)

		image_shape = tf.stack([1,orig_height,orig_width,3])
		image = tf.cast(tf.reshape(image,image_shape),tf.float32)

		pos_locs_shape = tf.stack([n_pos_locs,4])
		pos_locs = tf.reshape(pos_locs,pos_locs_shape)

		neg_locs_shape = tf.stack([n_neg_locs,4])
		neg_locs = tf.reshape(neg_locs,neg_locs_shape)

		positive_cropped = tf.image.crop_and_resize(image,pos_locs,tf.zeros([n_pos_locs],dtype=tf.int32),[227,227])
		negative_cropped = tf.image.crop_and_resize(image,neg_locs,tf.zeros([n_neg_locs],dtype=tf.int32),[227,227])

		all_images = tf.concat([positive_cropped,negative_cropped],axis=0)

		positive_labels = tf.ones([n_pos_locs])
		negative_labels = tf.zeros([n_neg_locs])

		all_labels = tf.concat([positive_labels,negative_labels],axis=0)

		tf.random_shuffle(all_images,seed=7)
		tf.random_shuffle(all_labels,seed=7)
		
		images,labels = tf.train.shuffle_batch([all_images,all_labels],enqueue_many=True,batch_size=self.batch_size,num_threads=1,capacity=1000,min_after_dequeue=500)
		
		return images,labels

	
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
		print 'Model Variables:'
		for var in variables:
			print var.name, ' ', var.get_shape()


			

