import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tqdm import tqdm
from pdb import set_trace as brk
import sys

class HyperFace(object):

	def __init__(self,load_model,tf_record_file_path=None,model_save_path=None,best_model_save_path=None,restore_model_path=None):

		self.batch_size = 32
		self.img_height = 227
		self.img_width = 227
		self.channel = 3

		self.num_epochs =10 

		# Hyperparameters  1,5,0.5,5,2
		self.weight_detect = 1      
		self.weight_landmarks = 5
		self.weight_visibility = 0.5
		self.weight_pose = 5
		self.weight_gender = 2

		#tf_Record Paramters
		self.tf_record_file_path = tf_record_file_path
		self.filename_queue = tf.train.string_input_producer([self.tf_record_file_path], num_epochs=self.num_epochs)
		self.images, self.labels, self.land, self.vis, self.po, self.gen= self.load_from_tfRecord(self.filename_queue)

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
		self.landmarks = tf.placeholder(tf.float32, [self.batch_size, 42], name='landmarks')
		self.visibility = tf.placeholder(tf.float32, [self.batch_size,21], name='visibility')
		self.pose = tf.placeholder(tf.float32, [self.batch_size,3], name='pose')
		self.gender = tf.placeholder(tf.int32, [self.batch_size], name='gender')
		
		net_output = self.network(self.X) # (out_detection, out_landmarks, out_visibility, out_pose, out_gender)
		self.test_model = net_output
		self.loss_detection = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=net_output[0], labels=tf.one_hot(self.detection, 2)))

		detection_mask = tf.cast(tf.expand_dims(self.detection, axis=1),tf.float32)
		
		visibility_mask = tf.reshape(tf.tile(tf.expand_dims(self.visibility, axis=2), [1,1,2]), [self.batch_size, -1])
		self.loss_landmarks = tf.reduce_mean(tf.square(detection_mask*visibility_mask*(net_output[1] - self.landmarks)))
		
		self.loss_visibility = tf.reduce_mean(tf.square(detection_mask*(net_output[2] - self.visibility)))
		self.loss_pose = tf.reduce_mean(tf.square(detection_mask*(net_output[3] - self.pose)))
		self.loss_gender = tf.reduce_mean(detection_mask*tf.nn.sigmoid_cross_entropy_with_logits(logits=net_output[4], labels=tf.one_hot(self.gender,2)))

		
		self.loss = self.weight_detect*self.loss_detection + self.weight_landmarks*self.loss_landmarks  \
		 			+ self.weight_visibility*self.loss_visibility + self.weight_pose*self.loss_pose  \
		 			+ self.weight_gender*self.loss_gender

		self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(net_output[0],1),tf.int32),self.detection),tf.float32))

		#self.loss = self.loss_detection
		#self.optimizer = tf.train.AdamOptimizer(1e-7).minimize(self.loss)
		self.optimizer = tf.train.MomentumOptimizer(1e-3,0.9,use_nesterov=True).minimize(self.loss)	
		self.saver = tf.train.Saver(max_to_keep=4, keep_checkpoint_every_n_hours=4)
		self.best_saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=4)


	def train(self):
		
		
		if self.load_model:
			print "Restoring Model"
			ckpt = tf.train.get_checkpoint_state(self.restore_model_path)
			if ckpt and ckpt.model_checkpoint_path:
				self.saver.restore(self.sess,ckpt.model_checkpoint_path)
				self.sess.run(tf.local_variables_initializer())
		else:
			print "Initializing Model"
			self.sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
			
		#self.load_det_weights(self.restore_model_path+'weights.npy')
			
		
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)

		writer = tf.summary.FileWriter('../logs', self.sess.graph)
		loss_summ = tf.summary.scalar('loss', self.loss)
		img_summ = tf.summary.image('images', self.images, max_outputs=5)
		label_summ = tf.summary.histogram('labels', self.detection)
		detect_summ = tf.summary.scalar('det_loss', self.loss_detection)
		landmarks_summ =  tf.summary.scalar('landmarks_loss', self.loss_landmarks)
		vis_summ = tf.summary.scalar('visibility_loss', self.loss_visibility)
		pose_summ = tf.summary.scalar('pose_loss', self.loss_pose)
		gender_summ = tf.summary.scalar('gender_loss', self.loss_gender)

		summ_op = tf.summary.merge_all()

		counter = 0
		best_loss = sys.maxint
		try:
			while not coord.should_stop():
				batch_imgs, batch_labels, batch_landmarks, batch_visibility, batch_pose, batch_gender = self.sess.run([self.images,self.labels,self.land, self.vis, self.po, self.gen])
				batch_imgs = (batch_imgs - 127.5) / 128.0
				input_feed={self.X: batch_imgs, self.detection: batch_labels, self.landmarks: batch_landmarks, self.visibility: batch_visibility, self.pose: batch_pose, self.gender: np.squeeze(batch_gender)}
				#input_feed={self.X: batch_imgs, self.detection: batch_labels}

				_,model_op,loss,l_d,l_l,l_v,l_p,l_g, summ, accuracy = self.sess.run([self.optimizer,self.test_model,self.loss,self.loss_detection,
self.loss_landmarks,self.loss_visibility,self.loss_pose,self.loss_gender, summ_op, self.accuracy], input_feed)
				
				writer.add_summary(summ, counter)

				if counter % self.save_after_steps == 0:
					self.saver.save(self.sess,self.model_save_path+'hyperface_model',global_step=int(counter),write_meta_graph=False)

				
				if loss <= best_loss:
					best_loss = loss
					self.best_saver.save(self.sess,self.best_model_save_path+'hyperface_best_model',global_step=int(counter),write_meta_graph=False)
					#self.save_weights(self.best_model_save_path)
					
				if counter % self.print_after_steps == 0:
					print "Iteration:{},Total Loss:{},Detection loss:{},Landmark loss:{},Visbility Loss :{},Pose Loss:{},Gender Loss:{},Accuracy:{}".format(counter,loss,l_d,l_l,l_v,l_p,l_g,accuracy)
					
				counter += 1

		except tf.errors.OutOfRangeError:
			print('Done training -- epoch limit reached')
		finally:
			coord.request_stop()

		coord.join(threads)	





	def network_det(self,inputs,reuse=False):

		if reuse:
			tf.get_variable_scope().reuse_variables()

		with slim.arg_scope([slim.conv2d, slim.fully_connected],
							 activation_fn = tf.nn.relu,
							 weights_initializer = tf.truncated_normal_initializer(0.0, 0.01)):
			
			conv1 = slim.conv2d(inputs, 96, [11,11], 4, padding= 'VALID', scope='conv1')
			max1 = slim.max_pool2d(conv1, [3,3], 2, padding= 'VALID', scope='max1')

			conv2 = slim.conv2d(max1, 256, [5,5], 1, scope='conv2')
			max2 = slim.max_pool2d(conv2, [3,3], 2, padding= 'VALID', scope='max2')
			conv3 = slim.conv2d(max2, 384, [3,3], 1, scope='conv3')

			conv4 = slim.conv2d(conv3, 384, [3,3], 1, scope='conv4')
			conv5 = slim.conv2d(conv4, 256, [3,3], 1, scope='conv5')
			pool5 = slim.max_pool2d(conv5, [3,3], 2, padding= 'VALID', scope='pool5')
			
			shape = int(np.prod(pool5.get_shape()[1:]))
			fc6 = slim.fully_connected(tf.reshape(pool5, [-1, shape]), 4096, scope='fc6')
			
			fc_detection = slim.fully_connected(fc6, 512, scope='fc_det1')
			out_detection = slim.fully_connected(fc_detection, 2, scope='fc_det2', activation_fn = None)
			
		return out_detection







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
			out_landmarks = slim.fully_connected(fc_landmarks, 42, scope='fc_landmarks2', activation_fn = None )
			out_visibility = slim.fully_connected(fc_visibility, 21, scope='fc_visibility2', activation_fn = None)
			out_pose = slim.fully_connected(fc_pose, 3, scope='fc_pose2', activation_fn = None)
			out_gender = slim.fully_connected(fc_gender, 2, scope='fc_gender2', activation_fn = None)

		return [out_detection, out_landmarks, out_visibility, out_pose, out_gender]



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
				'gender':tf.FixedLenFeature([], tf.int64),
				'pose': tf.FixedLenFeature([], tf.string),
				'landmarks':tf.FixedLenFeature([], tf.string),
				'visibility':tf.FixedLenFeature([], tf.string),

			})
		
		landmarks = tf.decode_raw(features['landmarks'], tf.float32)
		pose = tf.decode_raw(features['pose'], tf.float32)
		visibility = tf.decode_raw(features['visibility'], tf.int32)
		gender = tf.cast(features['gender'], tf.int32)

		landmarks_shape = tf.stack([1,21*2])
		pose_shape = tf.stack([1,3])
		visibility_shape = tf.stack([1,21])
		gender_shape = tf.stack([1,1])

		landmarks = tf.reshape(landmarks,landmarks_shape)
		visibility = tf.reshape(visibility,visibility_shape)
		pose = tf.reshape(pose,pose_shape)
		gender = tf.reshape(gender,gender_shape)

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


		positive_landmarks = tf.tile(landmarks,[n_pos_locs,1])
		negative_landmarks = tf.tile(landmarks,[n_neg_locs,1])

		positive_visibility = tf.tile(visibility,[n_pos_locs,1])
		negative_visibility = tf.tile(visibility,[n_neg_locs,1])

		positive_pose = tf.tile(pose,[n_pos_locs,1])
		negative_pose = tf.tile(pose,[n_neg_locs,1])

		positive_gender = tf.tile(gender,[n_pos_locs,1])
		negative_gender = tf.tile(gender,[n_neg_locs,1])
		
		all_landmarks = tf.concat([positive_landmarks,negative_landmarks],axis=0)
		all_visibility = tf.concat([positive_visibility,negative_visibility],axis=0)
		all_pose = tf.concat([positive_pose,negative_pose],axis=0)

		all_labels = tf.concat([positive_labels,negative_labels],axis=0)
		all_gender = tf.concat([positive_gender,negative_gender],axis=0)

		tf.random_shuffle(all_images,seed=7)
		tf.random_shuffle(all_labels,seed=7)
		tf.random_shuffle(all_landmarks,seed=7)
		tf.random_shuffle(all_visibility,seed=7)
		tf.random_shuffle(all_pose,seed=7)
		tf.random_shuffle(all_gender,seed=7)

		images,labels,landmarks_,visibility_,pose_,gender_ = tf.train.shuffle_batch([all_images,all_labels,all_landmarks,all_visibility,all_pose,all_gender]
			,enqueue_many=True,batch_size=self.batch_size,num_threads=1,capacity=1000,min_after_dequeue=500)
		
		return images,labels,landmarks_,visibility_,pose_,gender_

	
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


	def save_weights(self, path):
		variables = slim.get_model_variables()
		weights = {}
		for var in variables:
			weights[var.name] = self.sess.run(var)

		np.save(path+ '/weights', weights)

	def load_det_weights(self, path):
		variables = slim.get_model_variables()
		weights = np.load(path)
		for var in variables:
			if var.name in weights.item():
				print var.name
				self.sess.run(var.assign(weights.item()[var.name]))







			

