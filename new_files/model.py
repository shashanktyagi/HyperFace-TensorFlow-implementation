import cv2
import irp
import lnms
import selective_search
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from ipdb import set_trace as brk

class HyperFace(object):

	def __init__(self, sess,batch_size=None,num_epochs=None,forward_only=None):

		self.sess = sess
		self.forward_only = forward_only
		
		if self.forward_only == 1:
			self.batch_size = None
		else:
			self.batch_size = batch_size

		self.img_height = 227
		self.img_width = 227
		self.channel = 3

		self.num_epochs = num_epochs

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
		
		self.net_output = self.network(self.X) # (out_detection, out_landmarks, out_visibility, out_pose, out_gender)
		if self.forward_only == 0:
			loss_detection = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.net_output[0], self.detection))
			
			visibility_mask = tf.reshape(tf.tile(tf.expand_dims(self.visibility, axis=2), [1,1,2]), [self.batch_size, -1])
			loss_landmarks = tf.reduce_mean(tf.square(visibility_mask*(self.net_output[1] - self.landmarks)))
			
			loss_visibility = tf.reduce_mean(tf.square(self.net_output[2] - self.visibility))
			loss_pose = tf.reduce_mean(tf.square(self.net_output[3] - self.pose))
			loss_gender = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.net_output[4], self.gender))

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

			concat_feat = tf.concat( [conv1a, conv3a, pool5],3)
			conv_all = slim.conv2d(concat_feat, 192, [1,1], 1, padding= 'VALID', scope='conv_all')
			
			shape = int(np.prod(conv_all.get_shape()[1:]))
			fc_full = slim.fully_connected(tf.reshape(tf.transpose(conv_all, [0,3,1,2]), [-1, shape]), 3072, scope='fc_full')
			#fc_full = slim.fully_connected(tf.reshape(conv_all, [-1, shape]), 3072, scope='fc_full')
			fc_detection = slim.fully_connected(fc_full, 512, scope='fc_detection')
			fc_landmarks = slim.fully_connected(fc_full, 512, scope='fc_landmarks')
			fc_visibility = slim.fully_connected(fc_full, 512, scope='fc_visibility')
			fc_pose = slim.fully_connected(fc_full, 512, scope='fc_pose')
			fc_gender = slim.fully_connected(fc_full, 512, scope='fc_gender')

			out_detection = slim.fully_connected(fc_detection, 2, scope='out_detection',activation_fn = None)
			out_landmarks = slim.fully_connected(fc_landmarks, 42, scope='out_landmarks',activation_fn = None)
			out_visibility = slim.fully_connected(fc_visibility, 21, scope='out_visibility',activation_fn = None)
			out_pose = slim.fully_connected(fc_pose, 3, scope='out_pose',activation_fn = None)
			out_gender = slim.fully_connected(fc_gender, 2, scope='out_gender',activation_fn = None)

		
		return [out_detection, out_landmarks, out_visibility, out_pose, tf.nn.softmax(out_gender),conv_all]

	def load_from_tfRecord(self,filename_queue):
		
		reader = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		
		features = tf.parse_single_example(
			serialized_example,
			features={
				'image_raw':tf.FixedLenFeature([], tf.string),
				'width': tf.FixedLenFeature([], tf.int64),
				'height': tf.FixedLenFeature([], tf.int64),
				'batch_size':tf.FixedLenFeature([], tf.int64)
			})
		
		image = tf.decode_raw(features['image_raw'], tf.float32)
		orig_height = tf.cast(features['height'], tf.int32)
		orig_width = tf.cast(features['width'], tf.int32)
		batch_size = tf.cast(features['batch_size'], tf.int32)

		image_shape = tf.pack([batch_size,227,227,3])
		image_tf = tf.reshape(image,image_shape)

		images = tf.train.shuffle_batch([image_tf],batch_size=self.batch_size,enqueue_many=True,num_threads=1,capacity=50,min_after_dequeue=10)
		
		return images

	def load_model(self,model_path):
		for var in tf.all_variables():
			if var.name.find('weights') != -1:
				if var.name.find('conv') != -1:
					self.sess.run(var.assign(np.load(model_path+'/'+var.name.split('/')[0]+'/W.npy').transpose(2,3,1,0)))
				else:
					self.sess.run(var.assign(np.load(model_path+'/'+var.name.split('/')[0]+'/W.npy').T))
			if var.name.find('biases') != -1:
				self.sess.run(var.assign(np.load(model_path+'/'+var.name.split('/')[0]+'/b.npy')))

		print "Done Loading"

	def test_hyperface(self,ip_img,nms_threshold=0.2,irp_count=2):
		# 1) Take the input as image
		# 2) Run DLIB's selective search on that
		# 3) Pass the regions to the trained model
		# 4) For all the regions having detection score greater than a threshold.
		# 	4.1) Perform Iterative Region Proposal on it.
		# 5) Use the new localized boxes to perform landmark based LMS
		# 6) Again run the network on the localized boxes from the IRP 
		# 7) Find precision boxes as the  min and max of the fids
		# 8) Run NMS
		# 9) Keep the top k boxes and use the median of each to give the final output
		# 10) Apply Face Rect Calculator on the final fids
		
		ip_img_size = ip_img.shape[0:-1] 
		total_boxes = None
		ip_img = ip_img.astype(np.float32)/255.0

		for i in range(1+irp_count):
			if i ==0:
				boxes_op,iou_dump,coords_dump = selective_search.perform_selective_search(ip_img.astype(np.float32),ground_truth=None)

			cropped_imgs = tf.image.crop_and_resize(ip_img[np.newaxis,:].astype(np.float32),boxes_op, [0]*boxes_op.shape[0], crop_size=[227,227]).eval(session=self.sess)
			# cropped_imgs = np.load('db_imgs.npy')
			# cropped_labels = np.load('db_labels.npy')
			# cropped_landmarks = np.load('db_landmarks.npy')
			brk()
			normalized_imgs = cropped_imgs - 0.5

			# a = np.load('/home/shashank/Documents/CSE-252C/chainer_ref/hyperface/gt_ip.npy')
			# a = a.transpose(0,2,3,1)
			
			input_feed={self.X:normalized_imgs}
			net_op = self.sess.run([self.net_output],feed_dict=input_feed)
			
			all_landmarks = np.asarray(net_op[0][1]).reshape(-1,42)
			all_landmarks_x = all_landmarks[:,::2].reshape(-1,21)
			all_landmarks_y = all_landmarks[:,1::2].reshape(-1,21)
			loc_w = (boxes_op[:,3] - boxes_op[:,1])*ip_img_size[1]
			loc_h = (boxes_op[:,2] - boxes_op[:,0])*ip_img_size[0]
			c_x = boxes_op[:,1]*ip_img_size[1] + loc_w/2.0
			c_y = boxes_op[:,0]*ip_img_size[0] + loc_h/2.0
			all_landmarks_x = all_landmarks_x*loc_w.reshape(-1,1) + c_x.reshape(-1,1)
			all_landmarks_y = all_landmarks_y*loc_h.reshape(-1,1) + c_y.reshape(-1,1)
			all_landmarks_x = all_landmarks_x[:,np.newaxis,:]
			all_landmarks_y = all_landmarks_y[:,np.newaxis,:]

			all_landmarks = np.concatenate([all_landmarks_x,all_landmarks_y],axis=1)

			detections = np.exp(np.asarray(net_op[0][0]).reshape(-1,2))
			detections = (detections/(np.sum(detections,axis=1).reshape(-1,1)))[:,1].reshape(-1,1)
			
			interests = np.where(detections>0.25)[0]

			visibility = np.asarray(net_op[0][2]).reshape(-1,21)[interests,:]
			visibility_mask = np.zeros_like(visibility)
			visibility_mask[np.where(visibility>0.5)] = 1

			landmarks = all_landmarks[interests,:,:].reshape(-1,2,21)
			
			detected_boxes =[]
			for i in range(len(interests)):
				mask = np.where(visibility_mask[i,:]==1)[0]
				y1,x1,y2,x2 = irp.region_proposal(landmarks[i,:,mask],mask,ip_img_size)
				if (y1 == y2) or (x1 == x2):
					continue
				detected_boxes.append([y1/float(ip_img_size[0]),x1/float(ip_img_size[1]),y2/float(ip_img_size[0]),x2/float(ip_img_size[1])])
			boxes_op = np.asarray(detected_boxes).astype(np.float32)
			

		#DO the final model run

		# cropped_imgs = tf.image.crop_and_resize(ip_img[np.newaxis,:].astype(np.float32),boxes_op, [0]*boxes_op.shape[0], crop_size=[227,227]).eval(session=self.sess)
		# normalized_imgs = (cropped_imgs - 127.5)/128.0
		# input_feed={self.X:normalized_imgs}
		# net_op = self.sess.run([self.net_output],feed_dict=input_feed)
		
		interests = np.where(detections>0.5)[0]
		landmarks = all_landmarks[interests,:,:].reshape(-1,2,21)
		visibility = np.asarray(net_op[0][2]).reshape(-1,21)[interests,:]
		poses = np.asarray(net_op[0][3]).reshape(-1,3)[interests,:]
		genders = np.asarray(net_op[0][4])[:,1].reshape(-1,1)[interests,:]

		visibility_mask = np.zeros_like(visibility)
		visibility_mask[np.where(visibility>0.5)] = 1
		
		# min_x = np.min(landmarks[:,0,visibility_mask],axis=1).reshape(-1,1)
		# min_y = np.min(landmarks[:,1,visibility_mask],axis=1).reshape(-1,1)
		# max_x = np.max(landmarks[:,0,visibility_mask],axis=1).reshape(-1,1)
		# max_y = np.max(landmarks[:,1,visibility_mask],axis=1).reshape(-1,1)
		precise_boxes = []
		for i in range(landmarks.shape[0]):
			min_x = np.min(landmarks[i,0,np.where(visibility_mask[i,:]==1)[0]])
			min_y = np.min(landmarks[i,1,np.where(visibility_mask[i,:]==1)[0]])
			max_x = np.max(landmarks[i,0,np.where(visibility_mask[i,:]==1)[0]])
			max_y = np.max(landmarks[i,1,np.where(visibility_mask[i,:]==1)[0]])
			precise_boxes.append([min_x,min_y,max_x,max_y])
		precise_boxes = np.asarray(precise_boxes)

		#precise_boxes = np.concatenate([min_x,min_y,max_x,max_y],axis=1)
		nms_op_dict = lnms.fast_nms(precise_boxes,nms_threshold)
		final_res = {'landmarks':[],'gender':[],'location':[],'pose':[]}
		for key in nms_op_dict:
			value = nms_op_dict[key]
			final_res['gender'].append(np.median(genders[value,:],axis=0))
			final_res['pose'].append(np.median(poses[value,:],axis=0))
			temp = np.median(landmarks[value,:,:],axis=0).T[np.where(np.median(visibility[value,:],axis=0)>0.5)[0],:]
			brk()
			final_res['landmarks'].append(temp)
			y1,x1,y2,x2 = irp.region_proposal(temp,np.where(np.median(visibility[value,:],axis=0)>0.5)[0],ip_img_size)
			final_res['location'].append([x1,y1,x2,y2])

		print "Done"
		return final_res
		



	def print_variables(self):
		variables = slim.get_model_variables()
		print 'Model Variables:'
		for var in variables:
			print var.name, ' ', var.get_shape()


			

