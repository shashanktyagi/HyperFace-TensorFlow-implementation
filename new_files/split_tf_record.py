import tensorflow as tf
import numpy as np
import dlib
import cv2

tf_record_file = 'aflw_train.tfrecords'

def calc_2D_IOU(bb1,bb2):
	top_left_x1 = bb1[0]
	top_left_y1 = bb1[1]
	bottom_right_x1 = bb1[2]
	bottom_right_y1 = bb1[3]

	top_left_x2 = bb2[0]
	top_left_y2 = bb2[1]
	bottom_right_x2 = bb2[2]
	bottom_right_y2 = bb2[3]

	intersect_top_left_x = max(bb1[0],bb2[0])
	intersect_top_left_y = max(bb1[1],bb2[1])
	intersect_bottom_right_x = max(min(bb1[2],bb2[2]),intersect_top_left_x)
	intersect_bottom_right_y = max(min(bb1[3],bb2[3]),intersect_top_left_y)

	intersect_area = (intersect_bottom_right_x-intersect_top_left_x+1)*(intersect_bottom_right_y-intersect_top_left_y+1)
	total_area = (bottom_right_x1-top_left_x1+1)*(bottom_right_y1-top_left_y1+1) + (bottom_right_x2-top_left_x2+1)*(bottom_right_y2-top_left_y2+1) - intersect_area
	iou = float(intersect_area)/float(total_area+0.0)
	return iou

def perform_scale_down(image,max_size_allowed):

	orig_h = image.shape[0]
	orig_w = image.shape[1]

	new_h = orig_h
	new_w = orig_w

	if new_h > max_size_allowed(1):
		new_w = (new_w*max_size_allowed(1)) / (new_h+0.0)
		new_h = max_size_allowed(1)

	if new_w > max_size_allowed(0):
		new_h = (new_h*max_size_allowed(0)) / (new_w+0.0)
		new_w = max_size_allowed(0)

	if new_h != orig_h or new_w != orig_w:
		return cv2.resize(image, (int(new_w), int(new_h)))
	else:
		return image


def perform_selective_search(img,w,h,ground_truth):
	rects=[]
	max_size=(500,500)
	img  = perform_scale_down(img,max_size)
	
	dlib.find_candidate_object_locations(img, rects, kvals=(50, 200, 2), min_size=2200)
	filter_positive_rects=[]
	filter_negative_rects=[]
	
	for rect in rects:
		iou = calc_2D_IOU(ground_truth,(rect.left(),rect.top(),rect.right(),rect.bottom()))
		
		if DEBUG_FLAG:
			debug_fp_csv.writerow([iou,ground_truth[0],ground_truth[1],ground_truth[2],ground_truth[3],rect.left(),rect.top(),rect.right(),rect.bottom()])
		if  iou > 0.5:
			filter_positive_rects.append([rect.top()/h,rect.left()/w,rect.bottom()/h,rect.right()/w])
		elif iou < 0.35:
			filter_negative_rects.append([rect.top()/h,rect.left()/w,rect.bottom()/h,rect.right()/w])

	return np.asarray(filter_positive_rects),np.asarray(filter_negative_rects)

def split_(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	
	features = tf.parse_single_example(
		serialized_example,
		features={
			'image_raw':tf.FixedLenFeature([], tf.string),
			'width': tf.FixedLenFeature([], tf.int64),
			'height': tf.FixedLenFeature([], tf.int64),
			'batch_size':tf.FixedLenFeature([], tf.int64)
			# 'roll':tf.FixedLenFeature([], tf.float32),
			# 'pitch':tf.FixedLenFeature([], tf.float32),
			# 'yaw':tf.FixedLenFeature([], tf.float32),
			# 'gender':tf.FixedLenFeature([], tf.int64),
			# 'roll':tf.FixedLenFeature([], tf.float32),
			# 'roll':tf.FixedLenFeature([], tf.float32),
			# 'landmarks':tf.FixedLenFeature([], tf.string),
			# 'locations':tf.FixedLenFeature([], tf.string)
		})
	
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	# locations = tf.decode_raw(features['locations'], tf.float32)
	# landmarks = tf.decode_raw(features['landmarks'], tf.float32)
	
	batch_size = tf.cast(features['batch_size'], tf.int32)
	orig_height = tf.cast(features['height'], tf.int32)
	orig_width = tf.cast(features['width'], tf.int32)
	
	image_shape = tf.pack([batch_size,227,227,3])

	image_tf = tf.reshape(image,image_shape)
	
	#resized_image = tf.image.resize_image_with_crop_or_pad(image_tf,target_height=500,target_width=500)

	# image_shape = tf.pack([height, width, 3])
	# image = tf.reshape(image, image_shape)
	# boxes,box_ind = perform_selective_search(,tf.cast(width,tf.float32),tf.cast(height,tf.float32),(loc_x,loc_y,loc_x+loc_w,loc_y+loc_h))

	# resized_and_cropped_image = tf.image.crop_and_resize(image, boxes, box_ind, crop_size=[227,227])
	
	images = tf.train.shuffle_batch([image_tf],enqueue_many=True,batch_size=32,num_threads=1,capacity=50000,min_after_dequeue=10000)
	
	return images
def split_spn(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	
	features = tf.parse_single_example(
		serialized_example,
		features={
			'image_raw':tf.FixedLenFeature([], tf.string),
			'width': tf.FixedLenFeature([], tf.int64),
			'height': tf.FixedLenFeature([], tf.int64),
			'loc_x': tf.FixedLenFeature([], tf.int64),
			'loc_y': tf.FixedLenFeature([], tf.int64),
			'loc_w': tf.FixedLenFeature([], tf.int64),
			'loc_h': tf.FixedLenFeature([], tf.int64)
		})
	
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	
	height = tf.cast(features['height'], tf.int32)
	width = tf.cast(features['width'], tf.int32)
	loc_x = tf.cast(features['loc_x'], tf.float32)
	loc_y = tf.cast(features['loc_y'], tf.float32)
	loc_w = tf.cast(features['loc_w'], tf.float32)
	loc_h = tf.cast(features['loc_h'], tf.float32)

	image_shape = tf.pack([height, width, 3])
	image_1 = tf.reshape(image, image_shape)
	image_shape = tf.pack([1,height, width, 3])
	image_2 = tf.cast(tf.reshape(image, image_shape),tf.float32)
	height = tf.cast(features['height'], tf.float32)
	width = tf.cast(features['width'], tf.float32)
	crop_index = tf.pack([[tf.divide(loc_y,height),tf.divide(loc_x,width),tf.divide(loc_y+loc_h,height),tf.divide(loc_w+loc_x,width)]])
	#boxes,box_ind = perform_selective_search(,tf.cast(width,tf.float32),tf.cast(height,tf.float32),(loc_x,loc_y,loc_x+loc_w,loc_y+loc_h))
	
	resized_image = tf.image.resize_image_with_crop_or_pad(image=image_1,target_height=500,target_width=500)
	resized_and_cropped_image = tf.image.crop_and_resize(image_2,crop_index,[0]*1,crop_size=[227,227])
	orig_images,cropped_images = tf.train.shuffle_batch([resized_image,resized_and_cropped_image],batch_size=10,num_threads=1,capacity=50,min_after_dequeue=10)
	
	return orig_images,cropped_images
filename_queue = tf.train.string_input_producer([tf_record_file], num_epochs=1)

ip1,ip2 = split_spn(filename_queue)

init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

print "Model Done"
with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	op = sess.run([ip1,ip2])
	
	output = np.asarray(op[0])
	#print output.shape 
	for i in range(output.shape[0]):
		cv2.imshow('result',output[i,:,:,:]/255.0)
		cv2.waitKey(0)
		break
	output = np.asarray(op[1]) 
	#print output.shape
	for i in range(output.shape[0]):
		cv2.imshow('result',output[i,0,:,:,:]/255.0)
		cv2.waitKey(0)
		break
	coord.request_stop()
	coord.join(threads)