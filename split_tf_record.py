import tensorflow as tf
import numpy as np
import dlib
from pdb import set_trace as brk

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
	intersect_bottom_right_x = min(bb1[2],bb2[2])
	intersect_bottom_right_y = min(bb1[3],bb2[3])

	intersect_area = (intersect_bottom_right_x-intersect_top_left_x+1)*(intersect_bottom_right_y-intersect_top_left_y+1)
	total_area = (bottom_right_x1-top_left_x1+1)*(bottom_right_y1-top_left_y1+1) + (bottom_right_x2-top_left_x2+1)*(bottom_right_y2-top_left_y2+1) - intersect_area
	iou = float(intersect_area)/float(total_area+0.0)
	return iou


def perform_selective_search(img,w,h,ground_truth):
	print "Came:"
	rects=[]
	dlib.find_candidate_object_locations(img, rects, min_size=500)
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

def split_(filename_queue, sess):
	brk()
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
	loc_x = tf.cast(features['loc_x'], tf.int32)
	loc_y = tf.cast(features['loc_y'], tf.int32)
	loc_w = tf.cast(features['loc_w'], tf.int32)
	loc_h = tf.cast(features['loc_h'], tf.int32)

	image_shape = tf.pack([height, width, 3])
	image = tf.reshape(image, image_shape)
	height,width,loc_x,loc_y,loc_h,loc_w = sess.run([height,width,loc_x,loc_y,loc_h,loc_w])
	# boxes,box_ind = perform_selective_search(,tf.cast(width,tf.float32),tf.cast(height,tf.float32),(loc_x,loc_y,loc_x+loc_w,loc_y+loc_h))
	boxes = np.asarray([[loc_y/float(height),loc_x/float(width),(loc_y+loc_h)/float(height),(loc_x+loc_w)/float(width)]])
	resized_and_cropped_image = tf.image.crop_and_resize(image.astype(np.float32), boxes.astype(np.float32), [0]*1, crop_size=[227,227])
	

	images = tf.train.shuffle_batch([resized_and_cropped_image],batch_size=10,num_threads=1,capacity=50,min_after_dequeue=10)
	
	return images
	
filename_queue = tf.train.string_input_producer([tf_record_file], num_epochs=1)



init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

print "model done"

with tf.Session() as sess:
	
	sess.run(init_op)
	images = split_(filename_queue, sess)
	
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	op_images = sess.run([images])
	print np.asarray(op_images).shape
	
	coord.request_stop()
	coord.join(threads)