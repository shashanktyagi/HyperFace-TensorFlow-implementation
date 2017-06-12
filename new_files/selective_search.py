#!/home/shashank/anaconda2/bin
import dlib
from skimage import io
import cv2
import numpy as np
import tensorflow as tf
import csv
# from multiprocessing import Pool
# from multiprocessing import Manager
# from multiprocessing import Queue
# from multiprocessing.dummy import Pool as ThreadPool
import os
import math
import time
from tqdm import tqdm
from pdb import set_trace as brk
# import pdb
# pdb.set_trace()

DEBUG_FLAG = False
VIS_FLAG = False
MAKE_TF_RECORD = False
tfrecords_full_filename = 'aflw_test.tfrecords'

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_full_filename)
tfrecords_training_pos_filename = 'aflw_training_pos.tfrecords'
tfrecords_training_neg_filename = 'aflw_training_neg.tfrecords'
writer_pos = tf.python_io.TFRecordWriter(tfrecords_training_pos_filename)
writer_neg = tf.python_io.TFRecordWriter(tfrecords_training_neg_filename)

N_TRAIN = 20000
N_TEST = 1000

if DEBUG_FLAG:
	debug_fp = open('debug.csv','wb')
	debug_fp_csv = csv.writer(debug_fp)

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))




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

	intersect_area = (intersect_bottom_right_x-intersect_top_left_x)*(intersect_bottom_right_y-intersect_top_left_y)
	total_area = (bottom_right_x1-top_left_x1)*(bottom_right_y1-top_left_y1) + (bottom_right_x2-top_left_x2)*(bottom_right_y2-top_left_y2) - intersect_area 
	iou = float(intersect_area)/float(total_area+0.0)
	return iou

def perform_scale_down(image,max_size_allowed):

	orig_h = image.shape[0]
	orig_w = image.shape[1]

	new_h = orig_h
	new_w = orig_w

	if new_h > max_size_allowed[1]:
		new_w = float(new_w*max_size_allowed[1]) / float(new_h)
		new_h = max_size_allowed[1]

	if new_w > max_size_allowed[0]:
		new_h = float(new_h*max_size_allowed[0]) / float(new_w)
		new_w = max_size_allowed[0]

	if new_h != orig_h or new_w != orig_w:
		return cv2.resize(image, (int(new_w), int(new_h))),float(orig_h)/float(new_h)
	else:
		return image,1.0

def perform_selective_search(img,ground_truth,gt2):
	
	rects=[]

	max_size = (500,500)
	h = float(img.shape[0])
	w = float(img.shape[1])
	img,scale  = perform_scale_down(img,max_size)
	dlib.find_candidate_object_locations(img, rects, kvals=(50, 200, 2), min_size=1200)
	filter_positive_rects=[]
	filter_negative_rects_hard=[]
	filter_negative_rects_easy=[]
	max_negatives = 50
	hard_negative_ratio = 0.6
	iou_list = []
	filter_negative_rects=[]
	for rect in rects:
		descaled_top_x = (rect.left()*scale)
		descaled_top_y = (rect.top()*scale)
		descaled_bottom_x = (rect.right()*scale)
		descaled_bottom_y = (rect.bottom()*scale)
		descaled_width = descaled_bottom_x - descaled_top_x#int(rect.width()*scale)
		descaled_height = descaled_bottom_y - descaled_top_y #int(rect.height()*scale)
		descaled_center_x = descaled_top_x + (descaled_width/2.0)
		descaled_center_y = descaled_top_y + (descaled_height/2.0)

		#iou,a1,a2 = rect_overlap_rate(gt2,(descaled_top_x,descaled_top_y,descaled_width,descaled_height))
		iou = calc_2D_IOU(ground_truth,(descaled_top_x,descaled_top_y,descaled_bottom_x,descaled_bottom_y))
		
		iou_list.append(iou)
		if DEBUG_FLAG:
			debug_fp_csv.writerow([iou,ground_truth[0],ground_truth[1],ground_truth[2],ground_truth[3],rect.left(),rect.top(),rect.right(),rect.bottom()])
		if  iou > 0.50:
			if VIS_FLAG:
				filter_positive_rects.append([int(descaled_top_x),int(descaled_top_y),int(descaled_bottom_x),int(descaled_bottom_y)])
			else:
				filter_positive_rects.append([descaled_top_y/h,descaled_top_x/w,descaled_bottom_y/h,descaled_bottom_x/w,
					descaled_center_x,descaled_center_y,descaled_width,descaled_height])
		elif iou <= 0.0:
			if VIS_FLAG:
				filter_negative_rects.append([int(descaled_top_x),int(descaled_top_y),int(descaled_bottom_x),int(descaled_bottom_y)])
			else:
				filter_negative_rects.append([descaled_top_y/h,descaled_top_x/w,descaled_bottom_y/h,descaled_bottom_x/w,
					descaled_center_x,descaled_center_y,descaled_width,descaled_height])
		# elif 0.25 <= iou < 0.35:
		# 	if VIS_FLAG:
		# 		filter_negative_rects_hard.append([int(descaled_top_x),int(descaled_top_y),int(descaled_bottom_x),int(descaled_bottom_y)])
		# 	else:
		# 		filter_negative_rects_hard.append([descaled_top_y/h,descaled_top_x/w,descaled_bottom_y/h,descaled_bottom_x/w])
		# elif iou < 0.25:
		# 	if VIS_FLAG:
		# 		filter_negative_rects_easy.append([int(descaled_top_x),int(descaled_top_y),int(descaled_bottom_x),int(descaled_bottom_y)])
		# 	else:
		# 		filter_negative_rects_easy.append([descaled_top_y/h,descaled_top_x/w,descaled_bottom_y/h,descaled_bottom_x/w])

	# if len(filter_negative_rects_easy) + len(filter_negative_rects_hard) < max_negatives:
	# 	filter_negative_rects = np.concatenate([np.asarray(filter_negative_rects_easy),np.asarray(filter_negative_rects_hard)],axis=0)
	# 	filter_negative_rects = filter_negative_rects.tolist()
	# else:
	# 	if len(filter_negative_rects_hard) < int(hard_negative_ratio*max_negatives):
	# 		index = np.random.choice(np.arange(len(filter_negative_rects_easy)),max_negatives -len(filter_negative_rects_hard)
	# 			,replace=False)
	# 		filter_negative_rects_easy = np.asarray(filter_negative_rects_easy)[index,:].tolist()
	# 	elif len(filter_negative_rects_easy) < int((1-hard_negative_ratio)*max_negatives):
	# 		index = np.random.choice(np.arange(len(filter_negative_rects_hard)),max_negatives -len(filter_negative_rects_easy),
	# 			replace=False)
	# 		filter_negative_rects_hard = np.asarray(filter_negative_rects_hard)[index,:].tolist()
	# 	else:
	# 		index = np.random.choice(np.arange(len(filter_negative_rects_hard)),int(hard_negative_ratio*max_negatives),replace=False)
	# 		filter_negative_rects_hard = np.asarray(filter_negative_rects_hard)[index,:].tolist()
	# 		index = np.random.choice(np.arange(len(filter_negative_rects_easy)),int((1-hard_negative_ratio)*max_negatives),replace=False)
	# 		filter_negative_rects_easy = np.asarray(filter_negative_rects_easy)[index,:].tolist()
	# 	filter_negative_rects = np.concatenate([np.asarray(filter_negative_rects_easy),np.asarray(filter_negative_rects_hard)],axis=0)
	# 	filter_negative_rects = filter_negative_rects.tolist()

	# Jittering the ground truth
	
	gt_top_x1 = ground_truth[0]
	gt_top_y1 = ground_truth[1]
	gt_bottom_x2 = ground_truth[2]
	gt_bottom_y2 = ground_truth[3]

	gt_w = gt_bottom_x2 - gt_top_x1
	gt_h = gt_bottom_y2 - gt_top_y1

	w_list = np.arange(-0.5*gt_w,0.5*gt_w,0.1*gt_w).tolist()
	h_list = np.arange(-0.5*gt_h,0.5*gt_h,0.1*gt_h).tolist()

	for w_shift in w_list:
		for h_shift in h_list:
			new_x1 = gt_top_x1 + w_shift
			new_y1 = gt_top_y1 + h_shift
			new_x2 = gt_bottom_x2 + w_shift
			new_y2 = gt_bottom_y2 + h_shift
			
			if new_x1 < 0.0:
				new_x1 = 0.0
			elif new_x1 > w :
				new_x1 = w

			if new_y1 < 0.0:
				new_y1 = 0.0
			elif new_y1 > h :
				new_y1 = h

			if new_x2 < 0.0:
				new_x2 = 0.0
			elif new_x2 > w :
				new_x2 = w
			
			if new_y2 < 0.0:
				new_y2 = 0.0
			elif new_y2 > h :
				new_y2 = h

			iou = calc_2D_IOU(ground_truth,(new_x1,new_y1,new_x2,new_y2))
			if  iou > 0.50:
				if VIS_FLAG:
					filter_positive_rects.append([int(new_x1),int(new_y1),int(new_x2),int(new_y2)])
				else:
					descaled_width = new_x2 - new_x1#int(rect.width()*scale)
					descaled_height = new_y2 - new_y1#int(rect.height()*scale)
					descaled_center_x = new_x1 + (descaled_width/2.0)
					descaled_center_y = new_y1 + (descaled_height/2.0)
					filter_positive_rects.append([(new_y1)/h,(new_x1)/w,(new_y2)/h,(new_x2)/w,
						descaled_center_x,descaled_center_y,descaled_width,descaled_height])

	if VIS_FLAG:				
		return filter_positive_rects,filter_negative_rects
	else:
		return np.asarray(filter_positive_rects).astype(np.float32),np.asarray(filter_negative_rects).astype(np.float32)

def visualise(img,rects,gt):

	
	for rect in rects:
		#new_img = img
		r,g,b = np.random.randint(0,255,3)

		cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(b,g,r),2)
		cv2.imshow('result',img)
		cv2.namedWindow('result', cv2.WINDOW_NORMAL)
		cv2.resizeWindow('result', 320,240)
		
	
	cv2.rectangle(img,(gt[0],gt[1]),(gt[2],gt[3]),(0,255,0),1)
	cv2.imshow('result',img/255.0)
	cv2.waitKey(0)

 

def extract_tfrecord(it):
	try:
		example = tf.train.Example()
		example.ParseFromString(it)
		session = tf.Session()
		img_string = example.features.feature['image_raw'].bytes_list.value[0]
		img_width = int(example.features.feature['width'].int64_list.value[0])
		img_height = int(example.features.feature['height'].int64_list.value[0])

		img_2d = np.fromstring(img_string, dtype=np.uint8).reshape(img_height,img_width,3)

		loc_x = int(example.features.feature['loc_x'].int64_list.value[0])
		loc_y = int(example.features.feature['loc_y'].int64_list.value[0])
		loc_w = int(example.features.feature['loc_w'].int64_list.value[0])
		loc_h = int(example.features.feature['loc_h'].int64_list.value[0])
		face_id = int(example.features.feature['face_id'].int64_list.value[0])

		landmark_string = example.features.feature['landmarks'].bytes_list.value[0]
		landmarks = np.fromstring(landmark_string, dtype=np.float32).reshape(21,2)
		sex = int(example.features.feature['sex'].int64_list.value[0])
		roll = float(example.features.feature['roll'].float_list.value[0])
		pitch = float(example.features.feature['pitch'].float_list.value[0])
		yaw = float(example.features.feature['yaw'].float_list.value[0])

		hard_postives,hard_negatives = perform_selective_search(img_2d,(loc_x,loc_y,loc_x+loc_w,loc_y+loc_h),(loc_x,loc_y,loc_w,loc_h))
		print "****************************"
		if os.path.exists('locations_test/'+str(face_id)):
			print face_id
			np.save('locations_test/'+str(face_id)+'/positive.npy',hard_postives)
			np.save('locations_test/'+str(face_id)+'/negative.npy',hard_negatives)
		else:
			os.mkdir('locations_test/'+str(face_id))
			np.save('locations_test/'+str(face_id)+'/positive.npy',hard_postives)
			np.save('locations_test/'+str(face_id)+'/negative.npy',hard_negatives)

		if VIS_FLAG:

			visualise(img_2d,hard_postives,(loc_x,loc_y,loc_x+loc_w,loc_y+loc_h))
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																						
		if MAKE_TF_RECORD:
			
			if hard_postives.shape[0] > 0:

				resized_and_cropped_image_pos = tf.image.crop_and_resize(img_2d[np.newaxis,:].astype(np.float32),hard_postives, [0]*hard_postives.shape[0], crop_size=[227,227]).eval(session=session)
				np.save('positive.npy',resized_and_cropped_image_pos)
				# np.random.shuffle(resized_and_cropped_image_pos)
				# resized_and_cropped_image_pos = resized_and_cropped_image_pos[:40,:,:,:]
				
				# example_pos = tf.train.Example(features=tf.train.Features(feature={
				# 	'image_raw':_bytes_feature(resized_and_cropped_image_pos.astype(np.uint8).tostring()),
				# 	'width': _int64_feature(img_2d.shape[1]),
				# 	'height': _int64_feature(img_2d.shape[0]),
				# 	'batch_size': _int64_feature(resized_and_cropped_image_pos.shape[0]),
				# 	'roll': _float_feature(roll),
				# 	'pitch':_float_feature(pitch),
				# 	'yaw':_float_feature(yaw),
				# 	'landmarks':_bytes_feature(landmarks.tostring()),
				# 	'gender':_int64_feature(sex),
				# 	'locations':_bytes_feature(hard_postives.tostring())
				# 	}))
				# writer_pos.write(example_pos.SerializeToString())
				#np.save('pos_files/'+str(face_id)+'.npy',resized_and_cropped_image_pos)

			if hard_negatives.shape[0] > 0:

				resized_and_cropped_image_neg = tf.image.crop_and_resize(img_2d[np.newaxis,:].astype(np.float32),hard_negatives, [0]*hard_negatives.shape[0], crop_size=[227,227]).eval(session=session)
				np.save('negative.npy',resized_and_cropped_image_neg)

				# np.random.shuffle(resized_and_cropped_image_neg)
				# resized_and_cropped_image_neg = resized_and_cropped_image_neg[:40,:,:,:]

				# example_neg = tf.train.Example(features=tf.train.Features(feature={
				# 	'image_raw':_bytes_feature(resized_and_cropped_image_neg.astype(np.uint8).tostring()),
				# 	'width': _int64_feature(img_2d.shape[1]),
				# 	'height': _int64_feature(img_2d.shape[0]),
				# 	'batch_size': _int64_feature(resized_and_cropped_image_neg.shape[0]),
				# 	'roll': _float_feature(roll),
				# 	'pitch':_float_feature(pitch),
				# 	'yaw':_float_feature(yaw),
				# 	'landmarks':_bytes_feature(landmarks.tostring()),
				# 	'gender':_int64_feature(sex),
				# 	'locations':_bytes_feature(hard_negatives.tostring())
				# 	}))
				# writer_neg.write(example_neg.SerializeToString())
		return 1
	except Exception as e:
		print e
		return 0
		#np.save('neg_files/'+str(face_id)+'.npy',resized_and_cropped_image_neg)

def listener(q):
	tfrecords_training_pos_filename = 'aflw_training_pos.tfrecords'
	tfrecords_training_neg_filename = 'aflw_training_neg.tfrecords'

	writer_pos = tf.python_io.TFRecordWriter(tfrecords_training_pos_filename)
	writer_neg = tf.python_io.TFRecordWriter(tfrecords_training_neg_filename)
	#f = open('check.txt','wb')
	while(1):
		m = q.get()
		if m == 'kill':
			break
		writer_pos.write(m.SerializeToString())
	#f.close()
	writer_neg.close()
	writer_pos.close()

if __name__ == '__main__':
	
	#pool = Pool(processes=4)
	
	# manager = Manager()
	# q = manager.Queue()
	# watcher = pool.apply_async(listener, (q,))

	#start_time = time.clock()
	# jobs =[]
	# for i in range(10):
	# 	job = pool.apply_async(extract_tfrecord,(record_iterator.next(),q))
	# 	jobs.append(job)
	# for job in jobs:
	# 	job.wait()
	# 	job.get()
	# q.put('kill')
	# pool.close()
	# results = [pool.apply_async(extract_tfrecord,args=(record_iterator.next())) for i in range(100)]
	# pool.close()
	# pool.join()
	# results = [p.get() for p in results]
		#jobs.append(job)
	# for job in jobs:
	# 	job.wait()
	# 	job.get()				

	# result = pool.map_async(extract_tfrecord,[record_iterator.next() for i in range(100)])
	# result.get()
	#print "Done in {}".format(time.clock() - start_time)
	

	while (1):
		try:
			extract_tfrecord(record_iterator.next())
			
		except Exception as e:
			print e
			break
	writer_pos.close()
	writer_neg.close()


	
	


