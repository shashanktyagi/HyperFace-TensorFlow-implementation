#!/home/shashank/anaconda2/bin
import dlib
from skimage import io
import cv2
import numpy as np
import tensorflow as tf
import csv
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

pool = ThreadPool(4)
DEBUG_FLAG = False
tfrecords_full_filename = 'aflw.tfrecords'
tfrecords_training_filename = 'aflw_training.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_training_filename)

if DEBUG_FLAG:
	debug_fp = open('debug.csv','wb')
	debug_fp_csv = csv.writer(debug_fp)

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


def perform_selective_search(img,ground_truth):
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

	return filter_positive_rects,filter_negative_rects

def visualise(img,rects):
	cv2.namedWindow('result', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('result', 600,600)
	for rect in rects:
		cv2.rectangle(img,(rect.left(),rect.top()),(rect.right(),rect.bottom()),(0,255,0),2)
	
	cv2.imshow('result',img)
	cv2.waitKey(0)


def extract_tfrecord():
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_full_filename)
	
	for string_record in record_iterator:
		example = tf.train.Example()
		example.ParseFromString(string_record)

		img_string = example.features.feature['image_raw'].bytes_list.value[0]
		img_width = int(example.features.feature['width'].int64_list.value[0])
		img_height = int(example.features.feature['height'].int64_list.value[0])
		img_2d = np.fromstring(img_string, dtype=np.uint8).reshape(img_height,img_width,3)
		loc_x = int(example.features.feature['loc_x'].int64_list.value[0])
		loc_y = int(example.features.feature['loc_y'].int64_list.value[0])
		loc_w = int(example.features.feature['loc_w'].int64_list.value[0])
		loc_h = int(example.features.feature['loc_h'].int64_list.value[0])
		hard_postives,hard_negatives = perform_selective_search(img_2d,(loc_x,loc_y,loc_x+loc_w,loc_y+loc_h))

		resized_and_cropped_image = tf.image.crop_and_resize(img_2d[np.newaxis,:], boxes, [0]*hard_postives.shape[0], crop_size=[227,227])
		break				
		#visualise(img_2d,hard_postives)
		break


if __name__ == '__main__':
	extract_tfrecord()




