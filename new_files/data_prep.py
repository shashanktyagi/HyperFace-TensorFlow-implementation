import numpy as np 
import tensorflow as tf
from skimage import io
from skimage import color
import sqlite3
import cv2
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from pdb import set_trace as brk
import sys
# The following are the database properties available (last updated version 2012-11-28):
#
# databases: db_id, path, description
# faceellipse: face_id, x, y, ra, rb, theta, annot_type_id, upsidedown
# faceimages: image_id, db_id, file_id, filepath, bw, widht, height
# facemetadata: face_id, sex, occluded, glasses, bw, annot_type_id
# facepose: face_id, roll, pitch, yaw, annot_type_id
# facerect: face_id, x, y, w, h, annot_type_id
# faces: face_id, file_id, db_id
# featurecoords: face_id, feature_id, x, y
# featurecoordtype: feature_id, descr, code, x, y, z
# AFLW 21 points landmark
#  0|LeftBrowLeftCorner
#  1|LeftBrowCenter
#  2|LeftBrowRightCorner
#  3|RightBrowLeftCorner
#  4|RightBrowCenter
#  5|RightBrowRightCorner
#  6|LeftEyeLeftCorner
#  7|LeftEyeCenter
#  8|LeftEyeRightCorner
#  9|RightEyeLeftCorner
#  10|RightEyeCenter
#  11|RightEyeRightCorner
#  12|LeftEar
#  13|NoseLeft
#  14|NoseCenter
#  15|NoseRight
#  16|RightEar
#  17|MouthLeftCorner
#  18|MouthCenter
#  19|MouthRightCorner
#  20|ChinCenter

select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h,faceimages.image_id,facemetadata.sex"
from_string = "faceimages, faces, facepose, facerect,facemetadata"
where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id and faces.face_id = facemetadata.face_id"
query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string




conn = sqlite3.connect('/home/shashank/Documents/CSE-252C/AFLW/aflw/data/aflw.sqlite')
c = conn.cursor()

img_path = '/home/shashank/Documents/CSE-252C/AFLW/'
loc_file_path = '/home/shashank/Documents/CSE-252C/hyperface/code/locations_test/'
tfrecords_train_filename = 'test_check.tfrecords'
tfrecords_test_filename = 'aflw_test_new.tfrecords'

writer_train = tf.python_io.TFRecordWriter(tfrecords_train_filename)
writer_test = tf.python_io.TFRecordWriter(tfrecords_test_filename)

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def test_names():
	l=[]
	names = os.listdir(img_path+'0')
	random.shuffle(names)
	l.append(['0/'+name for name in names[:300]])
	
	names = os.listdir(img_path+'2')
	random.shuffle(names)
	l.append(['2/'+name for name in names[:300]])

	names = os.listdir(img_path+'3')
	random.shuffle(names)
	l.append(['3/'+name for name in names[:400]])

	return l[0]+l[1]+l[2]

def make_tfrecord(test_images):
	
	it_test =0
	it_train = 0
	gender_dict={'m':1,'f':0}

	for row in (c.execute(query_string)):
		'''
		row[0] = image path str
		row[1] = face id int
		row[2] = roll float
		row[3] = pitch float
		row[4] = yaw float
		row[5] = x int
		row[6] = y int
		row[7] = w int
		row[8] = h int
		'''
		
		
		center_x = float(row[5]) + float(row[7])/2
		center_y = float(row[6]) + float(row[8])/2

		
		if not os.path.exists(loc_file_path+str(row[1])):
			continue

		select_str = "coords.feature_id, coords.x, coords.y"
		from_str = "featurecoords coords"
		where_str = "coords.face_id = {}".format(row[1])
		query_str = "SELECT " + select_str + " FROM " + from_str + " WHERE " + where_str
		landmark = np.zeros((21,2)).astype(np.float32)
		visibility = np.zeros((21,1)).astype(np.int32)

		c2 = conn.cursor()			
		
		for xx in c2.execute(query_str):
			landmark[xx[0]-1][0] = xx[1]#(xx[1] - center_x)/float(row[7])
			landmark[xx[0]-1][1] = xx[2]#(xx[2] - center_y)/float(row[8])
			visibility[xx[0]-1] = 1
		landmark = landmark.reshape(-1,42)

		c2.close()

		try:

			img_raw = (np.asarray(cv2.imread(img_path+row[0])).astype(np.float32))/255.0
			cv2.imwrite('save_im.jpg',img_raw*255)
			landmark_pos = None

			if len(img_raw.shape) !=3:
				continue#img_raw = color.gray2rgb(img_raw)
			if len(img_raw.shape) !=3 or img_raw.shape[2] != 3:
				continue
			print row[1]
			
			w = img_raw.shape[1]
			h = img_raw.shape[0]
			if os.path.isfile(loc_file_path+str(row[1])+'/positive.npy'):
				pos_locs = np.load(loc_file_path+str(row[1])+'/positive.npy')[:,:4]
				cof_locs = np.tile(np.load(loc_file_path+str(row[1])+'/positive.npy')[:,4:6],(1,21))
				dim_locs = np.tile(np.load(loc_file_path+str(row[1])+'/positive.npy')[:,6:8],(1,21))
				n_pos_locs = pos_locs.shape[0]
				
				landmark_pos = (landmark - cof_locs)/dim_locs
				visibility_pos = np.ones((landmark_pos.shape[0],21))
				visibility_pos[(np.where(landmark_pos > 0.5)[0],np.where(landmark_pos > 0.5)[1]/2)] = 0
				visibility_pos[(np.where(landmark_pos < -0.5)[0],np.where(landmark_pos < -0.5)[1]/2)] = 0

				# visibility_pos[np.where(landmark_pos)]
				pos_locs = pos_locs.astype(np.float32).tostring()

			# 	if pos_locs.shape[0] > 0:
			# 		pos_locs = np.concatenate([pos_locs,np.asarray([row[6]/float(h),row[5]/float(w),
			# 			(row[6]+row[8])/float(h),(row[5]+row[7])/float(w)]).reshape(1,4)],axis=0)
				
			# 		n_pos_locs = pos_locs.shape[0]
				
			# 		pos_locs = pos_locs.astype(np.float32).tostring()
			# 	else:
			# 		pos_locs = np.asarray([[row[6]/float(h),row[5]/float(w),(row[6]+row[8])/float(h),(row[5]+row[7])/float(w)]]).reshape(1,4)
			# 		n_pos_locs = pos_locs.shape[0]
			# 		pos_locs = pos_locs.astype(np.float32).tostring()	

			# else:
			# 	pos_locs = np.asarray([[row[6]/float(h),row[5]/float(w),(row[6]+row[8])/float(h),(row[5]+row[7])/float(w)]]).reshape(1,4)
			# 	n_pos_locs = pos_locs.shape[0]
			# 	pos_locs = pos_locs.astype(np.float32).tostring()
			


			if os.path.isfile(loc_file_path+str(row[1])+'/negative.npy'):
				neg_locs = np.load(loc_file_path+str(row[1])+'/negative.npy')[:,:4]
				n_neg_locs = neg_locs.shape[0]
				cof_locs = np.tile(np.load(loc_file_path+str(row[1])+'/negative.npy')[:,4:6],(1,21))
				dim_locs = np.tile(np.load(loc_file_path+str(row[1])+'/negative.npy')[:,6:8],(1,21))
				
				landmark_neg = (landmark - cof_locs)/dim_locs
				visibility_neg = np.zeros((landmark_neg.shape[0],21))

				# visibility_pos[np.where(landmark_pos)]
				neg_locs = neg_locs.astype(np.float32).tostring()
			
			all_landmarks =   np.concatenate([landmark_pos,landmark_neg],axis=0)
			all_visibilities = np.concatenate([visibility_pos,visibility_neg],axis=0)
			all_landmarks = all_landmarks.astype(np.float32).tostring()
			all_visibilities = all_visibilities.astype(np.int32).tostring()

			img_raw = img_raw.tostring()
				
			print "{},{}".format(n_pos_locs,n_neg_locs)

			pose_array = np.asarray([row[2],row[3],row[4]]).astype(np.float32)
			

			pose_array = pose_array.tostring()
			# landmark = landmark.tostring()
			# visibility=visibility.tostring()
			

			example = tf.train.Example(features=tf.train.Features(feature={
				'image_raw':_bytes_feature(img_raw),
				'width': _int64_feature(w),
				'height': _int64_feature(h),
				'face_id': _int64_feature(row[1]),
				'pose': _bytes_feature(pose_array),
				'loc_x': _int64_feature(row[5]),
				'loc_y': _int64_feature(row[6]),
				'loc_w': _int64_feature(row[7]),
				'loc_h': _int64_feature(row[8]),
				'gender':_int64_feature(gender_dict[row[10]]),
				'landmarks':_bytes_feature(all_landmarks),
				'visibility':_bytes_feature(all_visibilities),
				'pos_locs':_bytes_feature(pos_locs),
				'neg_locs':_bytes_feature(neg_locs),
				'n_pos_locs':_int64_feature(n_pos_locs),
				'n_neg_locs':_int64_feature(n_neg_locs)
				}))
			
			writer_train.write(example.SerializeToString())
			it_train += 1
			break
			# if it_train >= 1:
			# 	break
			# if row[0] in test_images:
			# 	writer_test.write(example.SerializeToString())
			# 	it_test += 1
			# else:
			# 	writer_train.write(example.SerializeToString())
			# 	it_train += 1

		except Exception as e:
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)
		
		
	print it_test,it_train	
	c.close()
	writer_train.close()
	writer_test.close()

def extract_tfrecord():
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_train_filename)
	count =0
	for string_record in tqdm(record_iterator):
		
		count += 1
		example = tf.train.Example()
		example.ParseFromString(string_record)

		img_string = example.features.feature['image_raw'].bytes_list.value[0]
		landmark_string = example.features.feature['landmarks'].bytes_list.value[0]
		landmarks = np.fromstring(landmark_string, dtype=np.float32).reshape(21,2)
		img_width = int(example.features.feature['width'].int64_list.value[0])
		img_height = int(example.features.feature['height'].int64_list.value[0])
		
		img_2 = np.fromstring(img_string, dtype=np.uint8).reshape(-1,1)
		
		img_1d = np.fromstring(img_string, dtype=np.uint8).reshape(img_height,img_width,3)
		print img_1d.shape
		loc_x = int(example.features.feature['loc_x'].int64_list.value[0])
		loc_y = int(example.features.feature['loc_y'].int64_list.value[0])
		loc_w = int(example.features.feature['loc_w'].int64_list.value[0])
		loc_h = int(example.features.feature['loc_h'].int64_list.value[0])
		sex = int(example.features.feature['gender'].int64_list.value[0])
		
		
		# center_x = img_width/2.0
		# center_y = img_height/2.0 

		# centers = np.tile(np.array([center_x,center_y]).reshape(1,2),(21,1))
		# normalized = landmarks - centers
		# w_h = np.tile(np.array([img_width,img_height]).reshape(1,2),(21,1))

		# normalized = normalized/w_h
		
		# for i in range(normalized.shape[0]):
		# 	if i == 5 or i == 9 or i==15 or i==16:
		# 		continue
		# 	point_x = normalized[i][0]*img_width + img_width/2.0
		# 	point_y = normalized[i][1]*img_height + img_height/2.0
			
		# 	cv2.circle(img_1d,(int(point_x),int(point_y)), 1, (0,0,255), 2)

		# cv2.rectangle(img_1d,(loc_x,loc_y),(loc_x+loc_w,loc_y+loc_h),(0,255,0),3)
		# cv2.imshow('result',img_1d)
		# cv2.waitKey(0)
		
		
		
if __name__ == '__main__':
	test_images = test_names()
	print len(test_images)
	make_tfrecord(test_images)
	#extract_tfrecord()

