import numpy as np 
import tensorflow as tf
#from skimage import io
import sqlite3
#import cv2
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm

# select_string = "faceimages.filepath, faces.face_id, facepose.roll, facepose.pitch, facepose.yaw, facerect.x, facerect.y, facerect.w, facerect.h"
# from_string = "faceimages, faces, facepose, facerect"
# where_string = "faces.face_id = facepose.face_id and faces.file_id = faceimages.file_id and faces.face_id = facerect.face_id"
# query_string = "SELECT " + select_string + " FROM " + from_string + " WHERE " + where_string

# conn = sqlite3.connect('/home/shashank/Documents/CSE-252C/AFLW/aflw/data/aflw.sqlite')
# c = conn.cursor()

img_path = '/home/shashank/Documents/CSE-252C/AFLW/'

# tfrecords_train_filename = 'aflw_train.tfrecords'
# tfrecords_test_filename = 'aflw_test.tfrecords'
tfrecords_filename = 'aflw_train.tfrecords'
# writer_train = tf.python_io.TFRecordWriter(tfrecords_train_filename)
# writer_test = tf.python_io.TFRecordWriter(tfrecords_test_filename)

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

	for row in c.execute(query_string):
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
		
		try:
			img_raw = np.asarray(io.imread(img_path+row[0]))
			w = img_raw.shape[1]
			h = img_raw.shape[0]
		
			img_raw = img_raw.tostring()
			example = tf.train.Example(features=tf.train.Features(feature={
				'image_raw':_bytes_feature(img_raw),
				'width': _int64_feature(w),
				'height': _int64_feature(h),
				'face_id': _int64_feature(row[1]),
				'roll': _float_feature(row[2]),
				'pitch': _float_feature(row[3]),
				'yaw': _float_feature(row[4]),
				'loc_x': _int64_feature(row[5]),
				'loc_y': _int64_feature(row[6]),
				'loc_w': _int64_feature(row[7]),
				'loc_h': _int64_feature(row[8])
				}))
			
			if row[0] in test_images:
				writer_test.write(example.SerializeToString())
				it_test += 1
			else:
				writer_train.write(example.SerializeToString())
				it_train += 1

		except:
			print row[0]
		
		if it_train > 50:
			break
	print it_test,it_train	
	c.close()
	writer_train.close()
	writer_test.close()

def extract_tfrecord(session):
	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
	save_data = None
	save_euler = []
	for string_record in record_iterator:
		example = tf.train.Example()
		example.ParseFromString(string_record)

		img_string = example.features.feature['image_raw'].bytes_list.value[0]
		img_width = int(example.features.feature['width'].int64_list.value[0])
		img_height = int(example.features.feature['height'].int64_list.value[0])
		img_1d = np.fromstring(img_string, dtype=np.uint8).reshape(img_height,img_width,3)
		loc_x = int(example.features.feature['loc_x'].int64_list.value[0])
		loc_y = int(example.features.feature['loc_y'].int64_list.value[0])
		loc_w = int(example.features.feature['loc_w'].int64_list.value[0])
		loc_h = int(example.features.feature['loc_h'].int64_list.value[0])
		roll = float(example.features.feature['roll'].float_list.value[0])
		yaw = float(example.features.feature['yaw'].float_list.value[0])
		pitch = float(example.features.feature['pitch'].float_list.value[0])

		boxes = np.asarray([[loc_y/float(img_height),loc_x/float(img_width),(loc_y+loc_h)/float(img_height),(loc_x+loc_w)/float(img_width)]])
		resized_and_cropped_image = tf.image.crop_and_resize(img_1d[np.newaxis,:,:,:].astype(np.float32), boxes.astype(np.float32), [0]*1, crop_size=[227,227])
		if save_data is not None:
			save_data = np.concatenate([save_data,resized_and_cropped_image.eval(session=session)],axis=0)
		else:
			save_data = resized_and_cropped_image.eval(session=session)
		save_euler.append([roll,yaw,pitch])

	np.save('truth_data.npy',save_data)
	np.save('annotations.npy',np.asarray(save_euler))
	
		# cv2.rectangle(img_1d,(loc_x,loc_y),(loc_x+loc_w,loc_y+loc_h),(0,255,0),3)
		# cv2.imshow('result',img_1d)
		# cv2.waitKey(0)
				
		
if __name__ == '__main__':
	#test_images = test_names()
	#make_tfrecord(test_images)
	session = tf.Session()
	extract_tfrecord(session)

