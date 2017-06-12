import cv2
import numpy as np
import tensorflow as tf
# import pdb
# pdb.set_trace()
aflw_template_landmark_coords=np.array([[-0.479962468147, 0.471864163876],[-0.30303606391, 0.508996844292],[-0.106451146305, 0.498075485229],[0.106451146305, 0.498075485229],[0.30303606391, 0.508996844292],[0.479962468147, 0.471864163876],[-0.447198301554, 0.321149080992],[-0.318325966597, 0.325517624617],[-0.163242310286, 0.308043420315],[0.163242310286, 0.308043420315],[0.318325966597, 0.325517624617],[0.447198301554, 0.321149080992],[-0.674257874489, -0.151652157307],[-0.170000001788, -0.075740583241],[0.0, 0.0],[0.170000001788, -0.075740583241],[0.674257874489, -0.151652157307],[-0.272456139326, -0.347239643335],[0.0, -0.336318254471],[0.272456139326, -0.347239643335],[0.0, -0.737950384617]], dtype=np.float32)
# tfrecords_train_filename = '/home/shashank/Documents/CSE-252C/hyperface/code/aflw_train.tfrecords'

def region_proposal(landmark_pts,visible_landmark_index,image_size,pad=0.1):
	
	x_template,y_template,w_template,h_template = cv2.boundingRect(aflw_template_landmark_coords)
	
	x_selective,y_selective,w_selective,h_selective = cv2.boundingRect(landmark_pts.astype(np.float32))

	x_selective = x_selective - (pad*w_selective)/2.0
	y_selective = y_selective - (pad*h_selective)/2.0
	w_selective = w_selective *(1+ pad)
	h_selective = h_selective *(1+ pad)

	visible_template_landmarks = aflw_template_landmark_coords[visible_landmark_index,:]

	#Now we have got the corresponding points or features in the two images. Using 2D Homography, find the projection matrix.
	#For the homography we need at least 4 features,hence
	if len(visible_landmark_index) < 4:
		return (0,0,0,0)

	H,__ = cv2.findHomography(visible_template_landmarks,landmark_pts,cv2.RANSAC)

	if H is None:
		return (0,0,0,0)
	source_pts = np.asarray([ [x_template,y_template,1.0],[x_template,y_template+h_template,1.0],[x_template+w_template,y_template,1.0],[x_template+w_template,y_template+h_template,1.0] ]).astype(np.float32)
	
	dst_points = np.dot(H,source_pts.T)
	dst_points = dst_points/dst_points[2,:]
	dst_points = dst_points[:2,:]

	
	min_x_proposed = np.min(dst_points[0,:])
	min_y_proposed = np.min(dst_points[1,:])

	max_x_proposed = np.max(dst_points[0,:])
	max_y_proposed = np.max(dst_points[1,:])

	w_proposed = max_x_proposed - min_x_proposed
	h_proposed = max_y_proposed - min_y_proposed

	final_x1 = min(min_x_proposed,x_selective)
	final_y1 = min(min_y_proposed,y_selective)

	final_x2 = max(max_x_proposed,x_selective+w_selective)
	final_y2 = max(max_y_proposed,y_selective+h_selective)

	final_x1 = max(final_x1,0)
	final_y1 = max(final_y1,0)

	final_x2 = min(image_size[1],final_x2)
	final_y2 = min(image_size[0],final_y2)

	return (final_y1,final_x1,final_y2,final_x2)

# def extract_tfrecord():
# 	record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_train_filename)

# 	for string_record in record_iterator:
# 		example = tf.train.Example()
# 		example.ParseFromString(string_record)

# 		img_string = example.features.feature['image_raw'].bytes_list.value[0]
# 		landmark_string = example.features.feature['landmarks'].bytes_list.value[0]
# 		landmarks = np.fromstring(landmark_string, dtype=np.float32).reshape(21,2)
# 		img_width = int(example.features.feature['width'].int64_list.value[0])
# 		img_height = int(example.features.feature['height'].int64_list.value[0])
# 		img_1d = np.fromstring(img_string, dtype=np.uint8).reshape(img_height,img_width,3)
# 		loc_x = int(example.features.feature['loc_x'].int64_list.value[0])
# 		loc_y = int(example.features.feature['loc_y'].int64_list.value[0])
# 		loc_w = int(example.features.feature['loc_w'].int64_list.value[0])
# 		loc_h = int(example.features.feature['loc_h'].int64_list.value[0])
# 		sex = int(example.features.feature['sex'].int64_list.value[0])
		
# 		center_x = loc_x + (loc_w/2.0)
# 		center_y = loc_y + (loc_h/2.0) 

# 		centers = np.tile(np.array([center_x,center_y]).reshape(1,2),(21,1))
# 		normalized = landmarks - centers
# 		w_h = np.tile(np.array([loc_w,loc_h]).reshape(1,2),(21,1))

# 		normalized = normalized/w_h
# 		landmarks_for_irp =[]
# 		visibility_for_irp=[]
		
# 		for i in range(normalized.shape[0]):
# 			if (landmarks[i][0] == 0.0) and (landmarks[i][0] == 0.0) :
# 				visibility_for_irp.append([0])
# 				continue
# 			else:
# 				visibility_for_irp.append([1])	
# 			point_x = normalized[i][0]*loc_w + center_x
# 			point_y = normalized[i][1]*loc_h + center_y
# 			landmarks_for_irp.append([point_x,point_y])	
			
# 			#cv2.circle(img_1d,(int(point_x),int(point_y)), 1, (0,0,255), 2)
# 		landmarks_for_irp = np.asarray(landmarks_for_irp)
# 		visibility_for_irp = np.asarray(visibility_for_irp)
# 		l1,l2,l3,l4 = region_proposal(landmarks_for_irp,visibility_for_irp,(img_width,img_height))
# 		cv2.rectangle(img_1d,(int(l1),int(l2)),(int(l3),int(l4)),(0,255,0),3)
# 		cv2.imshow('result',img_1d)
# 		cv2.waitKey(0)

# if __name__ == '__main__':
# 	extract_tfrecord()



