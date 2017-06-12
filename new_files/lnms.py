import os
import sys
import numpy as np
# import pdb
# pdb.set_trace()
def fast_nms(ip_boxes, ov_threshold):

	if len(ip_boxes) == 0:
		return None

	#Save the Area Computation
	area = (ip_boxes[:,0] - ip_boxes[:,2])*(ip_boxes[:,1] - ip_boxes[:,3])
	area = area.reshape(-1,1)

	#sorted_y_index = np.argsort(ip_boxes[:,3])
	sorted_y_index = np.argsort(area[:,0])
	keep = {}
	
	while len(sorted_y_index) > 0:
		index = sorted_y_index[-1]
		
		to_find = sorted_y_index[:-1]
		x1 = np.maximum(ip_boxes[to_find,0],ip_boxes[index,0])
		x2 = np.maximum(np.minimum(ip_boxes[to_find,2],ip_boxes[index,2]),x1)
		y1 = np.maximum(ip_boxes[to_find,1],ip_boxes[index,1])
		y2 = np.maximum(np.minimum(ip_boxes[to_find,3],ip_boxes[index,3]),y1)
		w = x2 - x1  
		h = y2 - y1
		intersection_area = (w*h).reshape(-1,1)
		total_area = (ip_boxes[to_find,2] - ip_boxes[to_find,0]).reshape(-1,1)*(ip_boxes[to_find,3] - ip_boxes[to_find,1]).reshape(-1,1) + (ip_boxes[index,2] - ip_boxes[index,0]).reshape(-1,1)*(ip_boxes[index,3] - ip_boxes[index,1]).reshape(-1,1) - intersection_area
		#overlap = intersection_area/(area[to_find,:]+1e-5)
		overlap = intersection_area/total_area
		keep[index]=list(to_find[np.where(overlap >ov_threshold)[0]])
		keep[index].append(index)
		
		sorted_y_index= np.delete(sorted_y_index,np.concatenate([[len(sorted_y_index)-1],np.where(overlap > ov_threshold)[0]]))

	return keep

# if __name__ == '__main__':
# 	a = np.load('/home/shashank/Documents/CSE-252C/chainer_ref/hyperface/ip1.npy')
# 	x1 = a[:,0].reshape(-1,1)
# 	y1 = a[:,1].reshape(-1,1)
# 	x2 = a[:,2].reshape(-1,1)
# 	y2 = a[:,3].reshape(-1,1)
# 	x2 = x2.reshape(-1,1) + x1
# 	y2 = y2.reshape(-1,1) + y1
# 	a = np.concatenate([x1,y1,x2,y2],axis=1)
# 	fast_nms(a,0.2)


			
