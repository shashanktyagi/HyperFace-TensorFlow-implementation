import cv2
from pdb import set_trace as brk
def vis_results(img,res_dict):
		
	for i in range(len(res_dict['location'])):
		cv2.rectangle(img,(int(res_dict['location'][i][0]),int(res_dict['location'][i][1])),(int(res_dict['location'][i][2]),
			int(res_dict['location'][i][3])),(0,255,0),2)
		for j in range(res_dict['landmarks'][i].shape[0]):
			print (int(res_dict['landmarks'][i][j,0]),int(res_dict['landmarks'][i][j,1]))
			cv2.circle(img,(int(res_dict['landmarks'][i][j,0]),int(res_dict['landmarks'][i][j,1])), 1, (0,0,255), 2)
		#Write M for male, F for Female
		center_x = int(int(res_dict['location'][i][0]) + (int(res_dict['location'][i][2]) - int(res_dict['location'][i][0])))
		center_y = int(int(res_dict['location'][i][1]) + (int(res_dict['location'][i][3]) - int(res_dict['location'][i][1])))
				
		if res_dict['gender'][i][0] < 0.5:
			#cv2.putText(img,'M',(center_x,center_y), cv2.FONT_HERSHEY_SIMPLEX, 1,(153,0,76),2,cv2.LINE_AA)
			cv2.putText(img,'M',(center_x,center_y), cv2.FONT_HERSHEY_SIMPLEX, 1,(127,0,255),2,cv2.LINE_AA)
		elif res_dict['gender'][i][0] >= 0.5:
			cv2.putText(img,'F',(center_x,center_y), cv2.FONT_HERSHEY_SIMPLEX, 1,(127,0,255),2,cv2.LINE_AA)
		
	cv2.imshow('result',img/255.0)
	cv2.waitKey(0)