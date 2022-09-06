import cv2 
import time
from predict2 import predict

cap = cv2.VideoCapture('vids/mach_sai_3.mp4')

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
	#reading the image  
	_,image = cap.read()

	edged = cv2.Canny(image, 10, 250) 
	
	#applying closing function  
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) 
	closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel) 
	
	#finding_contours  
	(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
	
	for c in cnts: 
		x,y,w,h = cv2.boundingRect(c) 
		peri = cv2.arcLength(c, True) 
		approx = cv2.approxPolyDP(c, 0.02 * peri, True) 
		# print(approx)
		# cv2.drawContours(image, [approx], -1, (0, 255, 0), 2) 
		if w>500 and h>500: 

			new_img=image[y:y+h,x:x+w] 

	resize_img = cv2.resize(new_img, (1050, 720), interpolation=cv2.INTER_LINEAR)
	pre = predict(resize_img)

	new_frame_time = time.time()
	fps = 1/(new_frame_time-prev_frame_time)
	prev_frame_time = new_frame_time
	fps = int(fps)
	fps = str(fps)
	cv2.putText(pre, f"{fps} FPS", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 0), 3)
	cv2.imshow("Output", pre) 
	# cv2.imwrite('imgs/pcb_detection_md.png', resize_img) 

	if cv2.waitKey(20) & 0xFF == 27:
		break

cap.release()

cv2.destroyAllWindows()