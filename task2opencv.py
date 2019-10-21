import numpy as np 
import cv2

LGray=np.array([0,0,0])
UGray=np.array([130,130,130])    

Red=(0,0,255)

LRed=np.array([0,0,50])
URed=np.array([50,50,255])    

Green=(0,255,0)

set_images=str(input("Enter the input path :"))
dimx=750
dimy=400
def dist(a,b):
	return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5

def sort_by_X_augmented(a,b):
	aug=[]
	for i in range(len(a)):
		aug.append([a[i],b[i]])
	aug=sorted(aug)
	return aug

def parse_augmented(a):
	b1=[]
	b2=[]
	for i in range(len(a)):
		b1.append(a[i][0])
		b2.append(a[i][1])
	return [b1,b2]

def detect_color(frame,L,U,line_colour,cent=[],rad=[]):
	mask_red=cv2.inRange(frame,L,U)

	refined_red=mask_red.copy()

	kernel = np.ones((2,2), np.uint8) 
	if(len(cent)==0):
		refined_red = cv2.erode(refined_red, kernel, iterations=1) 
		refined_red = cv2.dilate(refined_red, kernel, iterations=12) 
	else:
		refined_red = cv2.erode(refined_red, kernel, iterations=1) 
		refined_red = cv2.dilate(refined_red, kernel, iterations=1)

	valid=[]
	contours, hire= cv2.findContours(refined_red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	areas=[cv2.contourArea(i) for i in contours]
	if(len(areas)!=0):
		area_min=min(areas)
		area_max=max(areas)
	else:
		return 0,[],frame,[]

	if(area_max-area_min>=area_min/2 or cent!=[]):
		for j in range(len(areas)):
			if areas[j]>(area_min+area_max)/2:
				valid.append(contours[j])
	else:
		valid=contours
	contours_poly = [None]*len(valid)
	centers = [None]*len(valid)
	radius = [None]*len(valid)
	for i, c in enumerate(valid):
	    contours_poly[i] = cv2.approxPolyDP(c, 3, True)
	    centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
	if(len(cent)==0):
		for i in range(len(valid)):
			cv2.circle(frame, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), line_colour, 2)
	else:
		aug=sort_by_X_augmented(centers,radius)
		sep=parse_augmented(aug)
		for i in range(min(len(cent),len(sep[0]))):
			if(dist(cent[i],sep[0][i])>=rad[i] and sep[1][i]>1.4*min(rad)):
				cv2.circle(frame, (int(sep[0][i][0]), int(sep[0][i][1])), int(sep[1][i]), line_colour, 2)
	return len(valid),centers,frame,radius

frame=cv2.imread(set_images)
frame=cv2.resize(frame, (dimx,dimy))

a,b,c,d=detect_color(frame,LRed,URed,Green)
print("The number of Red Arrow Planes are: "+ str(a))
print("The list of coordinates of planes are: " + str(b))
l=sort_by_X_augmented(b,d)
l=parse_augmented(l)
_,_,final,_=detect_color(c,LGray,UGray,Red,l[0],l[1])

cv2.imshow("Tracker", final)

cv2.waitKey(0)
cv2.destroyAllWindows()