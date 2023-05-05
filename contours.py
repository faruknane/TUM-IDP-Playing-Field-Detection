import cv2
import numpy as np

img = cv2.imread("debug/edges2.png", cv2.IMREAD_UNCHANGED)

#Find my contours
contours =cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]


#Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
pass_it = False
cntrRect = []
for i in contours:
        epsilon = 0.05*cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i,epsilon,True)
        if len(approx) == 4:
            cntrRect.append(approx)
            
            if not pass_it:
                cimg = np.copy(img)
                cimg = np.stack([cimg,cimg,cimg], axis=2)

                cv2.drawContours(cimg,cntrRect,-1,(0,255,0),2)
                # resize cimg to its 1/5
                cimg = cv2.resize(cimg, (int(cimg.shape[1]/5), int(cimg.shape[0]/5)))

                cv2.imshow('Roi Rect ONLY', cimg)
                
                # read key if esc exit
                key = cv2.waitKey(0)
                if key == 27:
                    pass_it = True
            




print(cntrRect)