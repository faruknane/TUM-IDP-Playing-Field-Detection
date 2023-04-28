import math
import cv2
import numpy as np

def GaussianBlur(edges2, kernel, threshold):
    edges2 = cv2.GaussianBlur(edges2, kernel, 0)
    edges2 = edges2 >= threshold
    edges2 = (edges2*255).astype(np.uint8)
    return edges2

img = cv2.imread("deneme.png", cv2.IMREAD_ANYCOLOR)


# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("debug/gray.png", gray)

# gray = cv2.bilateralFilter(gray, 3, 75, 75)

# Apply Canny edge detection
edges = cv2.Canny(gray, 255, 35)
cv2.imwrite("debug/edges.png", edges)

# Apply Hough line detection
lines = cv2.HoughLinesP(edges, 1, math.pi/180, 200, minLineLength=80, maxLineGap=50)

resimg = np.copy(img)

pass_it = True

# Draw the lines as red on the original image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(resimg, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.circle(resimg, (x1, y1), 2, (0, 255, 0), 3)
    cv2.circle(resimg, (x2, y2), 2, (0, 255, 0), 3)
    
    if not pass_it:
        cimg = np.copy(img)
        cv2.line(cimg, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # resize cimg to 1080p
        cimg = cv2.resize(cimg, (1920, 1080))

        #show image
        cv2.imshow("image", cimg)
        
        # read key if esc exit
        key = cv2.waitKey(0)
        if key == 27:
            pass_it = True
            

# Save the resulting image
cv2.imwrite(f"debug/hough_result.png", resimg)