# Import cv2 module
import cv2
import video_process

# Read the image
# img = video_process.TakeFrameFromVideo("soccer_short.MP4", 1)
img = cv2.imread("deneme.png", cv2.IMREAD_ANYCOLOR)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Create default parametrization LSD
lsd = cv2.createLineSegmentDetector(0)

# Detect lines in the image
lines = lsd.detect(img)[0]

# Draw detected lines in the image
drawn_img = lsd.drawSegments(img, lines)

# drawn_img = cv2.resize(drawn_img, (0,0), fx=0.5, fy=0.5)
cv2.imwrite(f"debug/lsd.png", drawn_img)