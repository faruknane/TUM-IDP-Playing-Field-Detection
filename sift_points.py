import cv2
import numpy as np

# Read the image named "deneme.png"
img1 = cv2.imread("debug/edges2.png")
img2 = cv2.imread("deneme.png")

img = img1 * 0.5 + img2 * 0.5
img = img.astype(np.uint8)

# img = img2

# save img 
cv2.imwrite("debug/mixture.png", img)

# Create a SIFT object
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp, des = sift.detectAndCompute(img, None)

# Convert the descriptors to float32 for k-means clustering
des = des.astype(np.float32)

# Define the number of clusters
k = 15

# Apply k-means clustering on the descriptors
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(des, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Assign a random color to each cluster
colors = np.random.randint(0, 255, (k, 3)).astype(np.float)

# Draw keypoints on the image with the color of their cluster
img_kp = img.copy()
for i in range(len(kp)):
    x, y = kp[i].pt
    x = int(x)
    y = int(y)
    c = label[i][0]
    color = tuple(colors[c])
    cv2.circle(img_kp, (x, y), 5, color, -1)

# Save the resulting image
cv2.imwrite("debug/result_kmeans.png", img_kp)

