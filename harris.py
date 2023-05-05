import math
import cv2
import numpy as np

def GaussianBlur(edges2, kernel, threshold):
    edges2 = cv2.GaussianBlur(edges2, kernel, 0)
    edges2 = edges2 >= threshold
    edges2 = (edges2*255).astype(np.uint8)
    return edges2

# Read the image named "deneme.png"
img = cv2.imread("deneme.png")

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("debug/gray.png", gray)

# Apply Canny edge detection
edges = cv2.Canny(gray, 255, 35)
cv2.imwrite("debug/edges.png", edges)

edges2 = GaussianBlur(edges, (9, 9), 50)
edges2 = GaussianBlur(edges2, (9, 9), 80)
edges2 = GaussianBlur(edges2, (9, 9), 80)
edges2 = GaussianBlur(edges2, (9, 9), 80)
edges2 = GaussianBlur(edges2, (9, 9), 220)
edges2 = GaussianBlur(edges2, (9, 9), 220)

cv2.imwrite("debug/edges2.png", edges2)

# Apply Harris corner detection with a block size of 2 and a Sobel aperture of 3
dst = cv2.cornerHarris(edges2, 3, 3, 0.05)

# Dilate the result to mark the corners
dst = cv2.dilate(dst, None)

# Threshold the result to keep only the significant corners
img[dst > 0.10 * dst.max()] = [0, 0, 255]

cv2.imwrite("debug/harris_result.png", img)

