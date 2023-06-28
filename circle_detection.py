import math
import cv2
import lines_process
import numpy as np
import video_process
import line_detection


def GaussianBlur(edges2, kernel, threshold):
    edges2 = cv2.GaussianBlur(edges2, kernel, 0)
    edges2 = edges2 >= threshold
    edges2 = (edges2*255).astype(np.uint8)
    return edges2

def ApplyHorizontal(edges2, horizontal_size = 10, other_size = 1):
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(edges2, horizontalStructure)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, other_size))
    horizontal = cv2.dilate(horizontal, horizontalStructure)
    return horizontal

def ApplyVertical(edges2, vertical_size = 10, other_size = 1):
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(edges2, verticalStructure)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (other_size, vertical_size))
    vertical = cv2.dilate(vertical, verticalStructure)
    return vertical

def Preprocess(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("debug/gray.png", gray)

    # gray = cv2.bilateralFilter(gray, 3, 75, 75)

    # ratio = 1 # works for 0.8

    # # increase width and height of the image by 1.2
    # gray = cv2.resize(gray, (0, 0), fx=ratio, fy=ratio)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 200, 75)

    # decrease width and height of the image by 1.2
    # edges = cv2.resize(edges, (0, 0), fx=1/ratio, fy=1/ratio)

    cv2.imwrite("debug/edges.png", edges)

    edges2 = edges

    edges2 = GaussianBlur(edges2, (9, 9), 50)
    edges2 = cv2.dilate(edges2, np.ones((3, 3), np.uint8), iterations=1)
    edges2 = GaussianBlur(edges2, (9, 9), 180)
    cv2.imwrite("debug/edges1.5.png", edges2)
    edges2 = cv2.erode(edges2, np.ones((3, 3), np.uint8), iterations=1)
    edges2 = GaussianBlur(edges2, (5, 5), 150)
    edges2 = GaussianBlur(edges2, (15, 15), 50)
    edges2 = GaussianBlur(edges2, (15, 15), 150)
    edges2 = GaussianBlur(edges2, (15, 15), 100)
    edges2 = GaussianBlur(edges2, (15, 15), 100)
    cv2.imwrite("debug/edges2.png", edges2)

    return edges2

if __name__ == "__main__":

    # read image cv2
    img = video_process.TakeFrameFromVideo("videos/3.MP4", 10)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = Preprocess(img)

    # save gray image
    cv2.imwrite("debug/gray.png", gray)

    print("hi")

    # resize 4k to 720p use fx and fy
    img = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    gray = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.1, 100, param1=1500, param2=100)

    # ensure at least some circles were found
    print(len(circles[0]))
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
        cv2.imwrite("debug/circle.png", img)