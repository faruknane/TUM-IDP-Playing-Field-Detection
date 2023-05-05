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
    cv2.imwrite("debug/gray.png", gray)

    # gray = cv2.bilateralFilter(gray, 3, 75, 75)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 255, 35)
    cv2.imwrite("debug/edges.png", edges)

    edges2 = edges

    edges2 = GaussianBlur(edges2, (9, 9), 50)
    edges2 = GaussianBlur(edges2, (9, 9), 80)
    edges2 = cv2.dilate(edges2, np.ones((3, 3), np.uint8), iterations=1)
    edges2 = GaussianBlur(edges2, (5, 5), 140)
    edges2 = GaussianBlur(edges2, (5, 5), 180)
    edges2 = GaussianBlur(edges2, (5, 5), 220)
    edges2 = GaussianBlur(edges2, (5, 5), 220)
    cv2.imwrite("debug/edges2.png", edges2)
    
    cikar = ApplyVertical(ApplyHorizontal(edges2, 25), 25)

    edges3 = np.where(edges2 > cikar, 255, 0).astype(np.uint8)

    vertical_lines = ApplyVertical(edges3, 21, other_size=1)
    horizontal_lines = ApplyHorizontal(edges3, 21, other_size=1)

    edges3 = np.where(horizontal_lines >= 255 - vertical_lines, 255, 0).astype(np.uint8)
    
    cv2.imwrite("debug/edges3.png", edges3)

    return edges3

if __name__ == "__main__":

    # read image cv2
    img = video_process.TakeFrameFromVideo("Tennis_0069.mov", 93)

    # read image cv2
    # img = cv2.imread("deneme2.png")

    # preprocess image
    edges3 = Preprocess(img)

