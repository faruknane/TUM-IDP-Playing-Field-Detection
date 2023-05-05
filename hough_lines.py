import math
import cv2
import lines_process
import numpy as np
import video_process
import line_detection
import random

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

# def Preprocess(img):
#     # Convert the image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite("debug/gray.png", gray)

#     # gray = cv2.bilateralFilter(gray, 3, 75, 75)

#     # Apply Canny edge detection
#     edges = cv2.Canny(gray, 255, 35)
#     cv2.imwrite("debug/edges.png", edges)

#     edges2 = GaussianBlur(edges, (9, 9), 50)
#     edges2 = GaussianBlur(edges2, (9, 9), 80)
#     edges2 = GaussianBlur(edges2, (5, 5), 140)
#     edges2 = GaussianBlur(edges2, (5, 5), 180)
#     edges2 = GaussianBlur(edges2, (5, 5), 220)
#     edges2 = GaussianBlur(edges2, (5, 5), 220)
#     cv2.imwrite("debug/edges2.png", edges2)
    
#     cikar = ApplyVertical(ApplyHorizontal(edges2, 15), 15)

#     edges3 = np.where(edges2 > cikar, 255, 0).astype(np.uint8)

#     vertical_lines = ApplyVertical(edges3, 21, other_size=3)
#     horizontal_lines = ApplyHorizontal(edges3, 21, other_size=3)

#     edges3 = np.where(horizontal_lines >= 255 - vertical_lines, 255, 0).astype(np.uint8)
    
#     cv2.imwrite("debug/edges3.png", edges3)

#     return edges3

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

def ExtractDefaultHoughLines(edges2):
    # Apply Hough line detection
    lines = cv2.HoughLinesP(edges2, 1, math.pi/180, 200, minLineLength=80, maxLineGap=30)
    # lines2 = cv2.HoughLinesP(edges2, 1, math.pi/180, 200, minLineLength=80, maxLineGap=30)
    # lines = line_detection.GetLines(edges2)

    print("-------------")
    print(len(lines))

    lines = lines_process.FilterLines(edges2, lines, 11)
    # lines2 = lines_process.FilterLines(edges2, lines2, 11)
    print(len(lines))

    lines = lines_process.Process(lines, 2, 0.002)
    # lines2 = lines_process.Process(lines2, 2, 0.002)
    print(len(lines))

    lines = lines_process.Process(lines, 3, 0.003)
    # lines2 = lines_process.Process(lines2, 3, 0.003)
    print(len(lines))

    # # lines = np.vstack((lines, lines2))

    lines = lines_process.Process(lines, 6, 0.005)
    print(len(lines))

    lines = lines_process.Process(lines, 30, 0.020)
    lines = lines_process.Process(lines, 60, angle_threshold=0.020, angle_threshold2=0.0135)
    print(len(lines))
    print("-------------")

    lines = lines_process.FinalFilterLines(edges2, lines, dist=11, max_count=36)
    
    return lines

def DrawLines(resimg, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.line(resimg, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.circle(resimg, (x1, y1), 2, (0, 255, 0), 3)
        cv2.circle(resimg, (x2, y2), 2, (0, 255, 0), 3)
    

if __name__ == "__main__":


    frame = cv2.imread("deneme.png", cv2.IMREAD_UNCHANGED)

    frame_edges = Preprocess(frame)
    frame_lines = ExtractDefaultHoughLines(frame_edges)
    resimg = np.copy(frame)
    DrawLines(resimg, frame_lines)
    cv2.imwrite(f"debug/hough_result.png", resimg)

    
    