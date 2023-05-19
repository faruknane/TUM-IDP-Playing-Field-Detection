import math
import cv2
import lines_process
import numpy as np
import video_process
import line_detection
import random
import time

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
    # cv2.imwrite("debug/gray.png", gray)

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
    
    resimg = np.zeros((edges2.shape[0], edges2.shape[1], 3), np.uint8)
    resimg[:,:,0] = edges2
    resimg[:,:,1] = edges2
    resimg[:,:,2] = edges2
    DrawLines(resimg, lines)
    cv2.imwrite(f"debug/deneme.png", resimg)
    
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

def GetRectangularAreas(lines):

    paired_lines = lines_process.PairLines(lines, delta_e = 0.0872665, max_dist = 50)
    paired_pairs = lines_process.PairPairs(paired_lines, delta_e = 0.0872665, max_dist = 50, min_dist_same_lines=250)

    return paired_pairs

def FindMaxAreaPairedPair(paired_pairs):
    if len(paired_pairs) == 0:
        return None
    
    return max(paired_pairs, key=lines_process.PairedPairArea)

def DrawRectangle(resimg, rectangle, line_color = (0, 255, 0), line_thickness = 2, corner_color = (0, 165, 255), corner_thickness = 5):
    # [topleft, topright, bottomright, bottomleft] = lines_process.PairedPairToRectangle(paired_pair_max_area)
    [topleft, topright, bottomright, bottomleft] = rectangle

    # make corner points integer pairs
    topleft = (int(topleft[0]), int(topleft[1]))
    topright = (int(topright[0]), int(topright[1]))
    bottomright = (int(bottomright[0]), int(bottomright[1]))
    bottomleft = (int(bottomleft[0]), int(bottomleft[1]))

    # Draw the lines between the corners in green
    cv2.line(resimg, (topleft[0], topleft[1]), (topright[0], topright[1]), line_color, line_thickness)
    cv2.line(resimg, (topright[0], topright[1]), (bottomright[0], bottomright[1]), line_color, line_thickness)
    cv2.line(resimg, (bottomright[0], bottomright[1]), (bottomleft[0], bottomleft[1]), line_color, line_thickness)  
    cv2.line(resimg, (bottomleft[0], bottomleft[1]), (topleft[0], topleft[1]), line_color, line_thickness)

    # Draw the corner points in Orange
    cv2.circle(resimg, (topleft[0], topleft[1]), 2, corner_color, corner_thickness)
    cv2.circle(resimg, (topright[0], topright[1]), 2, corner_color, corner_thickness)
    cv2.circle(resimg, (bottomright[0], bottomright[1]), 2, corner_color, corner_thickness)
    cv2.circle(resimg, (bottomleft[0], bottomleft[1]), 2, corner_color, corner_thickness)
    

def DrawPairedPairs(resimg, paired_pairs, color = (0, 255, 0)):
    # Draw the lines of pairs with a green color on the original image
    for paired_pair in paired_pairs:
        for pairs in paired_pair:
            for line in pairs:
                x1, y1, x2, y2 = line[0]
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                cv2.line(resimg, (x1, y1), (x2, y2), color, 3)
                cv2.circle(resimg, (x1, y1), 2, (0, 255, 0), 3)
                cv2.circle(resimg, (x2, y2), 2, (0, 255, 0), 3)

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
    
def GetPossibleEdgeArea(frame, frame_max_area_pairedpair, thickness = 100):

    # extract 4 lines from pairedpair "frame_max_area_pairedpair"
    # do it as before above

    pair_i = frame_max_area_pairedpair[0]
    pair_j = frame_max_area_pairedpair[1]

    line_i1 = pair_i[0][0]
    line_i2 = pair_i[1][0]

    line_j1 = pair_j[0][0]
    line_j2 = pair_j[1][0]

    # create an image all black with same size as frame but only 1 channel
    new_img = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)

    # draw white lines on new_img
    cv2.line(new_img, (line_i1[0], line_i1[1]), (line_i1[2], line_i1[3]), (255, 255, 255), thickness)
    cv2.line(new_img, (line_i2[0], line_i2[1]), (line_i2[2], line_i2[3]), (255, 255, 255), thickness)
    cv2.line(new_img, (line_j1[0], line_j1[1]), (line_j1[2], line_j1[3]), (255, 255, 255), thickness)
    cv2.line(new_img, (line_j2[0], line_j2[1]), (line_j2[2], line_j2[3]), (255, 255, 255), thickness)

    # # imshow resize image to its %20
    # cv2.imshow("new_img", cv2.resize(new_img, (0, 0), fx=0.2, fy=0.2))
    # cv2.waitKey(0)

    return frame_max_area_pairedpair, new_img


start_time = None

def StartMeasuring():
    global start_time
    start_time = time.time()

def PrintElapsedTime(st):
    global start_time
    print(st + f" => Elapsed time: {time.time() - start_time}")


def DrawTemplateImage(resimg, rectangle, FieldTemplate):
    
     # [topleft, topright, bottomright, bottomleft] = lines_process.PairedPairToRectangle(paired_pair_max_area)
    [topleft, topright, bottomright, bottomleft] = rectangle

    # make corner points integer pairs
    topleft = (int(topleft[0]), int(topleft[1]))
    topright = (int(topright[0]), int(topright[1]))
    bottomright = (int(bottomright[0]), int(bottomright[1]))
    bottomleft = (int(bottomleft[0]), int(bottomleft[1]))

    # draw field template image using the rectangle parameter
    # first, define 4 points of the field template image
    # second, ward perspective transformation
    # third, combine the result with the original image

    # first
    pts1 = np.float32([[0, 0], [FieldTemplate.shape[1], 0], [FieldTemplate.shape[1], FieldTemplate.shape[0]], [0, FieldTemplate.shape[0]]])
    pts2 = np.float32([topleft, topright, bottomright, bottomleft])

    # second
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(FieldTemplate, M, (resimg.shape[1], resimg.shape[0]))

    # create a white image like field template image
    accepted_area = dst[:,:,3] / 255

    # third
    resimg[:,:] = (accepted_area[:,:,np.newaxis] * dst[:,:, 0:3]) +  ((1-accepted_area)[:,:,np.newaxis] * resimg[:,:]) 
        

if __name__ == "__main__":
    
    prev_max_area_pairedpair = None
    prev_possible_edge_area = None
    FieldTemplate = cv2.imread("videos/field_template4.png", cv2.IMREAD_UNCHANGED)

    for counter_i in range(0, 100):
            
        # frame = video_process.TakeFrameFromVideo("soccer_short.MP4", counter_i)
        # frame = video_process.TakeFrameFromVideo("videos/3.mp4", counter_i)
        frame = video_process.TakeFrameFromVideo("videos/DJI_0176.MP4", counter_i)
        
        StartMeasuring()
        frame_edges = Preprocess(frame)
        PrintElapsedTime("preprocessing time")

        if prev_possible_edge_area is not None:
            frame_edges = cv2.bitwise_and(frame_edges, prev_possible_edge_area)
            # cv2.imwrite(f"debug/possible_edge_area_{counter_i}.png", prev_possible_edge_area)
            # cv2.imwrite(f"debug/possible_edge_area_and_frame_edges_{counter_i}.png", frame_edges)

        frame_lines = ExtractDefaultHoughLines(frame_edges)
        
        resimg = np.copy(frame)
        DrawLines(resimg, frame_lines)
        cv2.imwrite(f"debug/hough_raw_result.png", resimg)

        frame_paired_pairs = GetRectangularAreas(frame_lines)
        frame_max_area_pairedpair = FindMaxAreaPairedPair(frame_paired_pairs)

        if frame_max_area_pairedpair is not None:
            max_area_rectangle = lines_process.PairedPairToRectangle(frame_max_area_pairedpair)
        PrintElapsedTime("lines & rectangles processing time")
        print("rectangle count: ", len(frame_paired_pairs))

        resimg = np.copy(frame)
        if frame_max_area_pairedpair is not None:
            DrawTemplateImage(resimg, max_area_rectangle, FieldTemplate)
        DrawLines(resimg, frame_lines)
        # DrawPairedPairs(resimg, frame_paired_pairs, color = (128,128,128))
        if frame_max_area_pairedpair is not None:
            DrawRectangle(resimg, max_area_rectangle, line_color = (0, 255, 0), corner_color = (0, 165, 255))
        cv2.imwrite(f"debug/video/hough_result_{counter_i}.png", resimg)
        
        
        if frame_max_area_pairedpair is None:
            print(f"can't find the rectangle in frame{counter_i}!")
        else:
            if prev_max_area_pairedpair is None:
                prev_max_area_pairedpair, prev_possible_edge_area = GetPossibleEdgeArea(frame, frame_max_area_pairedpair, thickness=50)
            else:
                area1 = lines_process.PairedPairArea(prev_max_area_pairedpair)
                area2 = lines_process.PairedPairArea(frame_max_area_pairedpair)

                if abs(area1 - area2) > 0.01 * area1:
                    print(f"can't find the rectangle in frame{counter_i}!")
                else:
                    prev_max_area_pairedpair, prev_possible_edge_area = GetPossibleEdgeArea(frame, frame_max_area_pairedpair)

        PrintElapsedTime("total time")
       