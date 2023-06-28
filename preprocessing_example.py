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

def DetectWhitePixels(old_white_horizontal, old_white_vertical):

    size1 = 25
    size2 = 25

    line_width = 7
    
    # create a kernel size of size1xsize2 where for x dimension, middle line_width pixels are 1 and upper and lower remaining pixels are -1
    kernel = np.zeros((size1, size2), np.float32) * -1
    kernel[math.floor(size1/2) - math.floor(line_width/2) : math.floor(size1/2) + math.floor(line_width/2), :] = 1    
    kernel /= size1*size2

    # create a kernel size of size2xsize1 where middle size2 pixels are 1 and upper and lower remaining pixels are -1
    kernel2 = np.zeros((size2, size1), np.float32) * -1
    kernel2[:, math.floor(size1/2) - math.floor(line_width/2) : math.floor(size1/2) + math.floor(line_width/2)] = 1
    kernel2 /= size1*size2

    # now apply those kernels as a convolution operation
    # this will give you the number of white pixels in the middle of the kernel
    # you can change the borderType to cv2.BORDER_REPLICATE if you don't want the borders to be 0
    # you can also change the borderType to cv2.BORDER_CONSTANT and set the borderValue to 1
    # if you want the borders to be 1
    white_horizontal = cv2.filter2D(old_white_horizontal, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    white_vertical = cv2.filter2D(old_white_vertical, -1, kernel2, borderType=cv2.BORDER_REPLICATE)

    # eliminate pixels < 0
    white_horizontal = np.where(white_horizontal < 0, 0, white_horizontal)
    white_vertical = np.where(white_vertical < 0, 0, white_vertical)

    # normalize white pixels in range 0-255
    white_horizontal = cv2.normalize(white_horizontal, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    white_vertical = cv2.normalize(white_vertical, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # save white pixels image
    cv2.imwrite("debug/white_horizontal.png", white_horizontal)
    cv2.imwrite("debug/white_vertical.png", white_vertical)

    # return white pixels
    return white_horizontal, white_vertical



def DetectDottedLines(old_white_horizontal, old_white_vertical):

    size1 = 55
    size2 = 55

    line_width = 7
    width = 27
    
    outer_place = 0
    inner_place = 1.99
    middle_place = 0
    # create a kernel size of size1xsize2 where for x dimension, middle line_width pixels are 1 and upper and lower remaining pixels are -1
    kernel = np.zeros((size1, size2), np.float32) * outer_place
    kernel[math.floor(size1/2) - math.floor(line_width/2) : math.floor(size1/2) + math.floor(line_width/2), 0:width] = inner_place  
    kernel[math.floor(size1/2) - math.floor(line_width/2) : math.floor(size1/2) + math.floor(line_width/2), size1-width:size1] = inner_place  
    kernel[math.floor(size1/2) + math.floor(line_width/2) : math.floor(size1/2) + 3*math.floor(line_width/2), 0:width] = -1   
    kernel[math.floor(size1/2) - 3*math.floor(line_width/2) : math.floor(size1/2) - math.floor(line_width/2), 0:width] = -1   
    kernel[math.floor(size1/2) + math.floor(line_width/2) : math.floor(size1/2) + 3*math.floor(line_width/2), size1-width:size1] = -1   
    kernel[math.floor(size1/2) - 3*math.floor(line_width/2) : math.floor(size1/2) - math.floor(line_width/2), size1-width:size1] = -1   
    
    kernel /= size1*size2

    # create a kernel size of size2xsize1 where middle size2 pixels are 1 and upper and lower remaining pixels are -1
    kernel2 = np.zeros((size2, size1), np.float32) * outer_place
    # kernel2[:, math.floor(size1/2) - math.floor(line_width/2) : math.floor(size1/2) + math.floor(line_width/2)] = inner_place
    # kernel2[math.floor(size1/2) - math.floor(black_width/2) : math.floor(size1/2) + math.floor(black_width/2), math.floor(size1/2) - math.floor(black_width/2) : math.floor(size1/2) + math.floor(black_width/2)] = middle_place
    # kernel2 /= size1*size2

    # draw kernels normalize them
    kernel_ = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    kernel2_ = cv2.normalize(kernel2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite("debug/kernel.png", kernel_)
    cv2.imwrite("debug/kernel2.png", kernel2_)
    
    # now apply those kernels as a convolution operation
    # this will give you the number of white pixels in the middle of the kernel
    # you can change the borderType to cv2.BORDER_REPLICATE if you don't want the borders to be 0
    # you can also change the borderType to cv2.BORDER_CONSTANT and set the borderValue to 1
    # if you want the borders to be 1
    white_horizontal = cv2.filter2D(old_white_horizontal, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    white_vertical = cv2.filter2D(old_white_vertical, -1, kernel2, borderType=cv2.BORDER_REPLICATE)

    # eliminate pixels < 0
    white_horizontal = np.where(white_horizontal < 0, 0, white_horizontal)
    white_vertical = np.where(white_vertical < 0, 0, white_vertical)

    # normalize white pixels in range 0-255
    white_horizontal = cv2.normalize(white_horizontal, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    white_vertical = cv2.normalize(white_vertical, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # save white pixels image
    cv2.imwrite("debug/white_horizontal.png", white_horizontal)
    cv2.imwrite("debug/white_vertical.png", white_vertical)

    # return white pixels
    return white_horizontal, white_vertical




if __name__ == "__main__":

    # read image cv2
    img = video_process.TakeFrameFromVideo("videos/DJI_0176.MP4", 0)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) 

    # save gray image
    cv2.imwrite("debug/gray.png", gray)

    old_white_horizontal, old_white_vertical = gray, gray
    
    for i in range(5):
        old_white_horizontal, old_white_vertical = DetectDottedLines(old_white_horizontal, old_white_vertical)
        # old_white_horizontal, old_white_vertical = DetectWhitePixels(old_white_horizontal, old_white_vertical)

    # read image cv2
    # img = cv2.imread("deneme2.png")

    # preprocess image
    # edges3 = Preprocess(img)

