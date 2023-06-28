# Import cv2 module
import cv2
import video_process
import numpy as np


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

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 255, 35)
    # cv2.imwrite("debug/edges.png", edges)

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

def GetLines(edges2):
    lsd = cv2.createLineSegmentDetector(
        refine=cv2.LSD_REFINE_STD, # Use advanced refinement
        scale=0.8, # Use 80% of the original image scale
        sigma_scale=0.6, # Use 0.6 as sigma for Gaussian filter
        quant=35.0, # Set quantization error to 2.0
        ang_th=22.5, # Set gradient angle tolerance to 22.5 degrees
        log_eps=100.0, # Set detection threshold to -log10(NFA) > 1.0
        density_th=0.1, # Set minimal density of region points to 0.6
        n_bins=1024*4 # Set number of bins to 1024
    )

    # lsd = cv2.createLineSegmentDetector(
    #     refine=cv2.LSD_REFINE_ADV, # Use advanced refinement
    #     scale=0.8, # Use 80% of the original image scale
    #     sigma_scale=0.9, # Use 0.6 as sigma for Gaussian filter
    #     quant=15.0, # Set quantization error to 2.0
    #     ang_th=22.5, # Set gradient angle tolerance to 22.5 degrees
    #     log_eps=10.0, # Set detection threshold to -log10(NFA) > 1.0
    #     density_th=0.1, # Set minimal density of region points to 0.6
    #     n_bins=1024*4 # Set number of bins to 1024
    # )

    lines = lsd.detect(edges2)[0]
    return lines


if __name__ == "__main__":
    img = video_process.TakeFrameFromVideo("videos/DJI_0176.MP4", 0)
    edges2 = Preprocess(img)
    lines = GetLines(edges2)

    print(len(lines))
    
    pass_it = True
    for line in lines:
            x1, y1, x2, y2 = line[0]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.circle(img, (x1, y1), 2, (0, 255, 0), 3)
            cv2.circle(img, (x2, y2), 2, (0, 255, 0), 3)
            
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

    cv2.imwrite("debug/line_det_result.png", img)