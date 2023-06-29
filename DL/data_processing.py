import cv2
import os
import numpy as np


def ProjectPoint(h, x, y):
    # h is homography matrix
    # x, y is the point to be projected
    vector = np.array([x, y, 1])
    vector = h @ vector
    return vector[:2] / vector[2]


def BackprojectPoint(h, x, y):
    # h is homography matrix
    # x, y is the point to be projected
    vector = np.array([x, y, 1])
    vector = np.linalg.inv(h) @ vector
    return vector[:2] / vector[2]

def GenerateSourceKeyPoints(image):
    h = image.shape[0]
    w = image.shape[1]

    # 4 corners
    top_left = np.array([0, h*0.6], dtype=np.float32)
    top_right = np.array([w, h*0.6], dtype=np.float32)
    bottom_right = np.array([w, h], dtype=np.float32)
    bottom_left = np.array([0, h], dtype=np.float32)

    return [top_left, top_right, bottom_right, bottom_left]


def GenerateTargetKeyPoints(homography, image):

    points = GenerateSourceKeyPoints(image)

    targets = []
    # project points
    for point in points:
        res = ProjectPoint(homography, point[0], point[1])
        targets.append(res)

    return targets


def ApplyHorizontalFlip(h, sw, tw):

    h1 = h[0, 0]
    h2 = h[0, 1]
    h3 = h[0, 2]
    h4 = h[1, 0]
    h5 = h[1, 1]
    h6 = h[1, 2]
    h7 = h[2, 0]
    h8 = h[2, 1]
    h9 = h[2, 2]

    a1 = h1 - tw*h7
    a2 = tw * h8 - h2
    a3 = tw * h9 - h3 - sw * a1

    a4 = - h4
    a5 = h5
    a6 = h6 - sw * a4

    a7 = - h7
    a8 = h8
    a9 = h9 - sw * a7

    new_homography = np.array([
        [a1, a2, a3],
        [a4, a5, a6],
        [a7, a8, a9],
    ], dtype=np.float32)

    return new_homography

if __name__ == "__main__":

    homography = np.array([
        [-7.0777386e-02, -4.8012934e-01, -2.7362627e+01],
        [-6.2416487e-02, -8.7684898e-01, 3.2449496e+02],
        [1.9209566e-04/5, -8.1501194e-03/5, 1.0000000e+00/5]
    ]  )

    # res = ProjectPoint(homography, 1267, 573)
    # print(res)

    # res = BackprojectPoint(homography, res[0], res[1])
    # print(res)

    # read img
    img = cv2.imread("soccer_data/train_val/159.jpg")


    # horizontal flip img
    img = cv2.flip(img, 1)
    # horizontal flip homography
    homography = ApplyHorizontalFlip(homography, 1280, 115*5)
    
    points = GenerateSourceKeyPoints(img)

    # 115, 75
    img = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))

    # draw points
    for point in points:
        point = ProjectPoint(homography, point[0], point[1])
        cv2.circle(img, (int(point[0]), int(point[1])), 10, (0, 255, 0), -1)
    
    # save
    cv2.imwrite("99_warp2.jpg", img)


