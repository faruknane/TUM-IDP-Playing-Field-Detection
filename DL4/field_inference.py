import numpy as np


def infer_field(points):

    left_top = points[0]
    right_top = points[1]
    right_bottom = points[2]
    left_bottom = points[3]
    center = points[4]

    if left_top == None and right_bottom == None:
        return {"points": None, "message": "Left top and right bottom points are not detected"}

    if left_bottom == None and right_top == None:
        return {"points": None, "message": "Left bottom and right top points are not detected"}

    if left_top == None:
        left_top = 2*center - right_bottom

    if right_bottom == None:
        right_bottom = 2*center - left_top

    if right_top == None:
        right_top = 2*center - left_bottom

    if left_bottom == None:
        left_bottom = 2*center - right_top

    points = np.array([left_top, right_top, right_bottom,
                      left_bottom, center], dtype=np.float32)

    return {"points": points, "message": "Field is inferred"}
