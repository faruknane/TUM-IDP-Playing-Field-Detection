import numpy as np
import cv2

def direct_infer(points):

    # left_top = points[0] 
    # right_top = points[1]
    # right_bottom = points[2]
    # left_bottom = points[3]
    # center = points[4]
    # top_center = points[5]
    # bottom_center = points[6]

    dep = [
        [[-1, 4, 2], [-1, 5, 1]], # 0
        [[3, 4, -1], [0, 5, -1]], # 1
        [[0, 4, -1], [3, 6, -1]], # 2
        [[-1, 4, 1], [-1, 6, 2]], # 3
        [[0, -1, 2], [3, -1, 1], [5, -1, 6]], # 4
        [[0, -1, 1], [-1, 4, 6]], # 5
        [[3, -1, 2], [5, 4, -1]], # 6
    ]

    for mypoint in range(len(dep)):

        if points[mypoint] is None:
            rules = dep[mypoint]

            myvalues = []
            for rule in rules:
                a, b, c = rule
                if a == -1 and points[b] is not None and points[c] is not None:
                    myvalues.append(2*points[b] - points[c])
                elif b == -1 and points[a] is not None and points[c] is not None:
                    myvalues.append((points[a] + points[c])/2)
                elif c == -1 and points[a] is not None and points[b] is not None:
                    myvalues.append(2*points[b] - points[a])

            if len(myvalues) > 0:
                points[mypoint] = np.mean(myvalues, axis=0)

    return points


def draw_tennis_field(image, pred_real_coords):

    # np array
    pred_real_coords = np.array(pred_real_coords, dtype=np.float32)

    left_top = pred_real_coords[0]
    right_top = pred_real_coords[1]
    right_bottom = pred_real_coords[2]
    left_bottom = pred_real_coords[3]

    # let's define 4 boundary lines
    top_line = np.array([left_top, right_top], dtype=np.float32)
    right_line = np.array([right_top, right_bottom], dtype=np.float32)
    bottom_line = np.array([right_bottom, left_bottom], dtype=np.float32)
    left_line = np.array([left_bottom, left_top], dtype=np.float32)

    # let's define inner lines of a tennis field
    top_center = (left_top + right_top) / 2
    bottom_center = (left_bottom + right_bottom) / 2
    mid_line = np.array([top_center, bottom_center], dtype=np.float32)

    right_p1 = (right_top * 7 + right_bottom) / 8
    right_p2 = (right_top + right_bottom * 7) / 8

    left_p1 = (left_top * 7 + left_bottom) / 8
    left_p2 = (left_top + left_bottom * 7) / 8

    right_top_p2 = (left_p1 * 18 + right_p1 * 60) / 78
    right_bottom_p2 = (left_p2 * 18 + right_p2 * 60) / 78

    left_top_p2 = (left_p1 * 60 + right_p1 * 18) / 78
    left_bottom_p2 = (left_p2 * 60 + right_p2 * 18) / 78

    c1 = (right_top_p2 + right_bottom_p2) / 2
    c2 = (left_top_p2 + left_bottom_p2) / 2

    lines = []

    lines.append(top_line)
    lines.append(right_line)
    lines.append(bottom_line)
    lines.append(left_line)
    
    lines.append(mid_line)
    
    lines.append(np.array([left_p1, right_p1], dtype=np.float32))
    lines.append(np.array([left_p2, right_p2], dtype=np.float32))

    lines.append(np.array([right_top_p2, right_bottom_p2], dtype=np.float32))
    lines.append(np.array([left_top_p2, left_bottom_p2], dtype=np.float32))
    
    lines.append(np.array([c1, c2], dtype=np.float32))

    # draw lines
    for line in lines:
        cv2.line(image, (int(line[0,0]), int(line[0,1])), (int(line[1,0]), int(line[1,1])), (0, 255, 0), 2)

    return image

def get_playing_field_lines(class_index, M):
    global field_types
    field_type = field_types[class_index]

    lines = []

    if field_type == "tennis":
        lines.append(np.array([[0, 0], [1, 0]], dtype=np.float32))
        lines.append(np.array([[1, 0], [1, 1]], dtype=np.float32))
        lines.append(np.array([[1, 1], [0, 1]], dtype=np.float32))
        lines.append(np.array([[0, 1], [0, 0]], dtype=np.float32))

        lines.append(np.array([[0.5, 0], [0.5, 1]], dtype=np.float32))

        lines.append(np.array([[0, 1/8], [1, 1/8]], dtype=np.float32))
        lines.append(np.array([[0, 7/8], [1, 7/8]], dtype=np.float32))

        lines.append(np.array([[18/78, 1/8], [18/78, 7/8]], dtype=np.float32))
        lines.append(np.array([[60/78, 1/8], [60/78, 7/8]], dtype=np.float32))

        lines.append(np.array([[18/78, 1/2], [60/78, 1/2]], dtype=np.float32))
    elif field_type == "frisbee":
        lines.append(np.array([[0, 0], [1, 0]], dtype=np.float32))
        lines.append(np.array([[1, 0], [1, 1]], dtype=np.float32))
        lines.append(np.array([[1, 1], [0, 1]], dtype=np.float32))
        lines.append(np.array([[0, 1], [0, 0]], dtype=np.float32))

        lines.append(np.array([[18/100, 0], [18/100, 1]], dtype=np.float32))
        lines.append(np.array([[82/100, 0], [82/100, 1]], dtype=np.float32))


    elif field_type == "football":
        lines.append(np.array([[0, 0], [1, 0]], dtype=np.float32))
        lines.append(np.array([[1, 0], [1, 1]], dtype=np.float32))
        lines.append(np.array([[1, 1], [0, 1]], dtype=np.float32))
        lines.append(np.array([[0, 1], [0, 0]], dtype=np.float32))

        lines.append(np.array([[0.5, 0], [0.5, 1]], dtype=np.float32))
        
        lines.append(np.array([[1, 15.5/75], [102/120, 15.5/75]], dtype=np.float32))
        lines.append(np.array([[102/120, 15.5/75], [102/120, 59.5/75]], dtype=np.float32))
        lines.append(np.array([[102/120, 59.5/75], [1, 59.5/75]], dtype=np.float32))

        
        lines.append(np.array([[0, 15.5/75], [18/120, 15.5/75]], dtype=np.float32))
        lines.append(np.array([[18/120, 15.5/75], [18/120, 59.5/75]], dtype=np.float32))
        lines.append(np.array([[18/120, 59.5/75], [0, 59.5/75]], dtype=np.float32))



    target_lines = []

    for line in lines:
        p1 = np.matmul(M, np.array([line[0,0], line[0,1], 1])) 
        p2 = np.matmul(M, np.array([line[1,0], line[1,1], 1])) 

        # normalize p and get x y
        if p1.shape[0] == 3:
            p1 = p1/p1[2]
        if p2.shape[0] == 3:
            p2 = p2/p2[2]

        target_lines.append(np.array([[p1[0], p1[1]], [p2[0], p2[1]]], dtype=np.float32))
  
    return target_lines


def SetFieldTypes(field_types_):
    global field_types
    field_types = field_types_

def GetFieldPoints():
    points = []
    points.append([0, 0])
    points.append([1, 0])
    points.append([1, 1])
    points.append([0, 1])
    points.append([0.5, 0.5])
    points.append([0.5, 0])
    points.append([0.5, 1])

    for i in range(len(points)):
        points[i] = np.array(points[i], dtype=np.float32)

    return points

def FixLinearity(new_source_points, new_target_points, target_points, class_index):
    for i in range(new_source_points.shape[0]):
        for j in range(i+1, new_source_points.shape[0]):
            for k in range(j+1, new_source_points.shape[0]):
                p1 = new_source_points[i]
                p2 = new_source_points[j]
                p3 = new_source_points[k]

                if np.abs(np.cross(p2-p1, p3-p1)) < 0.0001:
                    new_target_points = infer_field3(target_points, class_index)
                    new_source_points = GetFieldPoints()
                                    
                    new_source_points = np.array(new_source_points, dtype=np.float32)
                    new_target_points = np.array(new_target_points, dtype=np.float32)
                    return new_source_points, new_target_points

    return new_source_points, new_target_points
    
def infer_field(target_points, class_index):
    global field_types
    field_type = field_types[class_index]

    source_points = GetFieldPoints()
    if field_type == "frisbee":
        source_points[5] = None
        source_points[6] = None

    new_source_points = []
    new_target_points = []

    for i in range(len(target_points)):
        if target_points[i] is not None and source_points[i] is not None:
            new_source_points.append(source_points[i])
            new_target_points.append(target_points[i])

    # apply affine transformation cv2
    new_source_points = np.array(new_source_points, dtype=np.float32)
    new_target_points = np.array(new_target_points, dtype=np.float32)

    if new_source_points.shape[0] < 3:
        return None, None
    elif new_source_points.shape[0] == 3:
        # let us find if the points are linear in new_source_points
        p1 = new_source_points[0]
        p2 = new_source_points[1]
        p3 = new_source_points[2]

        if np.abs(np.cross(p2-p1, p3-p1)) < 0.0001:
            return None, None
        
        M = cv2.getAffineTransform(new_source_points, new_target_points)
    else:

        # detect if any 3 of new source_points are linear
        if new_source_points.shape[0] <= 4:
            new_source_points, new_target_points = FixLinearity(new_source_points, new_target_points, target_points, class_index)
                        
        # find homography by applying RANSAC
        M, mask = cv2.findHomography(new_source_points, new_target_points, cv2.RANSAC, 1.0)

    source_points = GetFieldPoints()

    # now apply M to source_points and get target_points
    target_points = []
    for i in range(len(source_points)):
        p = np.matmul(M, np.array([source_points[i][0], source_points[i][1], 1])) 
        
        # normalize p and get x y
        if p.shape[0] == 3:
            p = p/p[2]
        target_points.append([p[0], p[1]])
        
    return np.array(target_points, dtype=np.float32), M
   


def infer_field3(points, class_index=None):
    for i in range(7):
        points = direct_infer(points)

    for point in points:
        if point is None:
            return None
    return np.array(points, dtype=np.float32)


def infer_field2(points):

    left_top = points[0] 
    right_top = points[1]
    right_bottom = points[2]
    left_bottom = points[3]
    center = points[4]

    if left_top is None and right_bottom is None:
        return None

    if left_bottom is None and right_top is None:
        return None

    if center is None and left_top is not None and right_bottom is not None and left_bottom is not None and right_top is not None:
        center = (left_top + right_bottom + left_bottom + right_top)/4
    elif center is None and left_top is not None and right_bottom is not None:
        center = (left_top + right_bottom)/2
    elif center is None and left_bottom is not None and right_top is not None:
        center = (left_bottom + right_top)/2

    if left_top is None:
        left_top = 2*center - right_bottom

    if right_bottom is None:
        right_bottom = 2*center - left_top

    if right_top is None:
        right_top = 2*center - left_bottom

    if left_bottom is None:
        left_bottom = 2*center - right_top

    points = np.array([left_top, right_top, right_bottom,
                      left_bottom, center], dtype=np.float32)

    return points
