import numpy as np
import cv2


def calculate_miou(pred_points, label, debug=False):
    # label: [N, 5, 2]
    # pred_points: list of [5, 2] numpy arrays or Nones
    
    imageW, imageH = 512, 512

    none_count = 0
    iou = 0
    valid_count = 0

    for batch_i in range(label.shape[0]):
        label_points = label[batch_i]
        pred_points_batch = pred_points[batch_i]
        
        if pred_points_batch is None: 
            none_count += 1
            continue

        label_points = label_points[0:4]
        pred_points_batch = pred_points_batch[0:4]

        min_val_w = min(np.min(label_points[:, 0]), np.min(pred_points_batch[:, 0]))
        max_val_w = max(np.max(label_points[:, 0]), np.max(pred_points_batch[:, 0]))

        min_val_h = min(np.min(label_points[:, 1]), np.min(pred_points_batch[:, 1]))
        max_val_h = max(np.max(label_points[:, 1]), np.max(pred_points_batch[:, 1]))
        
        label_points_w = (label_points[:, 0] - min_val_w) / (max_val_w - min_val_w) * imageW
        label_points_h = (label_points[:, 1] - min_val_h) / (max_val_h - min_val_h) * imageH

        pred_points_batch_w = (pred_points_batch[:, 0] - min_val_w) / (max_val_w - min_val_w) * imageW
        pred_points_batch_h = (pred_points_batch[:, 1] - min_val_h) / (max_val_h - min_val_h) * imageH

        label_points = np.stack([label_points_w, label_points_h], axis=1)
        pred_points_batch = np.stack([pred_points_batch_w, pred_points_batch_h], axis=1)

        label_mask = np.zeros((imageW, imageH), dtype=np.uint8)
        label_points = label_points.reshape((-1, 1, 2))
        label_mask = cv2.fillPoly(label_mask, [label_points.astype(np.int32)], 1)

        pred_mask = np.zeros((imageW, imageH), dtype=np.uint8)
        pred_points_batch = pred_points_batch.reshape((-1, 1, 2))
        pred_mask = cv2.fillPoly(pred_mask, [pred_points_batch.astype(np.int32)], 1)


        if debug:
            cv2.imshow('label_mask', label_mask*255)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imshow('pred_mask', pred_mask*255)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        intersection = np.logical_and(label_mask, pred_mask)
        union = np.logical_or(label_mask, pred_mask)

        iou += np.sum(intersection) / np.sum(union)
        valid_count += 1

    return iou, valid_count, none_count

def calculate_oks(pred_points, label, debug=False):
    # label: [N, 5, 2]
    # pred_points: list of [5, 2] numpy arrays or Nones

    k = 0.1
    oks_total = 0
    valid_count = 0

    for batch_i in range(label.shape[0]):

        label_points = label[batch_i]
        pred_points_batch = pred_points[batch_i]

        if pred_points_batch is None:
            continue

        # first calculate s^2 the area of ground truth object
        # calculate w and h from label_points
        w = np.max(label_points[:, 0]) - np.min(label_points[:, 0])
        h = np.max(label_points[:, 1]) - np.min(label_points[:, 1])
        s_2 = w * h

        # calculate oks for each point
        oks_batch = 0

        for i in range(label.shape[1]):
            # calculate the distance between the predicted point and the ground truth point
            dist = np.linalg.norm(label_points[i] - pred_points_batch[i])
            oks_batch += np.exp(-dist**2 / (2 * s_2 * k**2))

        oks_total += oks_batch / label.shape[1]
        valid_count += 1

    return oks_total, valid_count




# img = np.zeros((512, 512, 3), np.uint8)

# # draw a rectangle at the center with 100 height using cv2.fillPoly() function
# points = np.array([[200, 200],  [300, 300],[300, 200], [200, 300]], dtype=np.int32)
# points = points.reshape((-1, 1, 2))
# cv2.fillPoly(img, [points], (255, 255, 255))

# # display the image
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

