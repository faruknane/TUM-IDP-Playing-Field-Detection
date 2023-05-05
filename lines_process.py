import numpy as np
import math
import cv2

# Define a function to calculate the angle (slope) of a line given two points
def angle(x1, y1, x2, y2):
	# Avoid division by zero
	if x2 == x1:
		return np.arctan(np.inf) 
	
	# Return the angle in radians
	return np.arctan((y2 - y1) / (x2 - x1))

def intersect(x1, y1, x2, y2, x3, y3, x4, y4):
  # Calculate the determinant of the matrix
  det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
  # If the determinant is zero, the lines are parallel or coincident
  if det == 0:
    return False
  # Otherwise, calculate the intersection point using Cramer's rule
  t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det
  u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / det
  # Check if the intersection point is within the segments
  if 0 <= t <= 1 and 0 <= u <= 1:
    return True
  # Otherwise, the lines do not intersect
  else:
    return False
  
def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude


#Calc minimum distance from a point and a line segment (i.e. consecutive vertices in a polyline).
def DistancePointLine (px, py, x1, y1, x2, y2):
    #http://local.wasp.uwa.edu.au/~pbourke/geometry/pointline/source.vba
    LineMag = lineMagnitude(x1, y1, x2, y2)

    if LineMag < 0.00000001:
        DistancePointLine = 9999
        return DistancePointLine

    u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
    u = u1 / (LineMag * LineMag)

    if (u < 0.00001) or (u > 1):
        #// closest point does not fall within the line segment, take the shorter distance
        #// to an endpoint
        ix = lineMagnitude(px, py, x1, y1)
        iy = lineMagnitude(px, py, x2, y2)
        if ix > iy:
            DistancePointLine = iy
        else:
            DistancePointLine = ix
    else:
        # Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        DistancePointLine = lineMagnitude(px, py, ix, iy)

    return DistancePointLine

def get_pixels_cv2(image, point1, point2):
  # image is a numpy array of pixel values
  # point1 and point2 are tuples of (x, y) coordinates
  # returns a list of tuples of (x, y) coordinates of pixels on the line

  # Get the shape of the image
  height, width = image.shape[:2]

  # Create a new image with the same shape and type as the original image
  new_image = np.zeros((height, width), dtype=image.dtype)

  # Draw a line between point1 and point2 on the new image
  cv2.line(new_image, point1, point2, color=255)

  # Find the indices of the pixels that are non-zero (i.e. on the line)
  indices = np.argwhere(new_image > 0)

  # Convert the indices to a list of tuples
  result = [tuple(index) for index in indices]

  return result

def get_pixels_cv2_optimized(point1, point2):
  # point1 and point2 are tuples of (x, y) coordinates
  # returns a list of tuples of (x, y) coordinates of pixels on the line

  # Find the minimum and maximum x and y values of the points
  min_x = min(point1[0], point2[0])
  max_x = max(point1[0], point2[0])
  min_y = min(point1[1], point2[1])
  max_y = max(point1[1], point2[1])

  # Create a new image with the smallest size that contains the points
  # and a single channel of type uint8
  new_image = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)

  # Draw a line between point1 and point2 on the new image
  # Adjust the coordinates by subtracting the minimum values
  cv2.line(new_image, (point1[0] - min_x, point1[1] - min_y), (point2[0] - min_x, point2[1] - min_y), color=255)

  # Find the indices of the pixels that are non-zero (i.e. on the line)
  indices = np.argwhere(new_image > 0)

  # Convert the indices to a list of tuples
  result = [tuple(index) for index in indices]

  # Add back the minimum values to get the original coordinates
  result = [(index[0] + min_y, index[1] + min_x) for index in result]

  return result

def CreateNewLine(mylinepoint1, mylinepoint2, dist = 5):
	# create the new line that has the same angle as my line, but it moves outwards in a perpendicular direction
	# the distance between the original line and new line should be 5 pixels
	mylineangle = angle(mylinepoint1[0], mylinepoint1[1], mylinepoint2[0], mylinepoint2[1])
	mynewlinepoint1 = (mylinepoint1[0] + dist*math.cos(math.radians(mylineangle+90)), mylinepoint1[1] + dist*math.sin(math.radians(mylineangle+90)))
	mynewlinepoint2 = (mylinepoint2[0] + dist*math.cos(math.radians(mylineangle+90)), mylinepoint2[1] + dist*math.sin(math.radians(mylineangle+90)))
	return mynewlinepoint1, mynewlinepoint2

def GetPixelsMean(gray_scale_image, indices, out_of_bounds_value = 0):
	sum_pixel = 0
	count = 0
	for index in indices:
		if index[0] >= 0 and index[0] < gray_scale_image.shape[0] and index[1] >= 0 and index[1] < gray_scale_image.shape[1]:
			sum_pixel += gray_scale_image[index]
			# gray_scale_image[index] = 0
			count += 1
	
	if count == 0:
		return out_of_bounds_value, 0, 1
	
	return sum_pixel / count, sum_pixel, count

def GetIfLineAccepted(gray_scale_image, mylinepoint1, mylinepoint2, dist):
	# Get the indices of pixels that are on the line
	indices = get_pixels_cv2_optimized((int(mylinepoint1[0]), int(mylinepoint1[1])), (int(mylinepoint2[0]), int(mylinepoint2[1])))
	
	newline1 = CreateNewLine(mylinepoint1, mylinepoint2, dist)
	newline2 = CreateNewLine(mylinepoint1, mylinepoint2, -dist)

	indices1 = get_pixels_cv2_optimized((int(newline1[0][0]), int(newline1[0][1])), (int(newline1[1][0]), int(newline1[1][1])))
	indices2 = get_pixels_cv2_optimized((int(newline2[0][0]), int(newline2[0][1])), (int(newline2[1][0]), int(newline2[1][1])))

	# create new line 
	
	# Get the mean of the pixels on the line
	mean, _, _ = GetPixelsMean(gray_scale_image, indices)
	mean1, _, _  = GetPixelsMean(gray_scale_image, indices1)
	mean2, _, _  = GetPixelsMean(gray_scale_image, indices2)

	return mean*2 > mean1 + mean2
	
def FilterLines(gray_scale_image, lines, dist):
	
	# iterate over lines
	# foreach line, find the indices of pixels that are on the line
	# then sum the pixels on the line

	accepted_lines = []

	for line in lines:
		# Get two points as tuple of myline
		mylinepoint1 = (float(line[0][0]), float(line[0][1]))
		mylinepoint2 = (float(line[0][2]), float(line[0][3]))

		if GetIfLineAccepted(gray_scale_image, mylinepoint1, mylinepoint2, dist):
			accepted_lines.append(line)

	return accepted_lines

def FinalFilterLines(gray_scale_image, lines, dist, max_count):
	
	gray_scale_image = cv2.dilate(gray_scale_image, np.ones((11,11), np.uint8), iterations = 1)
	 
	priorities = []

	for line in lines:
		# Get two points as tuple of myline
		mylinepoint1 = (float(line[0][0]), float(line[0][1]))
		mylinepoint2 = (float(line[0][2]), float(line[0][3]))

		# Get the indices of pixels that are on the line
		indices = get_pixels_cv2_optimized((int(mylinepoint1[0]), int(mylinepoint1[1])), (int(mylinepoint2[0]), int(mylinepoint2[1])))
		
		add1 = CreateNewLine(mylinepoint1, mylinepoint2, 1)
		add2 = CreateNewLine(mylinepoint1, mylinepoint2, -1)
		
		# newline1_1 = CreateNewLine(mylinepoint1, mylinepoint2, dist)
		# newline1_2 = CreateNewLine(mylinepoint1, mylinepoint2, dist+1)
		# newline1_3 = CreateNewLine(mylinepoint1, mylinepoint2, dist-1)
		# newline2_1 = CreateNewLine(mylinepoint1, mylinepoint2, -dist)
		# newline2_2 = CreateNewLine(mylinepoint1, mylinepoint2, -dist+1)
		# newline2_3 = CreateNewLine(mylinepoint1, mylinepoint2, -dist-1)

		add1_indices = get_pixels_cv2_optimized((int(add1[0][0]), int(add1[0][1])), (int(add1[1][0]), int(add1[1][1])))
		add2_indices = get_pixels_cv2_optimized((int(add2[0][0]), int(add2[0][1])), (int(add2[1][0]), int(add2[1][1])))

		# newline1_1_indices = get_pixels_cv2_optimized((int(newline1_1[0][0]), int(newline1_1[0][1])), (int(newline1_1[1][0]), int(newline1_1[1][1])))
		# newline1_2_indices = get_pixels_cv2_optimized((int(newline1_2[0][0]), int(newline1_2[0][1])), (int(newline1_2[1][0]), int(newline1_2[1][1])))
		# newline1_3_indices = get_pixels_cv2_optimized((int(newline1_3[0][0]), int(newline1_3[0][1])), (int(newline1_3[1][0]), int(newline1_3[1][1])))
		# newline2_1_indices = get_pixels_cv2_optimized((int(newline2_1[0][0]), int(newline2_1[0][1])), (int(newline2_1[1][0]), int(newline2_1[1][1])))
		# newline2_2_indices = get_pixels_cv2_optimized((int(newline2_2[0][0]), int(newline2_2[0][1])), (int(newline2_2[1][0]), int(newline2_2[1][1])))
		# newline2_3_indices = get_pixels_cv2_optimized((int(newline2_3[0][0]), int(newline2_3[0][1])), (int(newline2_3[1][0]), int(newline2_3[1][1])))
		
		# Get the mean of the pixels on the line
		add_mean, sum_mean, _ = GetPixelsMean(gray_scale_image, indices)
		add1_mean, sum1_mean, _ = GetPixelsMean(gray_scale_image, add1_indices)
		add2_mean, sum2_mean, _ = GetPixelsMean(gray_scale_image, add2_indices)

		# newline1_1_mean, _, _ = GetPixelsMean(gray_scale_image, newline1_1_indices)
		# newline1_2_mean, _, _ = GetPixelsMean(gray_scale_image, newline1_2_indices)
		# newline1_3_mean, _, _ = GetPixelsMean(gray_scale_image, newline1_3_indices)
		# newline2_1_mean, _, _ = GetPixelsMean(gray_scale_image, newline2_1_indices)
		# newline2_2_mean, _, _ = GetPixelsMean(gray_scale_image, newline2_2_indices)
		# newline2_3_mean, _, _ = GetPixelsMean(gray_scale_image, newline2_3_indices)

		# total_add_mean = (add1_mean + add2_mean + add_mean)/3
		# total_newline1_mean = (newline1_1_mean + newline1_2_mean + newline1_3_mean)/3
		# total_newline2_mean = (newline2_1_mean + newline2_2_mean + newline2_3_mean)/3

		# priorities.append(total_add_mean*2 - total_newline1_mean - total_newline2_mean)

		total_add_mean = max(add1_mean, add2_mean, add_mean)

		if total_add_mean == add_mean:
			add_sum = sum_mean
		elif total_add_mean == add1_mean:
			add_sum = sum1_mean
		else:
			add_sum = sum2_mean

		priorities.append(add_sum)


	# sort the lines according to their priorities
	# max priority is the first element

	sorted_lines = [x for _,x in sorted(zip(priorities, lines), reverse=True, key=lambda pair: pair[0])]

	return sorted_lines[0:max_count]

# gray_scale_image = np.ones((1000, 1000), dtype=np.uint8)*150

# lines = []

# lines.append([[200, 30, 600, 500]])

# FilterLines(gray_scale_image, lines, 0.5)

# cv2.imshow('gray_scale_image', gray_scale_image)
# cv2.waitKey(0)

def SortLines(lines):
	# sort lines according to their slope

	# get the slope of each line
	# sort the lines according to their slope
	# return the sorted lines

	slopes = []
	for line in lines:
		# Get two points as tuple of myline
		mylinepoint1 = (float(line[0][0]), float(line[0][1]))
		mylinepoint2 = (float(line[0][2]), float(line[0][3]))

		slope = angle(mylinepoint1[0], mylinepoint1[1], mylinepoint2[0], mylinepoint2[1])
 
		slopes.append(slope)


	# this gives error -> sorted_lines = [x for _,x in sorted(zip(slopes,lines))]
	# this works -> sorted_lines = [x for _,x in sorted(zip(slopes,lines), key=lambda pair: pair[0])]

	# sort the lines according to their slope
	sorted_lines = [x for _,x in sorted(zip(slopes,lines), key=lambda pair: pair[0])]

	return sorted_lines

def Process(lines, dist_threshold, angle_threshold, angle_threshold2 = 10):
	
	lines = SortLines(lines)

	index = 0
	while index < len(lines):

		index2 = index + 1
		while index2 < len(lines):

			myline = lines[index] 
			secondline = lines[index2]
			
			# Get two points as tuple of myline
			mylinepoint1 = (float(myline[0][0]), float(myline[0][1]))
			mylinepoint2 = (float(myline[0][2]), float(myline[0][3]))

			# Get two points as tuple of secondline
			secondlinepoint1 = (float(secondline[0][0]), float(secondline[0][1]))
			secondlinepoint2 = (float(secondline[0][2]), float(secondline[0][3]))

			# Get the angle of the my line using mylinepoint1 and mylinepoint2
			mylineangle = angle(mylinepoint1[0], mylinepoint1[1], mylinepoint2[0], mylinepoint2[1])

			# Get the angle of the second line
			secondlineangle = angle(secondlinepoint1[0], secondlinepoint1[1], secondlinepoint2[0], secondlinepoint2[1])
			
			# Get the angle difference
			angle_diff = abs(mylineangle - secondlineangle) 
			angle_diff = min(angle_diff, math.pi - angle_diff)

			# get the distance of point secondlinepoint1 to the line myline which does not extend infinity
			dist1 = DistancePointLine(secondlinepoint1[0], secondlinepoint1[1], mylinepoint1[0], mylinepoint1[1], mylinepoint2[0], mylinepoint2[1])
			
			# get the distance of point secondlinepoint2 to the line myline which does not extend infinity
			dist2 = DistancePointLine(secondlinepoint2[0], secondlinepoint2[1], mylinepoint1[0], mylinepoint1[1], mylinepoint2[0], mylinepoint2[1])

			dist = min(dist1, dist2)

			# Check if the two lines distance is less than 10 pixels and the angle difference is less than 0.03 radians
			if dist < dist_threshold and angle_diff < angle_threshold:
                                
				# calculate the distances of points between mylinepoint1 and mylinepoint2 and secondlinepoint1 and secondlinepoint2 use np.linalg.norm
				dist1 = np.linalg.norm(np.array(mylinepoint1) - np.array(secondlinepoint1))
				dist2 = np.linalg.norm(np.array(mylinepoint1) - np.array(secondlinepoint2))
				dist3 = np.linalg.norm(np.array(mylinepoint2) - np.array(secondlinepoint1))
				dist4 = np.linalg.norm(np.array(mylinepoint2) - np.array(secondlinepoint2))
                                
				dist5 = np.linalg.norm(np.array(mylinepoint1) - np.array(mylinepoint2))
				dist6 = np.linalg.norm(np.array(secondlinepoint1) - np.array(secondlinepoint2))
                                
				dist = max(dist1, dist2, dist3, dist4, dist5, dist6)
                                
				if dist == dist1:
					mynewline = [mylinepoint1[0], mylinepoint1[1], secondlinepoint1[0], secondlinepoint1[1]]
				elif dist == dist2:
					mynewline = [mylinepoint1[0], mylinepoint1[1], secondlinepoint2[0], secondlinepoint2[1]]	
				elif dist == dist3:
					mynewline = [mylinepoint2[0], mylinepoint2[1], secondlinepoint1[0], secondlinepoint1[1]]	
				elif dist == dist4:
					mynewline = [mylinepoint2[0], mylinepoint2[1], secondlinepoint2[0], secondlinepoint2[1]]
				elif dist == dist5:
					mynewline = [mylinepoint1[0], mylinepoint1[1], mylinepoint2[0], mylinepoint2[1]]
				elif dist == dist6:
					mynewline = [secondlinepoint1[0], secondlinepoint1[1], secondlinepoint2[0], secondlinepoint2[1]]
				
				# # weighted average of the two angles, weights are the lengths of the lines
				# newangle = (mylineangle * dist5 + secondlineangle * dist6) / (dist5 + dist6)

				# # get the new line length
				# newlength = np.linalg.norm(np.array(mynewline[0:2]) - np.array(mynewline[2:4]))

				# # get the center point of new line
				# newlinecenter = ((mynewline[0] + mynewline[2]) / 2, (mynewline[1] + mynewline[3]) / 2)

				# # now, expand from the newlinecenter along with the newangle to get the new line endpoints
				# newlinepoint1 = (newlinecenter[0] + newlength / 2 * np.cos(newangle), newlinecenter[1] + newlength / 2 * np.sin(newangle))
				# newlinepoint2 = (newlinecenter[0] - newlength / 2 * np.cos(newangle), newlinecenter[1] - newlength / 2 * np.sin(newangle))

				# mynewline = [newlinepoint1[0], newlinepoint1[1], newlinepoint2[0], newlinepoint2[1]] 

				# calculate the angle of mynewline
				mynewlineangle = angle(mynewline[0], mynewline[1], mynewline[2], mynewline[3])				

				new_diff_angle = abs(mylineangle - mynewlineangle)
				new_diff_angle = min(new_diff_angle, math.pi - new_diff_angle)
				new_diff_angle = min(new_diff_angle, math.pi*2 - new_diff_angle)

				new_diff_angle2 = abs(secondlineangle - mynewlineangle)
				new_diff_angle2 = min(new_diff_angle2, math.pi - new_diff_angle2)
				new_diff_angle2 = min(new_diff_angle2, math.pi*2 - new_diff_angle2)

				if new_diff_angle < angle_threshold2 and new_diff_angle2 < angle_threshold2:
					lines = np.delete(lines, index2, 0)
					lines[index] = [mynewline]
				else:
					index2 += 1

				# print("line deleted")

			else:
				index2 += 1

		index += 1


	return lines


# def ProcessOld(lines):

# 	index = 0
# 	while index < len(lines):
# 		myline = lines[index] 

# 		index2 = index + 1
# 		while index2 < len(lines):
# 			secondline = lines[index2]
			
# 			# Get two points as tuple of myline
# 			mylinepoint1 = (float(myline[0][0]), float(myline[0][1]))
# 			mylinepoint2 = (float(myline[0][2]), float(myline[0][3]))

# 			# Get two points as tuple of secondline
# 			secondlinepoint1 = (float(secondline[0][0]), float(secondline[0][1]))
# 			secondlinepoint2 = (float(secondline[0][2]), float(secondline[0][3]))

# 			# Get the angle of the my line using mylinepoint1 and mylinepoint2
# 			mylineangle = angle(mylinepoint1[0], mylinepoint1[1], mylinepoint2[0], mylinepoint2[1])


# 			# Get the angle of the second line
# 			secondlineangle = angle(secondlinepoint1[0], secondlinepoint1[1], secondlinepoint2[0], secondlinepoint2[1])
			
# 			# Get the angle difference
# 			angle_diff = abs(mylineangle - secondlineangle) 
			
# 			# get the min distance of 4 points of the two lines 
# 			dist1 = np.linalg.norm(np.array(mylinepoint1) - np.array(secondlinepoint1))
# 			dist2 = np.linalg.norm(np.array(mylinepoint1) - np.array(secondlinepoint2))
# 			dist3 = np.linalg.norm(np.array(mylinepoint2) - np.array(secondlinepoint1))
# 			dist4 = np.linalg.norm(np.array(mylinepoint2) - np.array(secondlinepoint2))
# 			dist = min(dist1, dist2, dist3, dist4)


# 			# Check if the two lines distance is less than 10 pixels and the angle difference is less than 0.05 radians
# 			if dist < 25 and angle_diff < 0.05:
					
# 				# calculate the distance between two points on the same line
# 				dist_myline = np.linalg.norm(np.array(mylinepoint1) - np.array(mylinepoint2))
# 				dist_secondline = np.linalg.norm(np.array(secondlinepoint1) - np.array(secondlinepoint2))

# 				mynewline = [mylinepoint1[0], mylinepoint1[1], mylinepoint2[0], mylinepoint2[1]]

# 				# merge the two lines by removing first_line_closest_point and second_line_closest_point
# 				if dist == dist1:
# 					mynewline[0] = secondlinepoint2[0]
# 					mynewline[1] = secondlinepoint2[1]
# 				elif dist == dist2: 
# 					mynewline[0] = secondlinepoint1[0]
# 					mynewline[1] = secondlinepoint1[1]
# 				elif dist == dist3: 
# 					mynewline[2] = secondlinepoint2[0]
# 					mynewline[3] = secondlinepoint2[1]
# 				else:
# 					mynewline[2] = secondlinepoint1[0]
# 					mynewline[3] = secondlinepoint1[1]

# 				# calculate the distance between two points on the new line
# 				dist_newline = np.linalg.norm(np.array(mynewline[0:2]) - np.array(mynewline[2:4]))

# 				# new line distance should be greater
# 				if dist_newline < (dist_myline + dist_secondline)*0.6:
# 					index2 += 1
# 				else:
# 					lines = np.delete(lines, index2, 0)
# 					lines[index] = [mynewline]

# 					print("line deleted")

# 			else:
# 				index2 += 1

# 		index += 1


# 	return lines
if __name__ == "__main__":

	p1 = [[3119, 1706, 3106,  841]]
	p2 = [[3107,  889, 3107,  579]]

	lines = [p2, p1]

	lines = Process(lines, 5, 0.015)

	print(lines)


	# mylinepoint1 = (5, 5)
	# mylinepoint2 = (100, 100)
	# mylineangle = angle(mylinepoint1[0], mylinepoint1[1], mylinepoint2[0], mylinepoint2[1])
	# # print(mylineangle)
	# print(abs(math.pi/2 - abs(mylineangle)))

	pass