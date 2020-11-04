# -*- coding: utf-8 -*-
import argparse

# import the necessary packages
import numpy as np
import cv2
import os
import json
from scipy.spatial.qhull import ConvexHull


#https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

def order_points(pts):
	# convexHull is more consistent in order the point clockwise
	cv = ConvexHull(pts)
	rect_r = cv.vertices
	# Order by top left point
	rect_r = [pts[i] for i in rect_r]
	rect_r = np.array(rect_r)
	top_left_point = np.where(rect_r[:, 0] + rect_r[:, 1] == min(rect_r[:, 0] + rect_r[:, 1]))[0][0]

	rect = np.append(rect_r[top_left_point:], rect_r[:top_left_point], axis=0)

	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--path_to_images", action="store", required=True, help="path to the images to warp.")
	parser.add_argument("--path_to_masks", action="store", required=True, help="path to the mask of the images.")
	parser.add_argument("--output", action="store", default="./", help="path where the results should be saved.")
	parser.add_argument("--invert", action="store_true", default=False, help="If true, invert the colors of the image, "																 "after converting the image to grayscale.")
	args = parser.parse_args()

	IMG_EXTENSIONS = ["png", "jpg", "jpeg"]
	print(args.path_to_images)
	IMAGES = [file for file in os.listdir(args.path_to_images) if (file.split(".")[-1] in IMG_EXTENSIONS)]
	LABELS = [file for file in os.listdir(args.path_to_masks) if file.split(".")[-1] == "json"]
	print(IMAGES)
	print(LABELS)
	if not os.path.isdir(args.output):
		os.makedirs(args.output)
	point_dict = {}
	for label in LABELS:
		with open(os.path.join(args.path_to_masks, label), "r") as json_file:
			data = json.load(json_file)
			point_dict[label.split(".")[0]] = data["shapes"][0]["points"]
			json_file.close()
	image_path_to_points = []
	for image_name in IMAGES:
		try:
			image_path_to_points.append((image_name, point_dict[image_name.split(".")[0]]))
		except KeyError as e:
			print("No Label for", image_name)

	print(image_path_to_points)

	for image_name, points in image_path_to_points:
		image_path = os.path.join(args.path_to_images, image_name)
		image = cv2.imread(image_path)
		pts = np.array(points, dtype="float32")
		# apply the four point transform to obtain a "birds eye view" of
		# the image
		warped = four_point_transform(image, pts)
		# show the original and warped images

		if args.invert:
			print("Writing inverted Image", image_name)
			gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
			w_to_b = cv2.bitwise_not(gray)
			image_to_write = w_to_b

		else:
			print("Writing image", image_name)
			image_to_write = warped

		cv2.imwrite(os.path.join(args.output, image_name.split(".")[0] + ".png"), image_to_write)
