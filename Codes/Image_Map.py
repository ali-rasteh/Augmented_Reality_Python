# import the necessary packages
import numpy as np
import imutils
import cv2
import glob
from AR import AR_Create

class Image_Mapper:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()

	def Draw_Border(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		[Card1, Card2, Card1_in_scene] = images
		Card1_in_scene_Border = list(Card1_in_scene)
		Card1_in_scene_Border = np.array(Card1_in_scene_Border)
		(kpsA, featuresA) = self.detectAndDescribe(Card1)
		(kpsB, featuresB) = self.detectAndDescribe(Card1_in_scene)
		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
								featuresA, featuresB, ratio, reprojThresh)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None
		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		h, w, RGB = Card1.shape
		pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
		if (type(H) != type(None)):
			dst = cv2.perspectiveTransform(pts, H)
			Card1_in_scene_Border = cv2.polylines(Card1_in_scene_Border, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(Card1, Card1_in_scene, kpsA, kpsB, matches,
								   status)
			# return a tuple of the stitched image and the
			# visualization
			return [Card1_in_scene_Border, vis]
		# return the stitched image
		return Card1_in_scene_Border


	def Draw_Cube(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		[Card1, Card2, Card1_in_scene] = images
		Card1_in_scene_cube = list(Card1_in_scene)
		Card1_in_scene_cube=np.array(Card1_in_scene_cube)
		(kpsA, featuresA) = self.detectAndDescribe(Card1)
		(kpsB, featuresB) = self.detectAndDescribe(Card1_in_scene)
		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
								featuresA, featuresB, ratio, reprojThresh)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None
		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		h, w, RGB = Card1.shape
		pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
		if(type(H) != type(None)):
			dst = cv2.perspectiveTransform(pts, H)
			# Figure_type="Cube"
			Figure_type="Coordinates"
			# Figure_type="Circle"
			Card1_in_scene_cube = AR_Create(Card1_in_scene_cube, dst, Figure_type)
			# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(Card1, Card1_in_scene, kpsA, kpsB, matches,
								   status)
			# return a tuple of the stitched image and the
			# visualization
			return [Card1_in_scene_cube, vis]
		# return the stitched image
		return Card1_in_scene_cube


	def Image_Map(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		[Card1, Card2, Card1_in_scene] = images
		Card2_in_scene = list(Card1_in_scene)
		Card2_in_scene=np.array(Card2_in_scene)
		(kpsA, featuresA) = self.detectAndDescribe(Card1)
		(kpsB, featuresB) = self.detectAndDescribe(Card1_in_scene)
		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
								featuresA, featuresB, ratio, reprojThresh)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None
		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		if (type(H) != type(None)):
			for i in range(Card2.shape[0]):
				for j in range(Card2.shape[1]):
					pts = np.float32([[j, i]]).reshape(-1, 1, 2)
					dst = cv2.perspectiveTransform(pts, H)
					x=int(dst[0,0,0])
					y=int(dst[0,0,1])
					if(y>=0 and y<480 and x>=0 and x<640):
						Card2_in_scene[y, x]=Card2[i,j]
		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(Card1, Card1_in_scene, kpsA, kpsB, matches,
								   status)
			# return a tuple of the stitched image and the
			# visualization
			return [Card2_in_scene, vis]
		# return the stitched image
		return Card2_in_scene


	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			# descriptor = cv2.xfeatures2d.SIFT_create()
			descriptor = cv2.AKAZE_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			# detector = cv2.FeatureDetector_create("SIFT")
			detector = cv2.AKAZE_create()
			kps = detector.detect(gray)
			# extract features from the image
			# extractor = cv2.DescriptorExtractor_create("SIFT")
			descriptor = cv2.AKAZE_create()
			extractor = descriptor.DescriptorExtractor_create()
			(kps, features) = extractor.compute(gray, kps)
		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])
		# return a tuple of keypoints and features
		return (kps, features)


	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		# matcher = cv2.DescriptorMatcher_create("BruteForce")
		matcher = cv2.BFMatcher()
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []
		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)
		# otherwise, no homograpy could be computed
		return None


	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
		# return the visualization
		return vis
