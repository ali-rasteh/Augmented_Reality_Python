import cv2
import numpy as np
import glob
from Camera_Calib import Camera_Calibrate

# Link : http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html#pose-estimation

def generate_camera_matrix(image):
    h, w = image.shape[:2]
    # let it be full frame matrix
    sx, sy = (36, 24)
    # focus length
    f = 50
    fx = w * f / sx
    fy = h * f / sy
    cx = w / 2
    cy = h / 2
    mtx = np.zeros((3, 3), np.float32)
    mtx[0, 0] = fx # [ fx  0  cx ]
    mtx[0, 2] = cx # [  0 fy  cy ]
    mtx[1, 1] = fy # [  0  0   1 ]
    mtx[1, 2] = cy
    mtx[2, 2] = 1
    return mtx


def generate_distorsions():
    return np.zeros((1, 4), np.float32)


# Codes for drawing circles on the image with specific points
def draw_circle_figure(img, pts):
    for i in range(0, pts.shape[0], 1):
        cv2.circle(img, (int(pts[i, 0, 0]), int(pts[i, 0, 1])), 2, (0, 0, 255), -1)
    return img


# Codes for drawing coordinates on the image with specific points
def draw_coordinates_figure(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img


# Codes for drawing Cube on the image with specific points
def draw_cube_figure(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


def AR_Create(img, corners, Figure_type):
    # mtx, dist, objpoints, imgpoints, rvecs, tvecs = Camera_Calibrate()
    mtx = generate_camera_matrix(img)
    dist = generate_distorsions()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((2*2,3), np.float32)
    objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
    # specifying Axis for drawing Figures
    if (Figure_type=="Cube"):
        axis = np.float32([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                       [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
    elif (Figure_type=="Coordinates"):
        axis = np.float32([[1,0,0], [0,1,0], [0,0,-1]]).reshape(-1,3)
    elif (Figure_type=="Circle"):
        axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).reshape(-1, 3)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
    temp = corners2[2,:]
    corners2[2,:]=corners2[3,:]
    corners2[3,:]=temp
    # using solvePnPRansac for specifying rvecs, tvecs
    retval, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    # specifying appropriate function to draw figures on the image
    if (Figure_type=="Cube"):
        img = draw_cube_figure(img, corners2, imgpts)
    elif (Figure_type=="Coordinates"):
        img = draw_coordinates_figure(img, corners2, imgpts)
    elif (Figure_type=="Circle"):
        img = draw_circle_figure(img, imgpts)
    return img
