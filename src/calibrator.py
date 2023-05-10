# from https://github.com/niconielsen32/ComputerVision/blob/master/cameraCalibration.py
# modified
import scipy
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
from src.painting import Painting


class Calibrator:
    img_path = 'assets/calibration_new/'
    
    def __init__(self, img_path, file_selector='*.jpg', frameSize=(1600,1200), chessboardSize=(7,5), chessboard_square_mm=29.15, termination_criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        np.set_printoptions(precision=9,suppress=True)
        self.img_path = img_path
        self.file_selector = file_selector
        self.frameSize = frameSize
        self.chessboardSize = chessboardSize
        self.size_of_chessboard_squares_mm = chessboard_square_mm
        self.criteria = termination_criteria

    ################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

    

    def calibrate(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)


        objp = objp * self.size_of_chessboard_squares_mm


        # Arrays to store object points and image points from all the images.
        self.objpoints = [] # 3d point in real world space
        self.imgpoints = [] # 2d points in image plane.


        self.images = glob.glob(self.img_path + self.file_selector)
        self.images.sort()

        for image in self.images:

            img = cv.imread(image)
            img = cv.resize(img, self.frameSize)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, self.chessboardSize, None)

            # If found, add object points, image points (after refining them)
            print(ret)
            if ret == True:

                self.objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                self.imgpoints.append(corners2)
        

        ############## CALIBRATION #######################################################

        self.ret, self.cameraMatrix, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.frameSize, None, None)

        #print("ImgPoints: ", imgpoints)

        print('')

        print("Camera calibrated: ", self.ret)
        print("Camera Matrix: ", self.cameraMatrix)
        print("Distortion Parameters: ", self.dist)
        print("Rotation Vectors: ", self.rvecs)
        print("Translation Vectors: ", self.tvecs)
        
        return self.cameraMatrix
    
    def calculateProjection(self, img_index):
        # ===========================  Calculate P  ==============================
        R = cv.Rodrigues(self.rvecs[img_index])[0]
        t = self.tvecs[img_index]
        W = np.concatenate([R,t], axis=-1) # [R|t]
        K = self.cameraMatrix
        P = K.dot(W)
        print('P for '+ self.file_selector + ', index: ' + str(img_index))
        print(P)
        return P,(K,W,R,t)


        ############## UNDISTORTION #####################################################

        #img = cv.imread(img_path + file_selector + "-0" + str(index+1) + ".jpg")
        #h,  w = img.shape[:2]
        #newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))


        # Undistort
        #dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

        # crop the image
        #x, y, w, h = roi
        #dst = dst[y:y+h, x:x+w]
        #cv.imwrite(out_path, dst)
    
    def plot_image(self, index, add_corners=True):
        arr = self.imgpoints_as_int
        painting = Painting(cv.imread(self.images[index]))
        if add_corners:
            painting.add_markers(arr)
        painting.plot()

    def imgpoints_as_int(self, index):
        return np.rint(self.imgpoints[index].reshape((-1,2)))


    def draw_corners(self):
        plt.figure(figsize=(30,20))
        for i, img_src in enumerate(self.images):
            # Draw and display the corners
            img = cv.imread(img_src)
            img = cv.resize(img, self.frameSize)
            corners_img = cv.drawChessboardCorners(img, (9,6), self.imgpoints[i], True)
            plt.subplot(2, 5, i+1)
            plt.imshow(corners_img)
            plt.axis("off")
        
        plt.show()
        

    # Undistort with Remapping
    #mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    #dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    #cv.imwrite('caliResult2.png', dst)




    # Reprojection Error
    #mean_error = 0

    #for i in range(len(objpoints)):
    #    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    #    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    #    mean_error += error

    #print( "total error: {}".format(mean_error/len(objpoints)) )
