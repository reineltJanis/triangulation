import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob


class Painting:
    def __init__(self, image) -> None:
        self.image = image
        self.base_image = image.copy()

    def add_markers(self, points):
        for i in range(0,len(points)):
            point = (round(points[i,0]),round(points[i,1]))
            marker = cv.MARKER_STAR if i != 9 else cv.MARKER_CROSS
            self.image = cv.drawMarker(self.image, point, (i/8*64%256,i/4*64%256,i*64%256), markerType=marker, markerSize=15, thickness=2, line_type=cv.LINE_AA)
        return self

    def resize(self, dimension):
        self.image = cv.resize(self.image, dimension)
        return self

    def add_horizontal_line(self, height, color=(255, 0, 0), strength=5):
        print(self.image.shape)
        cv.line(self.image, (0,height), (self.image.shape[1],height), color, strength)
        return self
    
    def warp_perspective(self, T, dimension=None):
        if dimension is None:
            dimension = (self.image.shape[0],self.image.shape[1])
        self.image = cv.warpPerspective(self.image,T,dimension,flags=cv.INTER_LINEAR)
        return self
        
    def plot(self):
        plt.imshow(self.image)
        return self
    
    def plot_base(self):
        plt.imshow(self.base_image)
        return self
