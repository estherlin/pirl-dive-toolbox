#!/usr/bin/env python

# Library imports needed for all processing in this file
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import re

# Filename regular expressions
BMP_REGEX = re.compile(
    r'.*(?P<date>\d{8})\_(?P<material>[a-z]+)\_(?P<wavelength_nm>\d+)\_[\+]*(?P<fringe_num>[-\d]+[\.]*\d*)\_(?P<item>[0-9]+).bmp',
    re.IGNORECASE
)

class VideoEdit(object):
    """
    :param vidFile: complete file name including directory of videoFile
    :returns: VideoFile object
    """

    # Constructor for initialization
    def __init__ (self, filename, cut=False, bin_num=None, angle=0.00, x_center=None, y_center=None):
        """ Clip at 110"""
        # Create a VideoCapture object
        self.vidFile = cv2.VideoCapture(filename)
        self.frames = []
        self.bin_img = []
        self.fps = self.vidFile.get(cv2.CAP_PROP_FPS)    # Frame rate

        count = 0
        # Read in each frame and sage
        success,image = self.vidFile.read()
        while success:
            image[image > 100] = 100 # Filter out the saturated values
            self.frames.append(image)
            success,image = self.vidFile.read()
            count = count + 1

        # Constants
        self.frame_count = len(self.frames)
        self.bin = bin_num if bin_num is not None else self.frame_count
        self.angle = angle

        # Other parameters
        self.x_center, self.y_center = VideoEdit.find_center(
            self.frames[0][:,:,1]) # Choose one of the earlier frames
        #print("%s: ( %.2f, %.2f)" % (filename, self.x_center, self.y_center))
        if x_center is not None:
            self.x_center = x_center
        if y_center is not None:
            self.y_center = y_center

        self.dx = 50   # ranges of what to cut 150
        self.dy = 50

        # Rotate images before adding
        self.rot_frames = self.rotate_vid(angle=self.angle)
        self.cut_frames = []

        # Cut images
        if cut:
            for i in np.arange(0, count):
                self.cut_frames.append(self.rot_frames[i][self.y_center-self.dy:self.y_center+self.dy, self.x_center-self.dx:self.x_center+self.dx,:])
            self.cut_frames = np.array(self.cut_frames)
        else:
            self.cut_frames = self.rot_frames

        # Update parameters
        self.xshape = self.cut_frames[0].shape[1]
        self.yshape = self.cut_frames[0].shape[0]
        self.x_pts = np.linspace(1, self.xshape, self.xshape)
        self.y_pts = np.linspace(1, self.yshape, self.yshape)

        # Bin images, add and normalize
        for i in np.arange(0, int(self.frame_count/self.bin)):
            img = np.array(self.cut_frames[self.bin*i][:,:,1], dtype=np.int64)

            for j in np.arange(1, self.bin):
                img = img + np.array(self.cut_frames[self.bin*i+j][:,:,1], dtype=np.int64)

            self.bin_img.append(img)
            #cv2.imwrite("frame%d.jpg" % i, self.bin_img[i])     # save frame as JPEG file


    def rotate_vid(self,angle):
        """
        TODO:
        Determines the orientation of the
        :param1: VideoEdit object
        :param2: angle to rotate
        :returns: change in angle (ccw +)

        Strategy:
        1. rotate about max intensity point in the image
        2. track when peak is maximized
        """

        # Find the location of the max point in the picture from the first image
        center = (self.x_center, self.y_center)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # list of all of the rotation matrices and the warpAffines
        results = []
        for i in np.arange(0, len(self.frames)):
            result = cv2.warpAffine(self.frames[i], rot_mat, self.frames[i].shape[1::-1], flags=cv2.INTER_LINEAR)
            results.append(result)
            #if i == 0:
                #cv2.imwrite("angle%d.jpg" % (angle), result)     # save frame as JPEG file
        return np.array(results)

    def project_1d_tox (self):
        """
        Projects each image to a one d array by summ up all pixel values
        in each column, then normalizing all values

        :param: VideoEdit object
        :returns: 3D arrays of corrected value, vs the x axis, over time
        """

        proj = []
        x_pts = np.linspace(1,self.xshape, self.xshape)
        for img in np.arange(0,len(self.bin_img)):
            proj_noisy = self.bin_img[img].sum(axis=0)
            proj_filtered = proj_noisy / np.max(proj_noisy)
            proj.append(proj_filtered)
            #print(proj[img].shape, x_pts.shape)
            #plt.plot(x_pts, proj[img],label="%d" % img)

        plt.ylabel('Intensity')
        plt.xlabel('x pixels')
        plt.title('Intensity vs x ')
        plt.grid(True)
        #plt.legend()
        #plt.show()
        return proj

    def project_1d_toy (self):
        """
        Projects each image to a one d array by summ up all pixel values
        in each column, then normalizing all values

        :param: VideoEdit object
        :returns: 3D arrays of corrected value, vs the y axis, over time
        """
        proj = []
        y_pts = np.linspace(1,self.yshape, self.yshape)
        for img in np.arange(0,len(self.bin_img)):
            proj_noisy = self.bin_img[img].sum(axis=1)
            proj_filtered = proj_noisy / np.max(proj_noisy)
            proj.append(proj_filtered)
            #print(proj[img].shape, x_pts.shape)
            #plt.plot(y_pts, proj[img],label="%d" % img)

        plt.ylabel('Intensity')
        plt.xlabel('y pixels')
        plt.title('Intensity vs y ')
        plt.grid(True)
        #plt.legend()
        #plt.show()
        return proj

    @staticmethod
    def find_center(img):
        """
        Finds the center of the image
        """
        proj_x, proj_y = VideoEdit.xy_projections(img)
        x_center = np.argmax(proj_x)
        y_center = np.argmax(proj_y)
        return x_center, y_center


    def find_centers(self):
        """
        Finds the centers needed to pivot the rotations for each binned image

        :param: VideoEdit object
        :returns: list of tuples
        """
        centers = []

        # Find the x and y projections
        proj_x = self.project_1d_tox()
        proj_y = self.project_1d_toy()

        for i in np.arange(0, len(self.bin_img)):
            centers.append((np.argmax(proj_x[i]), np.argmax(proj_y[i])))

        return centers

    @staticmethod
    def xy_projections(img):
        """
        Finds the the projection in the x and y direction for a given image
        :param: image
        :returns: x and y projections in 2 numpy arrays
        """
        proj_x = img.sum(axis=0)
        proj_y = img.sum(axis=1)

        return proj_x, proj_y

    @staticmethod
    def rotate_img(img, angle, x_center, y_center):
        """
        TODO:
        Determines the orientation of the
        :param1: VideoEdit object
        :param2: angle to rotate
        :returns: change in angle (ccw +)

        Strategy:
        1. rotate about max intensity point in the image
        2. track when peak is maximized
        """

        # Find the location of the max point in the picture from the first image
        center = (x_center, y_center)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # list of all of the rotation matrices and the warpAffines
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        return np.array(result)

    @staticmethod
    def normalize(data):
        """
        Normalizes each array with the max value in the input
        :param: data array
        :returns: normalized array
        """
        return data/np.max(data)


def get_filenames(dir):
    """
    Gets the files in a directoryself.Directory has to be the lowest level
    :param dir: directory to start searchself.
    :returns filename: list of all of the files in directory that match BMP_REGEX
    """
    filenames = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    return filenames

if __name__== "__main__":

    # Get all of the file names
    dir_name = "//win.desy.de/group/cfel/4all/mpsd_drive/massspec/Users/Esther/11092018/"
    all_filenames = get_filenames(dir_name)

    grouped_data = {}

    for filename in all_filenames:
        match = BMP_REGEX.match(filename)

        if match:
            fringe_num = float(match.group('fringe_num'))
            if fringe_num in grouped_data:
                grouped_data[fringe_num].append(filename)
            else:
                grouped_data[fringe_num] = [filename]

    for key, value in grouped_data.items():
        print(key, value)