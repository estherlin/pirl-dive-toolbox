# Library imports needed for all processing in this file
import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import re
from scipy.optimize import curve_fit
from interference import Interference
import collections
from videoparser import VideoEdit, get_filenames, BMP_REGEX
from scipy.interpolate import interp1d

if __name__== "__main__":

    # Constants
    x_pts = np.linspace(1,800, num=800)         # width of image in pixels
    spline_x_pts = np.linspace(1,800, num=8000) # number of points for spline
    rot_ang = 0.6                               # angle in degrees to rotate images by
    dx = 125                                    # radius of mask to generate
    keys = ["cs", "spline_x", "dPsi"]           # parameters to extract from fit
    num_pix = 29.4                              # Number of pixels from peak to peak 

    # Directory that contains images to analyze - extract all image filenames
    dir_name = "//win.desy.de/group/cfel/4all/mpsd_drive/massspec/Users/Esther/data/21092018/fringes/1511/"
    listdir(dir_name)
    all_filenames= get_filenames(dir_name)
    print("processing for %s..." % dir_name)

    # Group all images by angle or whether or not there is a sample
    grouped_data = {} 
    prob_no_sample = 0 # Image of no thin film present

    for filename in all_filenames:
        match = BMP_REGEX.match(filename)

        if match:
            fringe_num = float(match.group('fringe_num'))
            if fringe_num in grouped_data:
                grouped_data[fringe_num] = grouped_data[fringe_num] + VideoEdit(filename, cut=False, bin_num=1).frames[0][:,:,1]
            else:
                grouped_data[fringe_num] = VideoEdit(filename, cut=False, bin_num=1).frames[0][:,:,1]

        elif "non" in filename:
            prob_no_sample = VideoEdit(filename, cut=False, bin_num=1).frames[0][:,:,1] +prob_no_sample
        elif "probing_beam_with_sample.bmp" in filename:
            prob_with_sample = VideoEdit.xy_projections(VideoEdit(filename, cut=False, bin_num=1).frames[0][:,:,1])[0]

    # Remember to sort! i don't know why actually, but it doesn't hurt
    grouped_data = collections.OrderedDict(sorted(grouped_data.items()))


    # We start our fits by fitting to image that has thin film at 0 degrees
    print("Fit to image taken with thin film at 0 degrees ... ")

    # Find the calibration interference pattern first.
    calData = grouped_data[0]
    x_center, y_center = VideoEdit.find_center(calData)

    # create a circular mask with the centers first
    circle_img = np.zeros((calData.shape[0], calData.shape[1]))
    cv2.circle(circle_img, (x_center, y_center), dx, 1, thickness=-1)
    circle_img = circle_img.astype(np.int8)

    # Then mask all data with the and operation
    calData = cv2.bitwise_and(calData, calData, mask=circle_img)
    max_val = np.max(VideoEdit.xy_projections(VideoEdit.rotate_img(calData, rot_ang, x_center, y_center))[0])
    calData_x = VideoEdit.xy_projections(VideoEdit.rotate_img(calData, rot_ang, x_center, y_center))[0]/max_val
    calData_spline_func = interp1d(x_pts, calData_x , fill_value="extrapolate", kind=3)
    calData_spline_x = calData_spline_func(spline_x_pts)

    # Now proceed to fit each image:
    results = collections.OrderedDict()
    dPsi = collections.OrderedDict()

    for key, value in grouped_data.items():
        value = cv2.bitwise_and(value, value, mask=circle_img)
        proj_x = VideoEdit.xy_projections(VideoEdit.rotate_img(value, rot_ang, x_center, y_center))[0]/max_val
        currData_spline_func = interp1d(x_pts, proj_x , fill_value="extrapolate", kind=3)
        currData_spline_x = currData_spline_func(spline_x_pts)
        int_cs =  Interference.fitter(spline_x_pts[10*(x_center-dx):10*(x_center+dx)], currData_spline_x[10*(x_center-dx):10*(x_center+dx)], spline=calData_spline_func, mode="spline2")

        spline_fit = Interference.make_spline2(x_pts, calData_spline_func, *int_cs)
        results[key] = dict(zip(keys, [int_cs, spline_fit, int_cs[1]/num_pix]))
        dPsi[key] = int_cs[1]/num_pix

        plt.figure()
        plt.scatter(x_pts[x_center-dx:x_center+dx], proj_x[x_center-dx:x_center+dx], label="fringe %d data" % key, s=4)
        plt.plot(spline_x_pts[10*(x_center-dx):10*(x_center+dx)],  Interference.make_spline2(spline_x_pts, calData_spline_func, *int_cs)[10*(x_center-dx):10*(x_center+dx)], label="fringe %d fit, pixel shift = %.4f" % (key, int_cs[1]))
        plt.plot(spline_x_pts[10*(x_center-dx-30):10*(x_center+dx+30)], calData_spline_x[10*(x_center-dx-30):10*(x_center+dx+30)], label="0", linestyle=":")
        plt.grid(True)
        plt.legend()
        print("fringe %.1f: amp = %.2f, pix shift = %.4f, baseline = %.2f, dPsi = %.4f"% (key,int_cs[0], int_cs[1], int_cs[2], int_cs[1]/num_pix))
    plt.show()
    print(np.array(dPsi.values()))


    # Now we start the fit to no thin film set:
    print("\n Fit to image that was taken with no thin film ... ")

    # Find the calibration interference pattern first.
    calData = prob_no_sample
    x_center, y_center = VideoEdit.find_center(calData)
    x_center = x_center

    # create a circular mask with the centers first
    circle_img = np.zeros((calData.shape[0], calData.shape[1]))
    cv2.circle(circle_img, (x_center, y_center), dx, 1, thickness=-1)
    circle_img = circle_img.astype(np.int8)

    # Then mask all data with the and operation
    calData = cv2.bitwise_and(calData, calData, mask=circle_img)
    max_val = np.max(VideoEdit.xy_projections(VideoEdit.rotate_img(calData, rot_ang, x_center, y_center))[0])
    calData_x = VideoEdit.xy_projections(VideoEdit.rotate_img(calData, rot_ang, x_center, y_center))[0]/max_val
    calData_spline_func = interp1d(x_pts, calData_x , fill_value="extrapolate", kind=3)
    calData_spline_x = calData_spline_func(spline_x_pts)

    results = collections.OrderedDict()
    dPsi = collections.OrderedDict()

    # Now proceed to fit each image:
    for key, value in grouped_data.items():
        value = cv2.bitwise_and(value, value, mask=circle_img)
        proj_x = VideoEdit.xy_projections(VideoEdit.rotate_img(value, rot_ang, x_center, y_center))[0]/max_val
        currData_spline_func = interp1d(x_pts, proj_x , fill_value="extrapolate", kind=3)
        currData_spline_x = currData_spline_func(spline_x_pts)
        int_cs =  Interference.fitter(spline_x_pts[10*(x_center-dx):10*(x_center+dx)], currData_spline_x[10*(x_center-dx):10*(x_center+dx)], spline=calData_spline_func, mode="spline2")

        spline_fit = Interference.make_spline2(x_pts, calData_spline_func, *int_cs)
        results[key] = dict(zip(keys, [int_cs, spline_fit, int_cs[1]/num_pix]))
        dPsi[key] = int_cs[1]/num_pix

        plt.figure()
        plt.scatter(x_pts[x_center-dx:x_center+dx], proj_x[x_center-dx:x_center+dx], label="fringe %d data" % key, s=4)
        plt.plot(spline_x_pts[10*(x_center-dx):10*(x_center+dx)],  Interference.make_spline2(spline_x_pts, calData_spline_func, *int_cs)[10*(x_center-dx):10*(x_center+dx)], label="fringe %d fit, pixel shift = %.4f" % (key, int_cs[1]))
        plt.plot(spline_x_pts[10*(x_center-dx-30):10*(x_center+dx+30)], calData_spline_x[10*(x_center-dx-30):10*(x_center+dx+30)], label="0", linestyle=":")
        plt.grid(True)
        plt.legend()
        print("fringe %.1f: amp = %.2f, pix shift = %.4f, baseline = %.2f, dPsi = %.4f"% (key,int_cs[0], int_cs[1], int_cs[2], int_cs[1]/num_pix))
    plt.show()
    print(np.array(dPsi.values()))