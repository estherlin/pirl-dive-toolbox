import numpy as np
import cv2
import csv
import os
import re
from collections import OrderedDict
import matplotlib.pylab as plt
import sys
from scipy.optimize import curve_fit
from collections import OrderedDict

NUM_PIX_EQ_100_UM = 91 # Number of pixels for 100um. Measured with a syringe. 
AMBIENT_AIR_DENSITY = 1.225

TXT_REGEX = re.compile(
    r'File = (?P<filename>\d{8}\.\d{6}\.\d{3} delay=\d+ [a-z]+)\nPk-Pk\(\d\) =  [-+]?\d*\.\d+\nDelay\(\dR\-\dR\) =  (?P<delay>[-+]?\d*\.\d+E[-+]?\d*)\nAmplitude\(\d\) =  [-+]?\d*\.\d+E*[-+]?\d*\nArea \- Full Screen\(\d\) =  [-+]?\d*\.\d+E*[-+]?\d*\n*',
    re.IGNORECASE
)
BMP_REGEX = re.compile(
    r'.*(?P<date_time>\d{8}\.\d{6}\.\d{3}) delay=\d{7} (?P<material>[a-z]+)',
    re.IGNORECASE
)

def radii(t, a, b, c):
    """
    Function for calculating radius of shockwave from Frederik's 2018 paper
    Radius r = xi*E^0.2*t^0.4/rho^0.2

    Parameters
    ----------
    t: time [s] - np.array
    a: epsilon - float
    b: E [J] - float
    c: exponent - float

    Returns
    -------
    np.array
        array of radii in [m] at each given time point t

    """
    return a*np.power(b/AMBIENT_AIR_DENSITY,1/(2+c))*np.power(t,2/(2+c)) # m

def velocity(t,a,b,c):
    """
    Function for calculating radius of shockwave from Frederik's 2018 paper
    Radius r = xi*E^0.2*t^0.4/rho^0.2

    Parameters
    ----------
    t: time [s] - np.array
    a: epsilon - float
    b: E [J] - float
    c: exponent - float

    Returns
    -------
    np.array
        array of velocities [m/s] at each given time point t

    """
    return a*(2/(2+c))*np.power(b/AMBIENT_AIR_DENSITY, 1/(2+c))*np.power(t, (-1)*c/(2+c)) #m/s


def extract_timing(filename):
    """Extracts timing info for each frame

    Parameters
    ----------
    filename : string
        name of excel file to open
    Returns
    -------
    dict
        map of frame to it's delay timing

    """
    timing_data = {}
    workbook = xlrd.open_workbook(filename)

    sh = workbook.sheet_by_index(0)
    for rownum in range(1, sh.nrows):
        row_values = sh.row_values(rownum)
        timing_data[int(row_values[0])] = row_values[2]

    return timing_data

def add_timing_to_img(mainDir, img_names, timing_data,  date, material, makeVid=False):
    """
    Adds a footer containing timing and other info to each image

    Parameters
    ----------
    vidname : string
        name of video file
    timing_data : dict
        dict mapping each frame to delay time

    Returns
    -------
    dirname: string
        full name of directory containing the processed images

    """
    # Step 1: get image parameters
    img = cv2.imread(os.path.join(mainDir,img_names[0])+".bmp")
    height, width, channels = img.shape

    # text parameters
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (int(width*0.075), int(height*0.90))
    bottomRightCornerOfText= (int(width*0.825), int(height*0.90))
    topLeftCornerOfText    = (int(width*0.075), int(height*0.1))
    topRightCornerOfText   = (int(width*0.825), int(height*0.1))
    fontScale              = 1.5
    fontColor              = (100, 100, 100)
    fontColor2             = (1, 1, 1)
    lineType               = 3

    # make a folder to hold results
    dirName = os.path.join(mainDir, "processed/")

    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

    print("Processing images...")

    for name in img_names:
        img = cv2.imread(os.path.join(mainDir,name)+".bmp")

        # make footer
        zeros = np.zeros((int(height*0.2), width, channels))
        cv2.putText(zeros,'sample: %s' % material,
            (20,int(height*0.2/4 + 10)),
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.putText(zeros,'date: %s' % date,
            (20, int(height*0.2/2 + 20)),
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.putText(zeros,'fluence: 2.7 J/cm^2',
            (20, int(height*0.2*3/4 + 30)),
            font,
            fontScale,
            fontColor,
            lineType)
        zeros[int(height*0.2/4 + 10)-30:int(height*0.2/4 + 10), 1150:1150+NUM_PIX_EQ_100_UM] = fontColor
        cv2.putText(zeros,'100um',
            (1250, int(height*0.2/4 + 10)),
            font,
            fontScale,
            fontColor,
            lineType)
        cv2.putText(zeros,'delay=%.1fns' % timing_data.get(name),
            (1150, int(height*0.2/2 + 20)),
            font,
            fontScale,
            fontColor,
            lineType)
        frame = np.vstack((img, zeros))
        cv2.imwrite(dirName+name+'.png', frame)
    print('Finished processing images')
    return dirName


def img2vid(timing_file, imgDir):
    """
    Make a mp4 video from a collection of images, 
    ordered from shortest to longest delay timing

    Parameters
    ----------
    timing_file : string
        full name of video file
    imgDir : string
        full name of directory containing the images
    """

    img_names, date, material = get_img_filenames(imgDir)
    print(imgDir, img_names[0])

    frame = cv2.imread(os.path.join(imgDir,img_names[0]+'.png'))
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(os.path.join(imgDir,'output.mp4'), fourcc, 1, (width, height))

    # Read in timing file
    with open(timing_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[1] in img_names and float(row[0]) >= 0:
                print("Processing delay=%.1f, file %s" % (float(row[0]), row[1]))
                image_path = os.path.join(imgDir,row[1])+".png"
                frame = cv2.imread(image_path)

                out.write(frame) # Write out frame to video

                if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                    break

            # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()


def get_img_filenames(dir):
    """
    Gets the image filenames in a directoryself.
    Directory has to be the lowest level

    Parameters
    ----------
    dir : string
        directory to start search

    Returns
    ----------
    filename : string
        list of all of the files in directory that match BMP_REGEX
    """
    filenames = [f.replace('.bmp', '').replace('.png', '') for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and "reference" not in f and ".bmp" in f or ".png" in f and "./." not in f]

    match = BMP_REGEX.match(filenames[0])
    if match:
        img_date = match.group('date_time')
        img_material = match.group('material')
        return filenames, img_date, img_material

    return filenames, "00/00/0000", "none"

def get_txt_filenames(dir):
    """
    Gets the txt filenames in a directory.
    Directory has to be the lowest level

    Parameters
    ----------
    dir : string
        directory to start search

    Returns
    ----------
    filename : string
        list of all of the txt filenames in directory
    """
    filenames = [os.path.join(dir,f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f)) and ".txt" in f and "./." not in f]
    return filenames

def map_name2delay(dir, filenames, save=False):
    """
    Map an image to its delay time, save in a csv file.
    Note: the units of time are [ns] 
          the units of distance are [um]

    Parameters
    ----------
    dir : string
        directory to start search
    filnames : list of strings
        names of txt files to open
    save: boolean
        true: saves map to a csv file named 'timing.csv' in dir

    Returns
    ----------
    data_sorted : dict
        map of delay time to image
    """
    img2delay = {}

    for filename in filenames:
        print("Reading in file %s" % filename)
        f = open(filename, 'r')
        if f.mode == 'r':
            contents = f.read().split(os.linesep + os.linesep)

            for content in contents:
                match = TXT_REGEX.match(content)
                if match:
                    img_name = match.group('filename')
                    img_delay = (1.896e-6 + float(match.group('delay')))*1e9
                    img2delay[img_name] = img_delay
    data_sorted = {k: v for k, v in sorted(img2delay.items(), key=lambda x: x[1])}
    if save:
        myFile = open(os.path.join(dir,'timing.csv'), 'w')
        with myFile:
            writer = csv.writer(myFile)
            for key, value in data_sorted.items():
                writer.writerow([value, key])

    return data_sorted

def make2vids(imgDir1, timingFile1, imgDir2, timingFile2):
    """
    Concatenates two videos horizontally to play them simultaneously

    Parameters
    ----------
    imgDir1 : string
        directory to start search for image set 1
    timingFile1 : string
        filename of csv file containing the map for delay time to image filename
    imgDir2 : string
        directory to start search for image set 1
    timingFile2 : string
        filename of csv file containing the map for delay time to image filename
    """

    # Get the img names from both dirs
    img_names1, date, material = get_img_filenames(imgDir1)
    img_names2, date, material = get_img_filenames(imgDir2)

    print(imgDir1, imgDir2)

    # Get the info from both csvs and filter out the negative times
    with open(timingFile1, 'r') as f:
        reader = csv.reader(f)
        csv1   = [row for row in list(reader) if float(row[0]) >= 0]
    with open(timingFile2, 'r') as f:
        reader = csv.reader(f)
        csv2   = [row for row in list(reader) if float(row[0]) >= 0]

    # Get image frame parameters, assume same for both videos
    frame = cv2.imread(os.path.join(imgDir1,img_names1[0])+".png")
    #cv2.imshow('video',frame)
    height, width, channels = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
    out = cv2.VideoWriter(os.path.join("./",'combined.mp4'), fourcc, 1, (width*2, height))

    # Check which data set has more images
    if len(csv2) >= len(csv1):
        print("Processing, %s has less images" % imgDir1)
        total = len(csv1)
        count = 0

        for row1 in csv1:
            if row1[1] in img_names1:
                count = count + 1
                print("Processing %d / %d ..."% (count, total))
                print("Processing delay=%.1f, file %s" % (float(row1[0]), row1[1]))

                # Load in the first file
                image_path1 = os.path.join(imgDir1,row1[1])+".png"
                frame1 = cv2.imread(image_path1)

                # Find valid frames from vid2
                validFrames = [row for row in csv2 if row[1] in img_names2 and round(float(row[0]),1) <= round(float(row1[0]),1)]

                if len(validFrames) == 0:
                    print("No valid frames")
                    pass

                for row2 in csv2:
                    if row2[1] in img_names2 and float(row2[0]) <= float(row1[0]):
                        image_path2 = os.path.join(imgDir2,row2[1]+".png")
                        frame2 = cv2.imread(image_path2)



                        final_frame = np.hstack((frame1, frame2))

                        out.write(final_frame)

                        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                            break

                # remove valid Frames from original listdir
                csv2 = [row for row in csv2 if row not in validFrames]

    else:
        print("Processing, %s has less images" % imgDir2)
        total = len(csv2)
        count = 0

        for row2 in csv2:
            if row2[1] in img_names2:
                count = count + 1
                print("Processing %d / %d ..."% (count, total))
                print("Processing delay=%.1f, file %s" % (float(row2[0]), row2[1]))

                # Load in the first file
                image_path2 = os.path.join(imgDir2,row2[1]+".png")
                frame2 = cv2.imread(image_path2)

                # Find valid frames from vid2
                validFrames = [row for row in csv1 if row[1] in img_names1 and round(float(row[0]),1) <= round(float(row2[0]),1)]

                if len(validFrames) == 0:
                    print("No valid frames")
                    pass

                for row1 in csv1:
                    if row1[1] in img_names1 and float(row1[0]) <= float(row2[0]):

                        image_path1 = os.path.join(imgDir1,row1[1]+".png")
                        print(image_path1)
                        frame1 = cv2.imread(image_path1)

                        final_frame = np.hstack((frame1, frame2))

                        out.write(final_frame)

                        if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
                            break

                # remove valid Frames from original listdir
                csv1 = [row for row in csv1 if row not in validFrames]

    out.release()
    cv2.destroyAllWindows()
    print("Concatenated video completed")



if __name__ == '__main__':
    """
    Sample usage of this file. 
    You can run this file from the command line with 
    ```
    python plumes.py sampleSet1Directory sampleSet2Directory
    ```
    This will process the two data sets, create a csv timing sheet
    for each, create a video for each, and make a concatenated 
    video of the two. 
    """
    # Loop through and analyze both image sets
    for i in range (1,len(sys.argv)):

        # Get directory name
        sampleSet = sys.argv[i]

        # Extract the image names, date and material from the images
        img_names, date, material = get_img_filenames(sampleSet)

        # Find the txt files that contain the filenames of the images
        txt_names = get_txt_filenames(sampleSet)

        # Map an image to its delay time, save in a csv file that is
        # sorted from lowest to largest delay time, map [ns : um]
        im2delay = map_name2delay(sampleSet, txt_names, save=True)

        # Add the info footer to each image and put them in a new directory
        processedDir = add_timing_to_img(sampleSet, 
                                        img_names, 
                                        im2delay, 
                                        date, 
                                        material)
                                        
        # Sort all the images from shortest to longest delay time 
        # and combine to create a video with 1FPS
        img2vid(os.path.join(sampleSet, "timing.csv"), processedDir)

    # If you want to make a video that is the horizontal concatenation of two separate videos
    make2vids(os.path.join(sys.argv[1], "processed/"), 
              os.path.join(sys.argv[1], "timing.csv"), 
              os.path.join(sys.argv[2], "processed/"), 
              os.path.join(sys.argv[2], "timing.csv"))


