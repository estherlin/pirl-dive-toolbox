#!/usr/bin/env python

# Library imports needed for all processing in this file
import numpy as np
import matplotlib.pyplot as plt
import videoparser as vp
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.stats import chisquare
from joblib import Parallel, delayed
import timeit
import re
from os import listdir
from os.path import isfile, join
import collections
#from lmfit import Model
from bokeh.plotting import figure, show, output_file

# Filename regular expressions
FILENAME_REGEX = re.compile(
    r'.*(?P<date>\d+)\_(?P<material>[a-z]+)\_(?P<wavelength_nm>\d+)\_(?P<angle_deg>[-\d]+[\.]*\d*).avi',
    re.IGNORECASE
)


class Interference(object):
    """
    Makes an interference object for analysis
    """

    # Constructor for initialization
    def __init__ (self, filename, x_proj, x_axis, time=0):
        """
        Creates an interference object. One interference object is needed
        for any change, either angle or time
        """

        # Extract parameters from filename
        match = FILENAME_REGEX.match(filename)

        if match:
            self.exp_date   = match.group('date')
            self.material   = match.group('material')
            self.wavelength = float(match.group('wavelength_nm'))      # in nm
            self.exp_angle  = float(match.group('angle_deg'))          # in degrees
        else:
            self.exp_date   = "none"
            self.material   = "none"
            self.wavelength = float(1480)      # in nm
            self.exp_angle  = 0                # in degrees

        # More input Parameters
        self.time        = time
        self.x_data      = x_proj
        self.x_axis      = x_axis
        self.num_pts_x   = self.x_data.size

        # Calibration parameters
        self.dx = 50

        # Create spline points from subtracted data
        self.spline_x_func = interp1d(self.x_axis, self.x_data , fill_value="extrapolate", kind=3)
        self.spline_x_axis = np.linspace(np.min(self.x_axis), np.max(self.x_axis), self.x_axis.size*10)
        self.spline_x      = self.spline_x_func(self.spline_x_axis)


    @staticmethod
    def normalize(v):
        """
        :param v: vector to be normalized
        :return: normalized version of v
        """
        #norm = np.linalg.norm(v)
        norm = np.max(v)
        if norm == 0:
           return v
        return v / norm

    @staticmethod
    def spline_func(spline):
        """
        Fits to a given spline.
        :spline: spline to fit to
        :c1: amplitude
        :c2: phase shift
        :c3: intensity translation
        """
        def create_spline(x, c1):
            return spline(x-c1)
        return create_spline

    @staticmethod
    def make_spline(x, spline, c1):
        """
        Fits to a given spline.
        :spline: spline to fit to
        :x:
        :c1: amplitude
        :c2: phase shift
        :c3: intensity translation
        """
        return spline(x-c1)

    @staticmethod
    def spline_func2(spline):
        """
        Fits to a given spline.
        :spline: spline to fit to
        :c1: amplitude
        :c2: phase shift
        :c3: amplitudetranslation
        """
        def create_spline(x, c1, c2, c3):
            return c1*spline(x-c2)+c3
        return create_spline

    @staticmethod
    def make_spline2(x, spline, c1, c2, c3):
        """
        Fits to a given spline.
        :spline: spline to fit to
        :x:
        :c1: amplitude
        :c2: phase shift
        :c3: amplitude translation
        """
        return c1*spline(x-c2)+c3

    @staticmethod
    def fit_func(x, c1, c2, c3, c4, c5, c6, c7):
        """
        function for fitting
        :param1: x - the domain
        :param2-8: the coefficients for the functions
        :returns: function to fit to
        """
        return c1*np.exp(-c2*np.power(x-c3, 2.))+c4*np.cos(c5*x+c6) + c7

    @staticmethod
    def gaussian(x, c1, c2, c3, c4):
        """
        Returns a gaussian function
        :c1: amplitude
        :c2: width
        :c3: center
        :c4: baseline offest
        """
        return c1*np.exp(-1*np.power(x-c3, 2.)/c2) + c4

    @staticmethod
    def two_gaussian(x, c1, c2, c3, c4, c5, c6, c7):
        """
        Returns the sum of two gaussian functions
        :param c1: amplitude 1
        :param c2: width 1
        :param c3: center 1
        :param c4: amplitude 2
        :param c5: width 2
        :param c6: center 2
        :param c7: amplitude offset
        """
        return c1*np.exp(-1*np.power(x-c3, 2.)/c2) + c4*np.exp(-1*np.power(x-c6, 2.)/c5) + c7


    @staticmethod
    def gauss_cosine(x, c1, c2, c3, c4, c5, c6, c7):
        """
        Returns a cosine function with a gaussian as the amplitude
        :c1: amplitude
        :c2: width
        :c3: center
        :c4: amplitude of cosine
        :c5: angular frequency
        :c6: phase
        :c7: offset - should be zero though
        """
        return c1*np.exp(-1*np.power(x-c3, 2.)/c2)+np.cos(c4*x+c5) + c6

    @staticmethod
    def fitter(axis, data, mode, spline=None, low_bound=-12.00, p=None):
        """
        Fitting done here, along with approximations
        popt, pcov = curve_fit(Interference.fit_func, axis, data, bounds=bounds)
        :axis: the axis with which we are fitting the data with
        :data: the data we are fitting
        :mode: choose between gaussian fitting, etc.
        :returns: the coefficients of the fit
        """

        if mode == "gaussian":
            #init_vals = [3e6, 250000, 150, 1.2e6]
            #bounds = ((0, 100, 150, 0), (4.5e6, np.inf, 600, 3e6))
            #init_vals = [3e6, 250000, 150, 3e6, 250000, 150, 1.2e6]
            #bounds = ((0, 100, 100, 0, 100, 100, 0), (4.5e6, np.inf, 250, 4.5e6, np.inf, 250, 3e6))
            #popt, pcov = curve_fit(Interference.gaussian, axis, data)
            #popt, pcov = curve_fit(Interference.gaussian, axis, data, bounds=bounds, p0=init_vals)
            #cs = np.around(popt,1)
            #plt.plot(axis, Interference.gaussian(axis, *popt), label="%s" % cs)
            gmodel = Model(Interference.gaussian)
            result = gmodel.fit(data, x=axis, c1=1, c2=250, c3=250, c4=1)

            print(result.fit_report())
            #print([result.params[i][1].value for i in range(0,len(result.params))])

            #plt.plot(axis, data, 'bo')
            #plt.plot(axis, result.init_fit, 'k--')
            #plt.plot(axis, result.best_fit, 'r-')
            #plt.show()
            cs = [result.params['c1'].value, result.params['c2'].value, result.params['c3'].value, result.params['c4'].value,]

        elif mode == "gauss_cosine":
            init_vals = [3e6, 2500, 150, 0.1, 1, 1e6, 1e6]
            bounds = ((1e6, 1500, 100, 0, 0, 1e6, 0.5e6), (4.5e6, 5000, 250, 0.5, 10, 3e6, 1.5e6))
            popt, pcov = curve_fit(Interference.gauss_cosine, axis, data, bounds=bounds)
            cs = np.around(popt,1)
            #plt.plot(axis, Interference.gauss_cosine(axis, *popt), label="%s" % cs)

        elif mode == "spline":
            init_vals = [-1]
            bounds = ((0), (10))
            popt, pcov = curve_fit(Interference.spline_func(spline), axis, data, bounds=bounds, p0=init_vals)
            cs = popt
        elif mode == "spline2":
            if p is not None:
                init_vals = p[0]
                bounds = p[1]
            else:
                #init_vals = [1, -1, 0]
                bounds = ((0.99,0, -0.2), (1.01,10, 0.2))
            popt, pcov = curve_fit(Interference.spline_func2(spline), axis, data, bounds=bounds)
            cs = popt
        else:
            cs = 0
        return cs

    @staticmethod
    def residuals(spline, data, data_axis):
        """
        Calculates the residuals of two sets
        :spline: standard, can either be a function or an np.array
        :data: data that is to be compared to the standard
        :data_axis: axis of the data inputed. If spline is array, size of axis and spline should be equivalent
        :returns: an array of the residuals (subtraction) and TODO: the chi squared value
        """

        if callable(spline):
            resid = spline(data_axis) - data
        else:
            try:
                resid = spline - data
            except ValueError:
                print("Data needs to have the same shape")
                resid = np.nan
        return resid

class AggregateInterferences(object):
    """
    Aggregates a bunch of interferences together.
    Needs to take in the no sample video, and then a list of names of other videos.
    """

    def __init__ (self, calibrationVidName, leftName, rightName, filenames, n_jobs=2, bin_num=1, num_pix=None, angle=None, fit="spline", bounds=False):
        """
        """
# Determine the angle to rotate by
        if angle is not None:
            self.angle = angle
        else:
            self.angle = AggregateInterferences.find_rot_angle(leftName, rightName, calibrationVidName, n_jobs=n_jobs)
        #self.angle=-1.40
        print("Optimal rotation angle found to be: %.2f degrees \n" % self.angle)

        # Load in the individual beam profile projections
        print("Loading in left and right beam profiles for adjustments...")
        leftArmVid  = vp.VideoEdit(leftName, cut=False, angle=self.angle)
        rightArmVid = vp.VideoEdit(rightName, cut=False, angle=self.angle)
        leftArmProj  = Interference.normalize(vp.VideoEdit.xy_projections(leftArmVid.bin_img[0])[0])
        rightArmProj = Interference.normalize(vp.VideoEdit.xy_projections(rightArmVid.bin_img[0])[0])

        # Find the mean
        self.meanProj = Interference.normalize((leftArmProj + rightArmProj))

        print("Loading in video files and applying fits...")
        # Parameters that need to be the same as the calibration video
        #self.calVid = vp.VideoEdit(calibrationVidName, angle=self.angle)
        self.calVid = vp.VideoEdit(calibrationVidName, cut=False, angle=self.angle)
        self.x_center = self.calVid.x_center
        self.bin = self.calVid.bin

        # Other common parameters
        self.x_pts = np.linspace(1, self.calVid.xshape, self.calVid.xshape)

        # Create interference object for the calibration video
        proj_x= Interference.normalize(vp.VideoEdit.xy_projections(self.calVid.bin_img[0])[0])
        offset = np.mean(np.concatenate((np.subtract(self.meanProj[0:100], proj_x[0:100]), \
            np.subtract(self.meanProj[-101:-1], proj_x[-101:-1]))))/np.sqrt(2)
        self.calInt = Interference(calibrationVidName, \
            Interference.normalize(proj_x - self.meanProj + offset), self.x_pts)

        # Reference angle. Can be the calibration video angle
        self.ref_angle = self.calInt.exp_angle

        # Determine number of pixels per wavelength
        self.num_pix = self.num_pix_per_lambda()
        self.filenames = filenames

        # Parallelize this yay!
        if fit == "spline":
            self.data = Parallel(n_jobs,verbose=5,backend='threading')\
                (delayed(AggregateInterferences.start)(self.calVid, self.calInt, filenames[i], 0, self.num_pix, self.meanProj) for i in range(0,len(filenames)))
            print("\n".join(["%s-> pixel shift:%.3f" % (int.get("angle"), int.get("cs")[0]) for int in self.data]))

        else:
            self.data = Parallel(n_jobs,verbose=0,backend='threading')\
                (delayed(AggregateInterferences.start_data)(self.calVid, self.calInt, filenames[i], 0, self.num_pix, self.meanProj, bounds) for i in range(0,len(filenames)))
            print("\n".join(["%s-> amp: %.3f, pixel shift:%.3f, height: %.3f" % (int.get("angle"), int.get("cs")[0], int.get("cs")[1], int.get("cs")[2]) for int in self.data]))

    def num_pix_per_lambda(self):
        """
        Determine the number of pixels for each wavelength
        """
        # TODO: DON'T HARDCODE THIS
        x_axis = self.calInt.spline_x_axis[10*self.x_center-500:10*self.x_center+500]
        x_spline = self.calInt.spline_x[10*self.x_center-500:10*self.x_center+500]
        zero_crossings = np.where(np.diff(np.sign(x_spline)))[0]
        x_axis = x_axis[zero_crossings[0]:zero_crossings[-1]]
        x_spline = x_spline[zero_crossings[0]:zero_crossings[-1]]

        num_half_lambda = zero_crossings.size - 1
        return 2*x_axis.size/(10*num_half_lambda)

    @staticmethod
    def start(calVid, calInt, filename, time, num_pix, meanProj):
        """
        start up procedures in a static function for parallelization
        Procedure: create current video, and video interference
                  find spline fit parameters and generate fit solutions
        :param calVid: the calibration video
        :param calInt: the calibration interference pattern
        :param filename: the name of the file we want to load in
        :param time: current time step
        :return: a dict of the parameters
        """
        # Read in video object for each experiment
        currVid = vp.VideoEdit(filename, cut=False, angle=calVid.angle, x_center=calVid.x_center, y_center=calVid.y_center)
        proj_x = Interference.normalize(vp.VideoEdit.xy_projections(currVid.bin_img[time])[0])

        # Modify the projection
        offset = np.mean(np.concatenate((np.subtract(meanProj[0:100], proj_x[0:100]), \
            np.subtract(meanProj[-101:-1], proj_x[-101:-1]))))/np.sqrt(2)
        currInt = Interference(filename, Interference.normalize(proj_x - meanProj + offset), calVid.x_pts)

        # Fit to calibrationVid
        int_cs =  Interference.fitter(calInt.spline_x_axis[(calVid.x_center-calInt.dx)*10:(calVid.x_center+calInt.dx)*10], currInt.spline_x[(calVid.x_center-calInt.dx)*10:(calVid.x_center+calInt.dx)*10], spline=calInt.spline_x_func, mode="spline")
        spline_fit = Interference.make_spline(calVid.x_pts, calInt.spline_x_func, *int_cs)

        # Find the thickness from the pizel shift
        dPsi = int_cs[0]/num_pix
        ns = np.linspace(1.4, 1.6, 11).tolist()
        ts = np.array([AggregateInterferences.calc_t(n, currInt.wavelength, dPsi, currInt.exp_angle, 0) for n in ns])

        return dict(zip(["angle", "int", "cs", "spline", "dPsi", "ts"],
                        [currInt.exp_angle, currInt, int_cs, spline_fit, dPsi, ts]))

    @staticmethod
    def start_data(calVid, calInt, filename, time, num_pix, meanProj, bounds):
        """
        start up procedures in a static function for parallelization
        Procedure: create current video, and video interference
                  find spline fit parameters and generate fit solutions
        :param calVid: the calibration video
        :param calInt: the calibration interference pattern
        :param filename: the name of the file we want to load in
        :param time: current time step
        :return: a dict of the parameters
        """
        currVid = vp.VideoEdit(filename, cut=False, angle=calVid.angle, x_center=calVid.x_center, y_center=calVid.y_center)
        proj_x = Interference.normalize(vp.VideoEdit.xy_projections(currVid.bin_img[time])[0])

        # Modify the projection
        offset = np.mean(np.concatenate((np.subtract(meanProj[0:100], proj_x[0:100]), \
            np.subtract(meanProj[-101:-1], proj_x[-101:-1]))))/np.sqrt(2)
        currInt = Interference(filename, Interference.normalize(proj_x - meanProj + offset), calVid.x_pts)

        # Apply fit on the middle 100
        if bounds:
            int_cs =  Interference.fitter(calVid.x_pts[375:475], currInt.x_data[375:475], spline=calInt.spline_x_func, mode="spline2")
            spline_fit = Interference.make_spline2(calVid.x_pts, calInt.spline_x_func, *int_cs)
        else:
            int_cs =  Interference.fitter(calVid.x_pts[calVid.x_center-calInt.dx:calVid.x_center+calInt.dx], currInt.x_data[calVid.x_center-calInt.dx:calVid.x_center+calInt.dx], spline=calInt.spline_x_func, mode="spline2")
            spline_fit = Interference.make_spline2(calVid.x_pts, calInt.spline_x_func, *int_cs)

        # Find the thickness from the pixel shift
        dPsi = int_cs[1]/num_pix
        ns = np.linspace(1.4, 1.6, 11).tolist()
        ts = np.array([AggregateInterferences.calc_t(n, currInt.wavelength, dPsi, currInt.exp_angle, calInt.exp_angle) for n in ns])

        return dict(zip(["angle", "int", "cs", "spline", "dPsi", "ts"],
                        [currInt.exp_angle, currInt, int_cs, spline_fit, dPsi, ts]))


    @staticmethod
    def rot_ints(leftName, rightName, calibrationVidName, angle):
        """
        Determines the chisquared value of a video file for a given angle
        Only looks at the middle 100 points
        :param leftName: filename of the left beam profile video
        :param rightName: filename of the right beam profile video
        :param calibrationVidName: The calibration video file name
        :param angle: angle to rotate by
        :return: interference pattern's data and gaussian for x
        """
        vid = vp.VideoEdit(calibrationVidName, angle=angle)

        leftArmVid  = vp.VideoEdit(leftName, cut=False, angle=angle)
        rightArmVid = vp.VideoEdit(rightName, cut=False, angle=angle)
        leftArmProj  = Interference.normalize(vp.VideoEdit.xy_projections(leftArmVid.bin_img[0])[0])
        rightArmProj = Interference.normalize(vp.VideoEdit.xy_projections(rightArmVid.bin_img[0])[0])
        meanProj = Interference.normalize((leftArmProj + rightArmProj))

        # Create interference object for the calibration video
        proj_x= Interference.normalize(vp.VideoEdit.xy_projections(vid.bin_img[0])[0])
        offset = np.mean(np.concatenate((np.subtract(meanProj[0:100], proj_x[0:100]), \
            np.subtract(meanProj[-101:-1], proj_x[-101:-1]))))/np.sqrt(2)
        intf = Interference(calibrationVidName, \
            Interference.normalize(proj_x - meanProj + offset), vid.x_pts)

        #intf = Interference(calibrationVidName, proj_x[100:-100], vid.x_pts[100:-100])

        return chisquare(intf.x_data.tolist(), f_exp=intf.gaussian_x.tolist())

    @staticmethod
    def find_rot_angle(leftName, rightName, calibrationVidName, n_jobs=2):
        """
        Determines the optimal angle so that the fringes are perfectly vertical
        :param leftName: filename of the left beam profile video
        :param rightName: filename of the right beam profile video
        :param calibrationVidName: The calibration video file name
                                   Have to have array sizes of over 250 elements
        :return: angle
        """
        print("Determining which angle to rotate videos in range of 2 to -2 degrees...")

        # Angles to try:
        angles = np.linspace(-2.0, -0.2, 20).tolist()

        data = Parallel(n_jobs,verbose=5,backend='threading')\
                (delayed(AggregateInterferences.rot_ints)(leftName, rightName, calibrationVidName, angle) for angle in angles)
        chisq = np.array([result[0] for result in data])
        for i in range(0, len(angles)):
            print("angle %.2f: chisq %.2f \n" % (angles[i], chisq[i]/100000))
        return angles[np.argmax(chisq)]

    @staticmethod
    def calc_t(n, lambda_air, dPhase, a1, a2):
        """
        Calculates the thickness t for the given parameters
        :param n: index of refraction of material
        :param lambda_air: wavelength of laser in air in nm
        :param dPhase: phase change
        :param a1: angle 1 in degrees
        :param a2: angle 2 - if one of the angles is zero, make a2 zero: a2>=a1
        :return: t - thickness in um
        """
        if a1 == 0 and a2 == 0:
            theta = np.deg2rad(a1)
            return (1e-3)*(dPhase)*lambda_air/((n/np.cos(np.arcsin(np.sin(theta)/n)))-1)
        elif a2 == 0:
            theta = np.deg2rad(a1)
            #print(n, lambda_air, theta, dPhase, a1)
            part1 = (1+np.sin(theta-np.arcsin(np.sin(theta)/n)))/np.cos(np.arcsin(np.sin(theta)/n))
            #return (1e-3)*(dPhase)*(lambda_air/n)/(1/np.cos(np.arcsin(np.sin(theta)/n)) - 1)
            return (1e-3)*(dPhase)*(lambda_air/n)/(part1-1)
        else:

            a = np.deg2rad(a1)
            a_ref = np.deg2rad(a2)

            theta2 = np.arcsin(np.sin(a)/n)
            ref_theta_2 = np.arcsin(np.sin(a_ref)/n)

            part1 = (n + n*np.sin(a - theta2))/(np.cos(theta2))
            part2 = (n + n*np.sin(a_ref - ref_theta_2))/(np.cos(ref_theta_2))
            s = (1e-3)*(dPhase)*lambda_air/(part1 - part2)
            return s
            """
            theta1 = np.deg2rad(a1)
            theta2 = np.deg2rad(a2)
            return (1e-3)*(dPhase)*lambda_air/((n/np.cos(np.arcsin(np.sin(theta1)/n)))-(n/np.cos(np.arcsin(np.sin(theta2)/n))))
            """
            #return (1e-3)*dPhase*lambda_air/(n*((1/np.cos(np.arcsin(np.sin(theta1)/n)))-(1/np.cos(np.arcsin(np.sin(theta2)/n)))))

    @staticmethod
    def find_n(exp, num_pix):
        """
        Find n of the material by finding the lowest std
        :param exp: experiment object
        :param num_pix: number of pixels in one wavelength
        :return: optimal n value
        """
        # Calculate the 4 ts for an array of n
        ns = np.linspace(1.4, 1.6, 11).tolist()

        mean = []
        std = []
        print("num pixels per lambda: %.2f" %num_pix)

        all_ts = np.array([exp.data[i].get("ts") for i in range(0,len(exp.filenames))])

        for row in range(0, len(ns)):
            values = all_ts[:,row]
            mean.append(np.mean(values))
            std.append(np.std(values))
            print("n %.2f: %s, std: %.2f, mean: %.2f \n" % (ns[row], values, std[row], mean[row]))

        #std = [np.std(np.array([t1[i], t2[i], t3[i], t4[i]])) for i in range(0, len(ns))]
        #mean = [np.mean(np.array([t1[i], t2[i], t3[i], t4[i]])) for i in range(0, len(ns))]
        #std = [np.std(np.array([t1[i], t2[i], t3[i]])) for i in range(0, len(ns))]
        #mean = [np.mean(np.array([t1[i], t2[i], t3[i]])) for i in range(0, len(ns))]

        #std = [np.std(np.array([t1[i], t3[i], t4[i]])) for i in range(0, len(ns))]
        #mean = [np.mean(np.array([t1[i], t3[i], t4[i]])) for i in range(0, len(ns))]
        #calcs = [dict(zip(["0->0", "43->0", "53->0", "63->0"], [t1[i], t2[i], t3[i], t4[i]])) for i in
                 #range(0, len(ns))]
        #calcs = [dict(zip(["43->0", "53->0", "63->0"], [t1[i], t2[i], t3[i]])) for i in
                 #range(0, len(ns))]
        #calcs = [dict(zip(["0->0", "53->0", "63->0"], [t1[i], t3[i], t4[i]])) for i in
                 #range(0, len(ns))]
        #data = dict(zip(ns, calcs))

        #for i in range(0, len(ns)):
            #print("n %.2f: %s, std: %.2f, mean: %.2f \n" % (ns[i], calcs[i], std[i], mean[i]))

        return ns[np.argmin(std)]


    @staticmethod
    def get_filenames(dir):
        """
        Gets the files in a directoryself.Directory has to be the lowest level
        :param dir: directory to start searchself.
        :returns filename: list of all of the files in directory that match FILENAME_REGEX
        """
        filenames = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
        filenames.remove(join(dir, "calibration.bmp"))
        filenames.remove(join(dir, "labbook.txt"))
        return filenames.sort()


if __name__== "__main__":
    np.set_printoptions(precision=3)

    # Get filenames
    videoDir            = "/Volumes/Seagate Backup Plus Drive/mpsd_docs/data/30082018_2/"
    leftName         = videoDir+"30082018_left_beam_profile.avi"
    rightName        = videoDir+"30082018_right_beam_profile.avi"
    calibrationVidName  = videoDir+"30082018_prolene_1458_0.avi"
    filenames = [
        "/Volumes/Seagate Backup Plus Drive/mpsd_docs/data/30082018_2/30082018_prolene_1458_10.avi",
        "/Volumes/Seagate Backup Plus Drive/mpsd_docs/data/30082018_2/30082018_prolene_1458_20.avi",
        "/Volumes/Seagate Backup Plus Drive/mpsd_docs/data/30082018_2/30082018_prolene_1458_30.avi",
        "/Volumes/Seagate Backup Plus Drive/mpsd_docs/data/30082018_2/30082018_prolene_1458_40.avi",
        "/Volumes/Seagate Backup Plus Drive/mpsd_docs/data/30082018_2/30082018_prolene_1458_45.avi",
        "/Volumes/Seagate Backup Plus Drive/mpsd_docs/data/30082018_2/30082018_prolene_1458_50.avi",
        "/Volumes/Seagate Backup Plus Drive/mpsd_docs/data/30082018_2/30082018_prolene_1458_55.avi",
        "/Volumes/Seagate Backup Plus Drive/mpsd_docs/data/30082018_2/30082018_prolene_1458_60.avi"
    ]

    exp = AggregateInterferences(calibrationVidName, leftName, rightName, filenames, angle=-1.5, fit="spline")
    print(exp.num_pix)

    # Start Plotting all of them I guess
    for i in range(0, len(filenames)):
        plt.figure()
        plt.scatter(exp.calInt.x_axis[350:500], exp.data[i].get("int").x_data[350:500], label="angle %.2f original data" % exp.data[i].get("angle"), s=8)
        plt.plot(exp.calInt.x_axis[350:500], exp.data[i].get("spline")[350:500], label="angle %.2f fit to angle %.2f spline, pixel shift = %.2f" % (exp.data[i].get("angle"), exp.calInt.exp_angle, exp.data[i].get("cs")[0]), linestyle=':', linewidth=1)
        plt.plot(exp.calInt.spline_x_axis[3000:5000], exp.calInt.spline_x[3000:5000], label="angle %.2f original spline" % exp.calInt.exp_angle, linestyle=':', linewidth=1)
        plt.grid(True)
        plt.legend()
        plt.xlabel("x pixel")
        plt.ylabel("intensity")
        plt.show()
    plt.xlabel("x pixel")
    plt.ylabel("intensity")
    plt.show()

    """
    # Make videos out of all of them
    leftArmVid  = vp.VideoEdit(leftName, cut=False, angle=-1.5)
    rightArmVid = vp.VideoEdit(rightName, cut=False, angle=-1.5)
    calVid      = vp.VideoEdit(calibrationVidName, cut=False, angle=-1.5)

    # find projections for both and plot
    leftArmProj  = Interference.normalize(vp.VideoEdit.xy_projections(leftArmVid.bin_img[0])[0])
    rightArmProj = Interference.normalize(vp.VideoEdit.xy_projections(rightArmVid.bin_img[0])[0])
    calImgProj   = Interference.normalize(vp.VideoEdit.xy_projections(calVid.bin_img[0])[0])
    #subImgProj   = Interference.normalize(vp.VideoEdit.xy_projections(subImg)[0])
    #meanImgProj  = Interference.normalize(vp.VideoEdit.xy_projections(meanImg)[0])
    #sumImgProj   = Interference.normalize(vp.VideoEdit.xy_projections(sumImg)[0])
    meanImgProj = Interference.normalize((leftArmProj + rightArmProj))
    sumImgProj = (leftArmProj + rightArmProj)
    offset = np.mean(np.concatenate((np.subtract(meanImgProj[0:100], calImgProj[0:100]), \
        np.subtract(meanImgProj[-101:-1], calImgProj[-101:-1]))))/np.sqrt(2)
    #offset = np.mean(np.subtract(meanImgProj[0:100], calImgProj[0:100]))
    subImgProj = (calImgProj - meanImgProj + offset)

    # Plot this shit
    x = calVid.x_pts[200:-200]
    plt.plot(x, leftArmProj[200:-200], label="Left Beam Profile Projection")
    plt.plot(x, rightArmProj[200:-200], label="Right Beam Profile Projection")
    plt.plot(x, calImgProj[200:-200], label="0 Degrees Interference Projection")
    plt.plot(x, calImgProj[200:-200] + offset, label="0 Degrees Interference Projection + Offset")
    plt.title("Plots of the Projections for the Beam Profiles and Interference")
    #plt.plot(x, meanImgProj[100:-100], label="Mean Beam Profile Projection")
    #plt.plot(x, sumImgProj[100:-100], label="Sum Beam Profile Projection")
    plt.legend()
    plt.grid(True)
    plt.xlabel("pixels")
    plt.ylabel("normalized intensity")
    plt.figure()
    plt.plot(x, subImgProj[200:-200], label="Adjusted Interference Projection")
    plt.legend()
    plt.grid(True)
    plt.title("Adjusted Interference Projection")
    plt.xlabel("pixels")
    plt.ylabel("normalized intensity")
    plt.show()
    """
    """
    x = calVid.x_pts
    y = calVid.y_pts
    xx, yy = np.meshgrid(x, y)

    # CONSTANTS
    TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset,save"
    # Variables of data
    p = figure(tools = TOOLS,
               x_range=(0,int(np.max(x))),
               y_range=(0,int(np.max(y))),
               plot_width=int(np.max(x)),
               plot_height=int(np.max(y)),
               toolbar_location="above",
               title="original Interference Image")

    p.image(image=[calVid.bin_img[0]], x=0, y=0, dw=int(np.max(x)), dh=int(np.max(y)), palette="Greys9")
    show(p)
    """
