import cv2
import numpy as np
import libjevois as jevois
import math
from enum import Enum
import json


class PowerRectangularPrismTracker:
    def __init__(self):
        self.timer = jevois.Timer("sandbox", 100, jevois.LOG_INFO)
        self.HORIZONTAL_FIELD_OF_VIEW = 50
        self.CAMERA_HORIZONTAL_RESOLUTION = 320

        #######################
        # Constants from GRIP #
        #######################
        self.__normalize_type = cv2.NORM_MINMAX
        self.__normalize_alpha = 0.0
        self.__normalize_beta = 255.0

        self.normalize_output = None

        self.__blur_input = self.normalize_output
        self.__blur_type = BlurType.Box_Blur
        self.__blur_radius = 7.207207207207207

        self.blur_output = None

        self.__hsv_threshold_input = self.blur_output
        self.__hsv_threshold_hue = [32.37410071942446, 62.342954159592516]
        self.__hsv_threshold_saturation = [107.77877697841726, 181.4006791171477]
        self.__hsv_threshold_value = [96.31294964028777, 255.0]

        self.hsv_threshold_output = None


        self.__mask_mask = self.hsv_threshold_output

        self.mask_output = None

        self.__find_contours_input = self.hsv_threshold_output
        self.__find_contours_external_only = False

        self.find_contours_output = None

        self.__filter_contours_contours = self.find_contours_output
        self.__filter_contours_min_area = 14.0
        self.__filter_contours_min_perimeter = 1.0
        self.__filter_contours_min_width = 2.0
        self.__filter_contours_max_width = 1000.0
        self.__filter_contours_min_height = 2.0
        self.__filter_contours_max_height = 1000.0
        self.__filter_contours_solidity = [0.0, 100.0]
        self.__filter_contours_max_vertices = 1000000.0
        self.__filter_contours_min_vertices = 0.0
        self.__filter_contours_min_ratio = 0.0
        self.__filter_contours_max_ratio = 1000.0

        self.filter_contours_output = None

        ##############################
        # End of Constants From GRIP #
        ##############################


    def process(self, inframe, outframe):
        # Get the next camera image (may block until it is captured) and here convert it to OpenCV BGR by default. If
        # you need a grayscale image instead, just use getCvGRAY() instead of getCvBGR(). Also supported are getCvRGB()
        # and getCvRGBA():
        source0 = inimg = inframe.getCvBGR()


        #jevois.LINFO("Input image is {} {}x{}".format(jevois.fccstr(inimg.fmt), inimg.width, inimg.height))
        outimg = inimg = inframe.getCvBGR()

        self.timer.start()

        #############
        # GRIP CODE #
        #############

        # Step Normalize0:
        self.__normalize_input = source0
        (self.normalize_output) = self.__normalize(self.__normalize_input, self.__normalize_type, self.__normalize_alpha, self.__normalize_beta)

        # Step Blur0:
        self.__blur_input = self.normalize_output
        (self.blur_output) = self.__blur(self.__blur_input, self.__blur_type, self.__blur_radius)

        # Step HSV_Threshold0:
        self.__hsv_threshold_input = self.blur_output
        (self.hsv_threshold_output) = self.__hsv_threshold(self.__hsv_threshold_input, self.__hsv_threshold_hue, self.__hsv_threshold_saturation, self.__hsv_threshold_value)

        # Step Mask0:
        self.__mask_input = source0
        self.__mask_mask = self.hsv_threshold_output
        (self.mask_output) = self.__mask(self.__mask_input, self.__mask_mask)

        # Step Find_Contours0:
        self.__find_contours_input = self.hsv_threshold_output
        (self.find_contours_output) = self.__find_contours(self.__find_contours_input, self.__find_contours_external_only)

        # Step Filter_Contours0:
        self.__filter_contours_contours = self.find_contours_output
        (self.filter_contours_output) = self.__filter_contours(self.__filter_contours_contours, self.__filter_contours_min_area, self.__filter_contours_min_perimeter, self.__filter_contours_min_width, self.__filter_contours_max_width, self.__filter_contours_min_height, self.__filter_contours_max_height, self.__filter_contours_solidity, self.__filter_contours_max_vertices, self.__filter_contours_min_vertices, self.__filter_contours_min_ratio, self.__filter_contours_max_ratio)

        #################
        # END GRIP CODE #
        #################

        ###############
        # Custom Code #
        ###############
        contours_by_size = self.sortByArea(self.filter_contours_output)
        is_first_contour = True
        for contour in contours_by_size:
            if is_first_contour:
                moment = cv2.moments(contour)
                cX = int(moment["m10"] / moment["m00"])
                cY = int(moment["m01"] / moment["m00"])
                cv2.circle(outimg, (cX, cY), 7, (255, 255, 255), -1)
                # jevois.LINFO("aX: {} x: {} y: {}".format(self.calculateOffset(cX, cY)[0], cX, cY))
                aX, aY = self.calculateOffset(cX, cY)
                aX, aY = int(aX), int(aY)
                info_string = "{aX};{aY}".format(aX=aX, aY=aY)
                jevois.sendSerial("JVTI:" + info_string)
            is_first_contour = False
        """
        if (len(self.filter_contours_output) > 0):
            biggest_contour = self.sortByArea(self.filter_contours_output)[0]
            moment = cv2.moments(biggest_contour)
            cX = int(moment["m10"] / moment["m00"])
            cY = int(moment["m01"] / moment["m00"])
            cv2.circle(outimg, (cX, cY), 7, (255, 255, 255), -1)
            jevois.LINFO("x: {} y: {}".format(cX, cY))
            # Draws Red Line around Outline of Contour
            #cv2.drawContours(outimg, [biggest_contour], -1, (0, 0, 255), 1)
            # Gets the (rotated) bounding box of the Contour
            #rect = cv2.minAreaRect(biggest_contour)
            #box = cv2.boxPoints(rect)
            #cv2.drawContours(outimg, [box], 0, (0, 0, 255), 2)
            #jevois.LINFO(biggest_contour.)
        """
        # Draws all contours on original image in red
        cv2.drawContours(outimg, self.filter_contours_output, -1, (0, 0, 255), 1)
        fps = self.timer.stop()
        height, width, channels = outimg.shape
        cv2.putText(outimg, fps, (3, height - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        outframe.sendCvBGR(outimg)

        """

        inimg = inframe.get()
        jevois.LINFO("Input image is {} {}x{}".format(jevois.fccstr(inimg.fmt), inimg.width, inimg.height))

        outimg = outframe.get()
        jevois.LINFO("Output image is {} {}x{}".format(jevois.fccstr(outimg.fmt), outimg.width, outimg.height))

        # Example of getting pixel data from the input and copying to the output:
        jevois.paste(inimg, outimg, 0, 0)

        # We are done with the input image:
        inframe.done()

        # Example of in-place processing:
        #jevois.hFlipYUYV(outimg)

        # Example of simple drawings:
        #jevois.drawCircle(outimg, int(outimg.width/2), int(outimg.height/2), int(outimg.height/2.2),
        #                  2, jevois.YUYV.White)
        #jevois.writeText(outimg, "Hi from Python!", 20, 20, jevois.YUYV.White, jevois.Font.Font10x20)

         # We are done with the output, ready to send it to host over USB:
        outframe.send()
        """

    def parseSerial(self, str):
        jevois.LINFO("parseserial received command [{}]".format(str))
        if str == "hello":
            return self.hello()
        return "ERR: Unsupported command"

    def supportedCommands(self):
        # use \n seperator if your module supports several commands
        return "hello - print hello using python"

    def hello(self):
        return "Hello from python!"

    ######################
    # Our Methods #
    ######################
    def getArea(self, con): # Gets the area of the contour
            return cv2.contourArea(con)

    def getYcoord(self, con): # Gets the Y coordinate of the contour
        M = cv2.moments(con)
        cy = int(M['m01']/M['m00'])
        return cy

    def getXcoord(self, con): # Gets the X coordinate of the contour
        M = cv2.moments(con)
        cy = int(M['m10']/M['m00'])
        return cy

    def sortByArea(self, conts) : # Returns an array sorted by area from smallest to largest
        contourNum = len(conts) # Gets number of contours
        sortedBy = sorted(conts, key=self.getArea) # sortedBy now has all the contours sorted by area
        return sortedBy

    def calculateOffset(self, x, y): # Returns a tuple (x, y) angle offset
        """
        This assumes that y starts at the top of the frame and increases in value as it goes "down".
        This also assumes that x starts at the left of the frame and increases in value as it goes to the right.
        """
        horizontal_pixels_per_degree = (self.CAMERA_HORIZONTAL_RESOLUTION / 2) / (self.HORIZONTAL_FIELD_OF_VIEW / 2)
        cX = x - (self.CAMERA_HORIZONTAL_RESOLUTION / 2)
        aX = cX / horizontal_pixels_per_degree
        return (aX, 0)


    ################################
    # Start of GRIP Static Methods #
    ################################

    @staticmethod
    def __normalize(input, type, a, b):
        """Normalizes or remaps the values of pixels in an image.
        Args:
            input: A numpy.ndarray.
            type: Opencv enum.
            a: The minimum value.
            b: The maximum value.
        Returns:
            A numpy.ndarray of the same type as the input.
        """
        return cv2.normalize(input, None, a, b, type)

    @staticmethod
    def __blur(src, type, radius):
        """Softens an image using one of several filters.
        Args:
            src: The source mat (numpy.ndarray).
            type: The blurType to perform represented as an int.
            radius: The radius for the blur as a float.
        Returns:
            A numpy.ndarray that has been blurred.
        """
        if(type is BlurType.Box_Blur):
            ksize = int(2 * round(radius) + 1)
            return cv2.blur(src, (ksize, ksize))
        elif(type is BlurType.Gaussian_Blur):
            ksize = int(6 * round(radius) + 1)
            return cv2.GaussianBlur(src, (ksize, ksize), round(radius))
        elif(type is BlurType.Median_Filter):
            ksize = int(2 * round(radius) + 1)
            return cv2.medianBlur(src, ksize)
        else:
            return cv2.bilateralFilter(src, -1, round(radius), round(radius))

    @staticmethod
    def __hsv_threshold(input, hue, sat, val):
        """Segment an image based on hue, saturation, and value ranges.
        Args:
            input: A BGR numpy.ndarray.
            hue: A list of two numbers the are the min and max hue.
            sat: A list of two numbers the are the min and max saturation.
            lum: A list of two numbers the are the min and max value.
        Returns:
            A black and white numpy.ndarray.
        """
        out = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        return cv2.inRange(out, (hue[0], sat[0], val[0]),  (hue[1], sat[1], val[1]))

    @staticmethod
    def __mask(input, mask):
        """Filter out an area of an image using a binary mask.
        Args:
            input: A three channel numpy.ndarray.
            mask: A black and white numpy.ndarray.
        Returns:
            A three channel numpy.ndarray.
        """
        return cv2.bitwise_and(input, input, mask=mask)

    @staticmethod
    def __find_contours(input, external_only):
        """Sets the values of pixels in a binary image to their distance to the nearest black pixel.
        Args:
            input: A numpy.ndarray.
            external_only: A boolean. If true only external contours are found.
        Return:
            A list of numpy.ndarray where each one represents a contour.
        """
        if(external_only):
            mode = cv2.RETR_EXTERNAL
        else:
            mode = cv2.RETR_LIST
        method = cv2.CHAIN_APPROX_SIMPLE
        im2, contours, hierarchy =cv2.findContours(input, mode=mode, method=method)
        return contours

    @staticmethod
    def __filter_contours(input_contours, min_area, min_perimeter, min_width, max_width,
                        min_height, max_height, solidity, max_vertex_count, min_vertex_count,
                        min_ratio, max_ratio):
        """Filters out contours that do not meet certain criteria.
        Args:
            input_contours: Contours as a list of numpy.ndarray.
            min_area: The minimum area of a contour that will be kept.
            min_perimeter: The minimum perimeter of a contour that will be kept.
            min_width: Minimum width of a contour.
            max_width: MaxWidth maximum width.
            min_height: Minimum height.
            max_height: Maximimum height.
            solidity: The minimum and maximum solidity of a contour.
            min_vertex_count: Minimum vertex Count of the contours.
            max_vertex_count: Maximum vertex Count.
            min_ratio: Minimum ratio of width to height.
            max_ratio: Maximum ratio of width to height.
        Returns:
            Contours as a list of numpy.ndarray.
        """
        output = []
        for contour in input_contours:
            x,y,w,h = cv2.boundingRect(contour)
            if (w < min_width or w > max_width):
                continue
            if (h < min_height or h > max_height):
                continue
            area = cv2.contourArea(contour)
            if (area < min_area):
                continue
            if (cv2.arcLength(contour, True) < min_perimeter):
                continue
            hull = cv2.convexHull(contour)
            solid = 100 * area / cv2.contourArea(hull)
            if (solid < solidity[0] or solid > solidity[1]):
                continue
            if (len(contour) < min_vertex_count or len(contour) > max_vertex_count):
                continue
            ratio = (float)(w) / h
            if (ratio < min_ratio or ratio > max_ratio):
                continue
            output.append(contour)
        return output

BlurType = Enum('BlurType', 'Box_Blur Gaussian_Blur Median_Filter Bilateral_Filter')
