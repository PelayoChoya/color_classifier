#!/usr/bin/env python

import os
import sys
import cv2
import numpy as np

class pixel_operations:

    def __init__(self):
        self.__pixel_colors = []
        # check if results directory exists
        # if not create it
        self.__dataset_path = os.path.dirname(os.path.abspath(__file__)) + \
            '/dataset'
        self.__create_directory()
        self.recording_running = False

    def __create_directory(self):
        if not os.path.exists(self.__dataset_path):
            os.makedirs(self.__dataset_path)

    def __write_statistics(self):
        with open(self.__dataset_path + '/'  + self.color + '.txt', 'a') as results:
            np.savetxt(results, self.__pixel_colors, fmt =
                  '%i')

    def __pixel_coordinates(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.__pixel_colors.append(self.image_lab[y][x])

    def setup_recording(self, color):
        self.__capt = cv2.VideoCapture(0)
        self.color = color
        cv2.namedWindow('frame',cv2.WINDOW_AUTOSIZE)
        # set mouse callback function
        cv2.setMouseCallback('frame', self.__pixel_coordinates)
        self.recording_running = True

    def terminate_recording(self):
        self.__capt.release()
        cv2.destroyAllWindows()
        self.__write_statistics()
        self.recording_running = False
        del self.__pixel_colors[:]

    def record_camera(self):
        image_params = self.__capt.read()
        self.image_lab = cv2.cvtColor(image_params[1], cv2.COLOR_BGR2LAB)
        if image_params[0] == True:
            cv2.imshow('frame', image_params[1])

def error_checking(user_input):
   if not user_input:
       raise ValueError('No arguments passed')
   else:
        valid_options = ['red','green','blue','fucshia','orange','yellow']
        for color in user_input:
           if color not in valid_options:
               raise ValueError(color + ' is not a valid argument')

if __name__ == '__main__':
    dataset_colors = sys.argv[1:]
    try:
        error_checking(dataset_colors)
    except ValueError:
        error_type, error_instance, traceback = sys.exc_info()
        error_instance.args = (error_instance.args[0] + ' <Valid:  \
                               red,green,blue,fucshia,orange,yellow>',)
        raise error_type, error_instance, traceback
    print 'Press ESC to exit'
    print 'Mouse Left Click to store image pixel position'
    px = pixel_operations()
    # get results for each of the specified colors
    for color in dataset_colors:
        print 'Recording results for ' + color + ' color'
        px.setup_recording(color)
        while(px.recording_running):
            px.record_camera()
            if cv2.waitKey(1) > -1:
                px.terminate_recording()
