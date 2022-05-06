import cv2
import numpy as np
import os
import glob
import sys
import argparse
import math
import threading
import time
start = time.time()

def resize(frame):
    r = 640.0 / frame.shape[1]
    dim = (640, int(frame.shape[0] * r))
    # Resize frame to be processed
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)    
    return frame 

def resize_mjpeg(frame):
    r = 320.0 / frame.shape[1]
    dim = (320, 200)#int(frame.shape[0] * r))
    # perform the actual resizing of the image and show it
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)    
    return frame  

def crop(image, box, dlibRect = False):

    if dlibRect == False:
       x, y, w, h = box
       return image[y: y + h, x: x + w] 

    return image[box.top():box.bottom(), box.left():box.right()]

# check size of bbox larger than pecentage of image
def check_size(image, face, percent):
    percent = percent/100
    if face.shape[1] > (percent * image.shape[1]):
        return True
    return False