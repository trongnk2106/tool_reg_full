import threading
import time
import numpy as np
import cv2
from utils import ImageUtils
import os
import argparse
import logging
import SurveillanceSystem

# from memory_profiler import profile

logger = logging.getLogger(__name__)

CAPTURE_HZ = 30.0 # Determines frame rate at which frames are captured from IP camera

class IPCamera(object):
	"""The IPCamera object continually captures frames
	from a camera and makes these frames available for
	proccessing and streamimg to the web client. A 
	IPCamera can be processed using 5 different processing 
	functions detect_motion, detect_recognise, 
	motion_detect_recognise, segment_detect_recognise, 
	detect_recognise_track. These can be found in the 
	SureveillanceSystem object, within the process_frame function"""

	def __init__(self,camURL, fpsTweak, detectionMethod, sizeFace, net):
		logger.info("Loading Stream From IP Camera: " + camURL)
		self.processing_frame = None
		self.tempFrame = None
		self.captureFrame  = None
		self.streamingFPS = 0 # Streaming frame rate per second
		self.processingFPS = 0
		self.FPSstart = time.time()
		self.FPScount = 0
		self.detectionMethod = detectionMethod
		self.sizeFace = int(sizeFace)
		self.motion = False # Used for alerts and transistion between system states i.e from motion detection to face detection
		self.people = {} # Holds person ID and corresponding person object 
		self.trackers = [] # Holds all alive trackers
		self.fpsTweak = fpsTweak # used to know if we should apply the FPS work around when you have many cameras
		self.rgbFrame = None
		self.faceBoxes = None
		self.captureEvent = threading.Event()
		self.captureEvent.set()
		self.peopleDictLock = threading.Lock() # Used to block concurrent access to people dictionary
		self.id_check = {}
		self.video = cv2.VideoCapture(camURL) # VideoCapture object used to capture frames from IP camera
		# self.video = cv2.VideoCapture('storage/upload-video/aa.mp4')
		# self.video = cv2.VideoCapture('/home/huytn/a_Thua/home_surveillance/system/testing/iphoneVideos/peopleTest.m4v')
		logger.info("We are opening the video feed.")
		self.url = camURL
		if not self.video.isOpened():
			self.video.open()
		logger.info("Video feed open.")
		# Start a thread to continuously capture frames.
		# The capture thread ensures the frames being processed are up to date and are not old
		self.captureLock = threading.Condition(threading.Lock()) # Sometimes used to prevent concurrent access
		self.captureThread = threading.Thread(name='video_captureThread',target=self.get_frame)
		self.captureThread.daemon = True
		self.captureThread.stop = False
		self.captureThread.start()
		self.paused = False
		self.random_key_stream = None
		self.net = net

	def __del__(self):
		self.video.release()

	# @profile
	def get_frame(self):
		logger.debug('Getting Frames')
		FPScount = 0
		warmup = 0
		#fpsTweak = 0  # set that to 1 if you want to enable Brandon's fps tweak. that break most video feeds so recommend not to
		FPSstart = time.time()

		while True:
			if self.captureThread.stop:
				break
			success, frame = self.video.read()
			self.captureEvent.clear() 
			if success:		
				self.captureFrame  = frame
				# print('self.captureFrame  get_frame', self.captureFrame[0][0])
				self.captureEvent.set() 
				# self.captureEvent.wait()
				# print('self.captureEvent.wait()', self.captureEvent.wait())


			FPScount += 1 

			if FPScount == 5:
				self.streamingFPS = 5/(time.time() - FPSstart)
				FPSstart = time.time()
				FPScount = 0

			if self.fpsTweak:
				if self.streamingFPS != 0:  # If frame rate gets too fast slow it down, if it gets too slow speed it up
					if self.streamingFPS > CAPTURE_HZ:
						time.sleep(1/CAPTURE_HZ)
					else:
						time.sleep(self.streamingFPS/(CAPTURE_HZ*CAPTURE_HZ))

	def read_jpg(self):
		"""We are using Motion JPEG, and OpenCV captures raw images,
		so we must encode it into JPEG in order to stream frames to
		the client. It is nessacery to make the image smaller to
		improve streaming performance"""

		capture_blocker = self.captureEvent.wait() 
		frame = self.captureFrame 	
		frame = ImageUtils.resize_mjpeg(frame)
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tostring()

	def read_frame(self):
		capture_blocker = self.captureEvent.wait()  
		frame = self.captureFrame 
		# print('self.captureFrame  read_frame', self.captureFrame[0][0])	
		return frame

	def read_processed(self):
		frame = None
		with self.captureLock:
			frame = self.processing_frame	
		while frame is None: # If there are problems, keep retrying until an image can be read.
			with self.captureLock:	
				frame = self.processing_frame
				# print('frame = self.processing_frame',self.processing_frame[0][0])

		frame = ImageUtils.resize_mjpeg(frame)
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tostring()
