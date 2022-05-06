import time
import argparse
import cv2
import os
import sys
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from configparser import SafeConfigParser
import threading

from numpy.lib.utils import info
import Camera
from database import DB
from logging.handlers import RotatingFileHandler
import tempfile

import gc
# from memory_profiler import profile





sys.path.append('src')
# OpenSeeFace
from facetracker_deepsort import *
from utils.ImageUtils import *
# OpenCV + CENTROID BASED
from centroidtracker import CentroidTracker

from tensorflow.keras.preprocessing import image

from tracking_deep_sort import track_frame
from yolo5face import yolov5
from scrfd import SCRFD



start = time.time()
np.set_printoptions(precision=2)

logger = logging.getLogger()
formatter = logging.Formatter("(%(threadName)-10s) %(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = RotatingFileHandler("logfile/surveillance.log", maxBytes=10000000, backupCount=10)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

####LOAD CONFIG####
config = SafeConfigParser()
config.read("configs/services.cfg")
LOG_PATH = str(config.get('main', 'LOG_PATH'))
SERVER_IP = str(config.get('main', 'SERVER_IP'))
SERVER_PORT = int(config.get('main', 'SERVER_PORT'))
UPLOAD_VIDEO = str(config.get('storage', 'UPLOAD_VIDEO'))
UPLOAD_FACE = str(config.get('storage', 'UPLOAD_FACE'))
CAFFEMODEL = str(config.get('model', 'CAFFEMODEL'))
PROTOTXT = str(config.get('model', 'PROTOTXT'))
YOLO5FACE = str(config.get('model', 'YOLO5FACE'))
SCRFDMODEL = str(config.get('model', 'SCRFD'))
if not os.path.exists(UPLOAD_FACE):
    os.makedirs(UPLOAD_FACE)

yolonet = yolov5(YOLO5FACE, 0.3, 0.5, 0.3)
scrfdnet = SCRFD(SCRFDMODEL, 0.5, 0.5)

class SurveillanceSystem(object):
    def __init__(self):
        self.trainingEvent = threading.Event() # Used to holt processing while training the classifier 
        self.trainingEvent.set() 
        self.drawing = True 
        self.alarmState = 'Disarmed' # Alarm states - Disarmed, Armed, Triggered
        self.alarmTriggerd = False
        self.alerts = [] # Holds all system alerts
        self.cameras = [] # Holds all system cameras
        self.camerasLock  = threading.Lock() # Used to block concurrent access of cameras []
        self.cameraProcessingThreads = []
        self.id = []
        self.peopleDB = []
        self.confidenceThreshold = 20 # Used as a threshold to classify a person as unknown
        

        # Used for testing purposes
        ###################################
        self.testingResultsLock = threading.Lock()
        self.detetectionsCount = 0
        self.trueDetections = 0
        self.counter = 0
        ####################################
        # processing frame threads 
        for i, cam in enumerate(self.cameras):       
            thread = threading.Thread(name='frame_process_thread_' + str(i),target=self.process_frame,args=(cam,))
            thread.daemon = False
            self.cameraProcessingThreads.append(thread)
            thread.start()
    
    
    def add_camera(self, camera):
        """Adds new camera to the System and generates a 
        frame processing thread"""
        print('/// ADD CAM //')
        self.cameras.append(camera)
        print('self.cameras: ',self.cameras)
        thread = threading.Thread(name='frame_process_thread_' + 
                                 str(len(self.cameras)),
                                 target=self.process_frame,
                                 args=(self.cameras[-1],))
        thread.daemon = False
        self.cameraProcessingThreads.append(thread)
        print('self.cameraProcessingThreads: ',self.cameraProcessingThreads)
        thread.start()
    
    def remove_camera(self, camID):
        """remove a camera to the System and kill its processing thread"""
        print('/// REMOVE CAM //')
        print('self.cameras: ',self.cameras)
        camID = int(camID)
        self.cameras[camID].__del__()
        self.cameras.pop(camID)
        print('self.cameras: ',self.cameras)
        print('self.cameraProcessingThreads: ',self.cameraProcessingThreads)
        self.cameraProcessingThreads.pop(camID)
        print('self.cameraProcessingThreads: ',self.cameraProcessingThreads)
        # camera.captureThread.stop = True
        gc.collect()
        print('/// REMOVE CAM //')

    
    def process_frame(self,camera):
        """This function performs all the frame proccessing.
        It reads frames captured by the IPCamera instance,
        resizes them, and performs 1 of 5 functions"""
        print('da vao process_frame', camera)
        logger.debug('Processing Frames')
        state = 1
        frame_count = 0;  
        first = True
        height = 0
        width = 0
        tracker = None
        FPScount = 0 # Used to calculate frame rate at which frames are being processed
        FPSstart = time.time()
        start = time.time()
        stop = camera.captureThread.stop
        reid = dict()
        current_id = 0

        # initialize our centroid tracker and frame dimensions
        ct = CentroidTracker()
        (H, W) = (None, None)

        while not stop:
            while camera.paused:
                continue
            # print('frame cam:', camera)
            # print('camera.captureThread.stop', camera.captureThread.stop)
            if camera.captureThread.stop:
                break
            ids = []
            list_id = []
            frame_count +=1
            logger.debug("Reading Frame")
            frame = camera.read_frame()
            if frame is None or np.array_equal(frame, camera.tempFrame):  # Checks to see if the new frame is the same as the previous frame
                continue
            # frame = resize(frame)
            height, width, channels = frame.shape

            # Frame rate calculation 
            if FPScount == 6:
                camera.processingFPS = 6/(time.time() - FPSstart)
                FPSstart = time.time()
                FPScount = 0

            FPScount += 1
            camera.tempFrame = frame

            if camera.detectionMethod == "Opencv":
                (H, W) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                camera.net.setInput(blob)
                detections = camera.net.forward()
                rects = []

                for i in range(0, detections.shape[2]):
                    if detections[0, 0, i, 2] > 0.7:
                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        rects.append(box.astype("int"))
                        (x1, y1, x2, y2) = box.astype("int")
                        cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 0), 2)

                objects = ct.update(rects)
                # loop over the tracked objects
                for (objectID, centroid) in objects.items():
                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    cv2.putText(frame,str(objectID),(x1, y1-10), 0, 1, (0, 0, 255), 2)
                    for (i, (x1, y1, x2, y2)) in enumerate(rects):
                        if int((x1 + x2) / 2.0) == centroid[0] and int((y1 + y2) / 2.0) == centroid[1]:
                            name_random = tempfile.NamedTemporaryFile(prefix=str(i)).name.split("/")[-1].replace('_','')
                            x1 = 0 if x1 < 0 else x1
                            y1 = 0 if y1 < 0 else y1
                            x2 = width if x2 > width else x2
                            y2 = height if y2 > height else y2
                            if x1 > width or x2 > width or y1 > height or y2 > height:
                                continue
                            face = frame[y1:y2, x1:x2]
                            if objectID not in camera.id_check.keys(): 
                                camera.id_check[objectID] = [face,-1]
                            elif camera.id_check[objectID][1] == 0: 
                                camera.id_check[objectID][0] = face
                    
                for i, (objectID, centroid) in enumerate(objects.items()):
                    if objectID in camera.id_check.keys() and camera.id_check[objectID][1] == -1:
                        camera.id_check[objectID][1] = 1
                        name_random = tempfile.NamedTemporaryFile(prefix=str(objectID)).name.split("/")[-1].replace('_','')
                        with camera.peopleDictLock:
                            if check_size(frame, camera.id_check[objectID][0], camera.sizeFace):
                                camera.people[name_random] = Person(0.9,camera.id_check[objectID][0],"unknown")
                    
                camera.processing_frame = frame
            elif camera.detectionMethod == "YOLO5face":
                dets = yolonet.detect(frame)
                frame, boxes = yolonet.postprocess(frame, dets)
                track_frame(boxes, frame)
                camera.processing_frame = frame
            elif camera.detectionMethod == "SCRFD":
                frame, boxes = scrfdnet.detect(frame)
                track_frame(boxes, frame)
                camera.processing_frame = frame
            else:
                if first:
                    first = False
                    height, width, channels = frame.shape
                    #sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    if camera.detectionMethod == "OpenSeeFace_m1":
                        tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
                    elif camera.detectionMethod == "OpenSeeFace_m2":
                        tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=4, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
                    elif camera.detectionMethod == "Retinaface":
                        tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=1, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
                faces = tracker.predict_bboxonly(frame)
                list_bbox = []
                face = []
                for face_num, f in enumerate(faces):
                    f = copy.copy(f)
                    box = f.bbox
                    x1, y1 = int(box[0]), int(box[1])
                    x2, y2 = int(box[2]) + x1, int(box[3]) + y1 

                    x1 = 0 if x1 < 0 else x1
                    y1 = 0 if y1 < 0 else y1
                    x2 = width if x2 > width else x2
                    y2 = height if y2 > height else y2

                    index = np.array(frame[y1:y2,x1:x2]).copy()
                    face.append([0,0,index])
                    list_bbox.append([x1,y1,x2,y2])
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
                    f.id += args.face_id_offset
                    ids.append(f.id)
                ### With track OSF
                for id in ids:
                    if id not in reid.keys() or reid[id][0] == -1:
                        current_id += 1
                        reid[id] = [current_id,0]
                for id in reid.keys():
                    if id not in ids:
                        if reid[id][1] == 0:
                            reid[id][1] = time.time()
                        if time.time() - reid[id][1] > 1.5:
                            reid[id] = [-1,0]
                    else:
                        reid[id][1] = 0 
                for i,bbox in enumerate(list_bbox):
                    id = reid[ids[i]][0]
                    list_id.append(id)
                    x1,y1,x2,y2 = bbox
                    cv2.putText(frame,str(id),(x1, y1-10), 0, 1, (0, 0, 255), 2)
                camera.processing_frame = frame

                list_id_cp = list_id.copy()
                for i,id in enumerate(list_id_cp):
                    if id not in camera.id_check.keys(): 
                        camera.id_check[id] = [face[i][2],-1]
                    elif camera.id_check[id][1] == 0: 
                        camera.id_check[id][0] = face[i][2]
                for i,id in enumerate(list_id_cp):
                    if id in camera.id_check.keys() and camera.id_check[id][1] == -1:
                        camera.id_check[id][1] = 1
                        name_random = tempfile.NamedTemporaryFile(prefix=str(id)).name.split("/")[-1].replace('_','')
                        name = "unknown"
                        with camera.peopleDictLock:
                            if check_size(frame, camera.id_check[id][0], camera.sizeFace):
                                camera.people[name_random] = Person(0.9,camera.id_check[id][0],name)
    
    def add_face(self,name,image):
        """Adds face to directory used for training the classifier"""
        database = DB()
        
        if name in database.dist_name_id:
            id_db = database.dist_name_id[name]
        else:
            id_db = tempfile.NamedTemporaryFile().name.split("/")[-1].replace('_', '')

        path_face = os.path.join(UPLOAD_FACE, id_db)

        num = 0
        if not os.path.exists(path_face):
            try:
                logger.info( "Creating New Face Dircectory: " + name)
                os.makedirs(path_face)
            except OSError:
                logger.info( OSError)
                return False
            pass
        else:
            num = len([nam for nam in os.listdir(path_face) if os.path.isfile(os.path.join(path_face, nam))])

        logger.info( "Writing Image To Directory: " + name)
        cv2.imwrite(path_face+"/"+ id_db + "_"+str(num) + ".jpg", image)
        self.get_face_database_names()

        json_path = os.path.join(path_face,id_db + '.json')
        if not os.path.exists(json_path):
            self.create_info(json_path, name, id_db)

        return True

    def get_face_database_names(self):
        """Gets all the names that were most recently 
        used to train the classifier""" 
        self.peopleDB = []
        for name in os.listdir(UPLOAD_FACE):
            if (name == 'cache.t7' or name == '.DS_Store' or name[0:7] == 'unknown'):
                continue
        self.peopleDB.append(name)
        logger.info("Known faces in our db for: " + name + " ")
        self.peopleDB.append('unknown')
    
    def create_info(self, json_path, name, id_random):
        
        json_content = {'name': name, 'id': id_random, 'age': '10', 'gender':'male'}
        with open(json_path, 'w') as out:
            json.dump(json_content, out)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
class Person(object):
    """Person object simply holds all the
    person's information for other processes
    """
    person_count = 0

    def __init__(self,confidence = 0, face = None, name = "unknown"):  

        if "unknown" not in name: # Used to include unknown-N from Database
            self.identity = name
        else:
            self.identity = "unknown"
        self.count = Person.person_count
        self.confidence = confidence  
        self.thumbnails = []
        self.face = face
        if face is not None:
            ret, jpeg = cv2.imencode('.jpg', face) # Convert to jpg to be viewed by client
            self.thumbnail = jpeg.tostring()
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            self.img_re = cv2.resize(face, (224, 224))
            self.img_to_arr = image.img_to_array(self.img_re, dtype=int)
            # print("type imt_to_arr", type(self.img_to_arr))
        self.thumbnails.append(self.thumbnail) 
        Person.person_count += 1 
        now = datetime.now() + timedelta(hours=7)
        self.time = now.strftime("%m/%d/%Y, %I:%M:%S%p")
        self.istracked = False

        self.images_array = []
        self.images_array.append(self.img_to_arr)

    def set_identity(self, identity):
        self.identity = identity

    def set_time(self): # Update time when person was detected
        now = datetime.now() + timedelta(hours=7)
        self.time = now.strftime("%m/%d/%Y, %I:%M:%S%p")

    def set_thumbnail(self, face):
        self.face = face
        ret, jpeg = cv2.imencode('.jpg', face) # Convert to jpg to be viewed by client
        self.thumbnail = jpeg.tostring()

    def add_to_thumbnails(self, face):
        ret, jpeg = cv2.imencode('.jpg', face) # Convert to jpg to be viewed by client
        self.thumbnails.append(jpeg.tostring())

class list_Tracker:
    """Keeps track of person position"""
    tracker_count = 0

    def __init__(self, person):
        self.person = person


