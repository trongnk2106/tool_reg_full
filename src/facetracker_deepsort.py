import copy
import os
import sys
import argparse
import traceback
import gc
from tracker import *
from access_camera import *
# from keras.models import load_model
# import tensorflow as tf
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--ip", help="Set IP address for sending tracking data", default="127.0.0.1")
parser.add_argument("-p", "--port", type=int, help="Set port for sending tracking data", default=11573)
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras", type=int, help="Set this to 1 to list the available cameras and quit, set this to 2 or higher to output only the names", default=0)
    parser.add_argument("-a", "--list-dcaps", type=int, help="Set this to -1 to list all cameras and their available capabilities, set this to a camera id to list that camera's capabilities", default=None)
    parser.add_argument("-W", "--width", type=int, help="Set camera and raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set camera and raw RGB height", default=360)
    parser.add_argument("-F", "--fps", type=int, help="Set camera frames per second", default=15)
    parser.add_argument("-D", "--dcap", type=int, help="Set which device capability line to use or -1 to use the default camera settings", default=None)
    parser.add_argument("-B", "--blackmagic", type=int, help="When set to 1, special support for Blackmagic devices is enabled", default=0)
else:
    parser.add_argument("-W", "--width", type=int, help="Set raw RGB width", default=640)
    parser.add_argument("-H", "--height", type=int, help="Set raw RGB height", default=360)
parser.add_argument("-c", "--capture", type=int, help="Set camera ID (0, 1...) or video file", default="0")
parser.add_argument("-m", "--max-threads", type=int, help="Set the maximum number of threads", default=1)
parser.add_argument("-t", "--threshold", type=float, help="Set minimum confidence threshold for face tracking", default=None)
parser.add_argument("-d", "--detection-threshold", type=float, help="Set minimum confidence threshold for face detection", default=0.6)
parser.add_argument("-v", "--visualize", type=int, help="Set this to 1 to visualize the tracking, to 2 to also show face ids, to 3 to add confidence values or to 4 to add numbers to the point display", default=0)
parser.add_argument("-P", "--pnp-points", type=int, help="Set this to 1 to add the 3D fitting points to the visualization", default=0)
parser.add_argument("-s", "--silent", type=int, help="Set this to 1 to prevent text output on the console", default=1)
parser.add_argument("--faces", type=int, help="Set the maximum number of faces (slow)", default=5)
parser.add_argument("--scan-retinaface", type=int, help="When set to 1, scanning for additional faces will be performed using RetinaFace in a background thread, otherwise a simpler, faster face detection mechanism is used. When the maximum number of faces is 1, this option does nothing.", default=0)
parser.add_argument("--scan-every", type=int, help="Set after how many frames a scan for new faces should run", default=3)
parser.add_argument("--discard-after", type=int, help="Set the how long the tracker should keep looking for lost faces", default=1)
parser.add_argument("--max-feature-updates", type=int, help="This is the number of seconds after which feature min/max/medium values will no longer be updated once a face has been detected.", default=900)
parser.add_argument("--no-3d-adapt", type=int, help="When set to 1, the 3D face model will not be adapted to increase the fit", default=1)
parser.add_argument("--try-hard", type=int, help="When set to 1, the tracker will try harder to find a face", default=0)
parser.add_argument("--video-out", help="Set this to the filename of an AVI file to save the tracking visualization as a video", default=None)
parser.add_argument("--video-scale", type=int, help="This is a resolution scale factor applied to the saved AVI file", default=1, choices=[1,2,3,4])
parser.add_argument("--video-fps", type=float, help="This sets the frame rate of the output AVI file", default=24)
parser.add_argument("--raw-rgb", type=int, help="When this is set, raw RGB frames of the size given with \"-W\" and \"-H\" are read from standard input instead of reading a video", default=0)
parser.add_argument("--log-data", help="You can set a filename to which tracking data will be logged here", default="")
parser.add_argument("--log-output", help="You can set a filename to console output will be logged here", default="")
parser.add_argument("--model", type=int, help="This can be used to select the tracking model. Higher numbers are models with better tracking quality, but slower speed, except for model 4, which is wink optimized. Models 1 and 0 tend to be too rigid for expression and blink detection. Model -2 is roughly equivalent to model 1, but faster. Model -3 is between models 0 and -1.", default=1, choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir", help="This can be used to specify the path to the directory containing the .onnx model files", default=None)
parser.add_argument("--gaze-tracking", type=int, help="When set to 1, experimental blink detection and gaze tracking are enabled, which makes things slightly slower", default=1)
parser.add_argument("--face-id-offset", type=int, help="When set, this offset is added to all face ids, which can be useful for mixing tracking data from multiple network sources", default=0)
parser.add_argument("--repeat-video", type=int, help="When set to 1 and a video file was specified with -c, the tracker will loop the video until interrupted", default=0)
parser.add_argument("--dump-points", type=str, help="When set to a filename, the current face 3D points are made symmetric and dumped to the given file when quitting the visualization with the \"q\" key", default="")
parser.add_argument("--benchmark", type=int, help="When set to 1, the different tracking models are benchmarked, starting with the best and ending with the fastest and with gaze tracking disabled for models with negative IDs", default=0)
parser.add_argument("--method",type=str, help="Choice methods detection", default="OSF")
parser.add_argument("--mask", help="Choice mask detection", default=False)

if os.name == 'nt':
    parser.add_argument("--use-dshowcapture", type=int, help="When set to 1, libdshowcapture will be used for video input instead of OpenCV", default=1)
    parser.add_argument("--blackmagic-options", type=str, help="When set, this additional option string is passed to the blackmagic capture library", default=None)
    parser.add_argument("--priority", type=int, help="When set, the process priority will be changed", default=None, choices=[0, 1, 2, 3, 4, 5])
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

class OutputLog(object):
    def __init__(self, fh, output):
        self.fh = fh
        self.output = output
    def write(self, buf):
        if not self.fh is None:
            self.fh.write(buf)
        self.output.write(buf)
        self.flush()
    def flush(self):
        if not self.fh is None:
            self.fh.flush()
        self.output.flush()
output_logfile = None
if args.log_output != "":
    output_logfile = open(args.log_output, "w")
sys.stdout = OutputLog(output_logfile, sys.stdout)
sys.stderr = OutputLog(output_logfile, sys.stderr)

if os.name == 'nt':
    import dshowcapture
    if args.blackmagic == 1:
        dshowcapture.set_bm_enabled(True)
    if not args.blackmagic_options is None:
        dshowcapture.set_options(args.blackmagic_options)
    if not args.priority is None:
        import psutil
        classes = [psutil.IDLE_PRIORITY_CLASS, psutil.BELOW_NORMAL_PRIORITY_CLASS, psutil.NORMAL_PRIORITY_CLASS, psutil.ABOVE_NORMAL_PRIORITY_CLASS, psutil.HIGH_PRIORITY_CLASS, psutil.REALTIME_PRIORITY_CLASS]
        p = psutil.Process(os.getpid())
        p.nice(classes[args.priority])


import numpy as np
import time
import cv2
import socket
import struct
import json
# from input_reader import InputReader, VideoReader, DShowCaptureReader, try_int
from tracker import Tracker, get_model_base_path
# from sort import Sort

def track_deep_sort():
    log = None
    out = None
    first = True
    height = 0
    width = 0
    tracker = None
    sock = None
    total_tracking_time = 0.0
    tracking_time = 0.0
    tracking_frames = 0
    frame_count = 0
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        start = time.time()

        attempt = 0
        need_reinit = 0
        frame_count += 1
        now = time.time()
        if first:
            first = False
            height, width, channels = frame.shape
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
        inference_start = time.perf_counter()
        faces = tracker.predict_bboxonly(frame)

        packet = bytearray()
        detected = False
        list_face_bbox = []
        for face_num, f in enumerate(faces):
            f = copy.copy(f)
            box = f.bbox
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2]) + x1
            y2 = int(box[3]) + y1
            list_face_bbox.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
            f.id += args.face_id_offset
            detected = True
        end = time.time()
        
        fps = str(int(1/(end-start)))
        cv2.putText(frame, fps,(5,25),
        cv2.FONT_HERSHEY_SIMPLEX,1,  # font scale
        (255, 0, 255),3)
        frame = track_frame(list_face_bbox,frame)
        cv2.imshow('OpenSeeFace Visualization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def track_sort():
    log = None
    out = None
    first = True
    height = 0
    width = 0
    tracker = None
    sock = None
    total_tracking_time = 0.0
    tracking_time = 0.0
    tracking_frames = 0
    frame_count = 0
    cap = cv2.VideoCapture(0)

    ids = []
    track_sort = Sort()

    while cap.isOpened():
        ret, frame = cap.read()
        attempt = 0
        need_reinit = 0
        frame_count += 1
        start = time.time()
        if first:
            first = False
            height, width, channels = frame.shape
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
        inference_start = time.perf_counter()
        faces = tracker.predict_bboxonly(frame)

        packet = bytearray()
        detected = False
        list_face_bbox = []
        for face_num, f in enumerate(faces):
            f = copy.copy(f)
            box = f.bbox
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2]) + x1
            y2 = int(box[3]) + y1
            list_face_bbox.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
            f.id += args.face_id_offset
            detected = True
        ## Sort Tracking
        predict=track_sort.update(np.array(list_face_bbox))

        for pre in predict:
            x1, y1, x2, y2,id=int(pre[0]),int(pre[1]),int(pre[2]),int(pre[3]),int(pre[4])
            if frame_count>2:
                if id not in ids:
                    ids.append(id)
            cv2.rectangle(frame,(x1,y1) ,(x2,y2), (0,0,255), 2)
            cv2.putText(frame, str(id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 4)
        ## Deep Sort
        # frame = track_frame(list_face_bbox,frame)
        end = time.time()
        
        fps = str(int(1/(end-start)))
        cv2.putText(frame, fps,(5,25),
        cv2.FONT_HERSHEY_SIMPLEX,1,  # font scale
        (255, 0, 255),3)
        cv2.imshow('OpenSeeFace Visualization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def track_OSF():
    log = None
    out = None
    first = True
    height = 0
    width = 0
    tracker = None
    sock = None
    total_tracking_time = 0.0
    tracking_time = 0.0
    tracking_frames = 0
    frame_count = 0
    cap = cv2.VideoCapture(2)

    ids = []
    track_sort = Sort()
    reid = dict()
    current_id = 0
    black_list = []
    list_id = []
    while cap.isOpened():
        ids = []
        ret, frame = cap.read()
        attempt = 0
        need_reinit = 0
        frame_count += 1
        start = time.time()
        if first:
            first = False
            height, width, channels = frame.shape
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
        inference_start = time.perf_counter()
        faces = tracker.predict_bboxonly(frame)

        packet = bytearray()
        detected = False
        list_bbox = []
        for face_num, f in enumerate(faces):
            f = copy.copy(f)
            box = f.bbox
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2]) + x1
            y2 = int(box[3]) + y1
            list_bbox.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
            f.id += args.face_id_offset
            ids.append(f.id)
            detected = True
        ## Reid
        # if current_face == 0 and len(ids) > 0:
        #     current_face = max(ids)
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
        # for face_num, f in enumerate(faces):
        #     f = copy.copy(f)
        #     f.id = reid[f.id][0]
        #     box = f.bbox
        #     x1 = int(box[0])
        #     y1 = int(box[1])
        #     x2 = int(box[2]) + x1
        #     y2 = int(box[3]) + y1
        #     cv2.rectangle(frame,(x1,y1) ,(x2,y2), (255,0,0), 2)
        #     cv2.putText(frame, str(f.id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        for i,bbox in enumerate(list_bbox):
            id = reid[ids[i]][0]
            x1,y1,x2,y2 = bbox
            cv2.rectangle(frame,(x1,y1) ,(x2,y2), (255,0,0), 2)
            cv2.putText(frame, str(id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        end = time.time()
        
        fps = str(int(1/(end-start)))
        cv2.putText(frame, fps,(5,25),
        cv2.FONT_HERSHEY_SIMPLEX,1,  # font scale
        (255, 0, 255),3)
        cv2.imshow('OpenSeeFace Visualization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

image_h, image_w = 128, 128
def convert_img(img):
    # img = tf.io.read_file(image_path)
    # img = tf.image.decode_image(img, channels=3)
    # img.set_shape([None,None,3])
    img = tf.image.resize(img, [image_w, image_h])
    img  = img/127.5-1
    return img
def wearing_mask():
    log = None
    out = None
    first = True
    height = 0
    width = 0
    tracker = None
    sock = None
    total_tracking_time = 0.0
    tracking_time = 0.0
    tracking_frames = 0
    frame_count = 0
    cap = cv2.VideoCapture(2)

    ids = []
    track_sort = Sort()
    reid = dict()
    current_id = 0
    black_list = []
    list_id = []

    model = load_model('./model/model1.h5')
    while cap.isOpened():
        ids = []
        ret, frame = cap.read()
        attempt = 0
        need_reinit = 0
        frame_count += 1
        start = time.time()
        if first:
            first = False
            height, width, channels = frame.shape
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tracker = Tracker(width, height, threshold=args.threshold, max_threads=args.max_threads, max_faces=args.faces, discard_after=args.discard_after, scan_every=args.scan_every, silent=False if args.silent == 0 else True, model_type=args.model, model_dir=args.model_dir, no_gaze=False if args.gaze_tracking != 0 and args.model != -1 else True, detection_threshold=args.detection_threshold, use_retinaface=args.scan_retinaface, max_feature_updates=args.max_feature_updates, static_model=True if args.no_3d_adapt == 1 else False, try_hard=args.try_hard == 1)
        inference_start = time.perf_counter()
        faces = tracker.predict_bboxonly(frame)

        packet = bytearray()
        detected = False
        list_bbox = []
        for face_num, f in enumerate(faces):
            f = copy.copy(f)
            box = f.bbox
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2]) + x1
            y2 = int(box[3]) + y1
            list_bbox.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),2)
            f.id += args.face_id_offset
            ids.append(f.id)
            detected = True

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
            x1,y1,x2,y2 = bbox

            box_face = frame[y1:y2,x1:x2]
            box_face = convert_img(box_face)
            y_pred = model.predict(np.expand_dims(box_face,axis=0))
            y_pred = np.argmax(y_pred,axis=1)
            print(y_pred)
            cv2.rectangle(frame,(x1,y1) ,(x2,y2), (255,0,0), 2)
            cv2.putText(frame, str(id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        end = time.time()
        
        fps = str(int(1/(end-start)))
        cv2.putText(frame, fps,(5,25),
        cv2.FONT_HERSHEY_SIMPLEX,1,  # font scale
        (255, 0, 255),3)
        cv2.imshow('OpenSeeFace Visualization', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# wearing_mask()