from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import sys

sys.path.append('./../')

from utils import draw_bbox_maxmin, write_text
from utils import get_config

# deep_sort
from libs import preprocessing
from libs import Tracker
from libs import Detection
from libs import nn_matching
from utils import generate_detections as gdet

# PROTOTXT = 'models/dnn/deploy.prototxt.txt'
# MODEL = 'models/dnn/res10_300x300_ssd_iter_140000.caffemodel'

def setup_config(cfg):
    if cfg.MAIN.path_file_config_cam != None:
        cfg.merge_from_file(cfg.MAIN.path_file_config_cam)
    
    if cfg.MAIN.path_file_deep_sort != None: 
        cfg.merge_from_file(cfg.MAIN.path_file_deep_sort)

def setup_deep_sort(cfg):
    encoder = gdet.create_box_encoder(
            cfg.DEEPSORT.MODEL, batch_size=4)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", cfg.DEEPSORT.MAX_COSINE_DISTANCE, cfg.DEEPSORT.NN_BUDGET)
    tracker = Tracker(cfg, metric)

    return encoder, metric, tracker

cfg = get_config()
cfg.merge_from_file("configs/main_process.yaml")
setup_config(cfg)
encoder, metric, tracker = setup_deep_sort(cfg)

def track_frame(list_face_bbox,frame):
    features = encoder(frame, list_face_bbox)
    detections = [Detection(bbox, 1.0, 'person', feature) for bbox,feature in
                    zip(list_face_bbox, features)]
    boxes = np.array([d.tlwh for d in detections])
        # boxes = np.array([d for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(
        boxes, cfg.DEEPSORT.NMS_MAX_OVERLAP, scores)
    detections = [detections[i] for i in indices]
    if len(list_face_bbox) != 0:
        tracker.predict()
        tracker.update(detections)
    else:
        tracker.predict()
    id = []
    list_bbox = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        # bbox = track.mean[:4].copy()
        bbox = track.to_tlbr()
        id.append(track.track_id)
        list_bbox.append(bbox)

    list_bbox = np.array(list_bbox).astype('int32')
    for i,box in enumerate(list_bbox):
        if id == [] or len(id) != len(list_bbox):
            break
        frame = draw_bbox_maxmin(frame, box, True, id[i])
    return frame,id