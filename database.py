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
from logging.handlers import RotatingFileHandler
import tempfile

# search name 
import hmni

matcher = hmni.Matcher(model='latin')

sys.path.append('src')

start = time.time()
np.set_printoptions(precision=2)

logger = logging.getLogger()
formatter = logging.Formatter("(%(threadName)-10s) %(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = RotatingFileHandler("logfile/database.log", maxBytes=10000000, backupCount=10)
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
IMG_PER_INDEX = int(config.get('dataset', 'IMG_PER_INDEX'))

if not os.path.exists(UPLOAD_FACE):
    os.makedirs(UPLOAD_FACE)


class DB(object):
    def __init__(self):
        self.list_name = []
        self.list_path = []
        self.list_path_single_page = []
        self.dist_name_id = {}
        self.dist_path = {}
        self.list_info = {}
        self.page_filelist = {}
        self.num_page = 1
        self.num_person = 0

        self.captureLock = threading.Lock() # Sometimes used to prevent concurrent access
        self.captureThread = threading.Thread(name='get_name_id',target=self.get_name)
        self.captureThread.daemon = True
        self.captureThread.start()
        self.captureThread.join()

        #self.captureThread.stop = False
    
    def get_name(self, index=0, name_search = None ,name_similarity=0.6):
        path, dirs, files = next(os.walk(UPLOAD_FACE))
        self.reload()
        for fn in dirs:
            path_face = os.path.join(path, fn)
            p_DB = Person_DB(fn)
            name = p_DB.name
            if name_search is not None:
                sim = matcher.similarity(name, name_search)
                if sim < name_similarity:
                    continue
            self.list_name.append(name)
        
            self.dist_name_id[name] = fn
            self.dist_path = {'id':fn, 'name': name ,'path_face':path_face}
            self.list_path.append(self.dist_path)

            p_DB.get_img_path(fn)
            self.list_info[fn] = p_DB.info
            self.list_face_img_path[fn] = p_DB.list_img_path
        
        self.num_page = int(len(self.list_path)/IMG_PER_INDEX) + 1
        self.num_person = len(self.list_path)
        index = int(index) - 1
        first_index = index * IMG_PER_INDEX
        if len(self.list_path)-1 > index+IMG_PER_INDEX:
            last_index = index * IMG_PER_INDEX + IMG_PER_INDEX
            self.page_filelist = self.list_path[first_index:last_index]
        else:
            last_index = len(self.list_path)
            self.page_filelist = self.list_path[first_index:last_index]
        

    def reload(self):
        self.list_path = []
        self.dist_path = {}
        self.list_name = []
        self.dist_name_id = {}
        self.list_info = {}
        self.list_face_img_path = {}
    

        
class Person_DB(object):
    def __init__(self , id):
        self.root_path = os.path.join(UPLOAD_FACE, id)
        self.name = ''
        self.age = ''
        self.gender = ''
        self.list_img_path = []
        self.info = {}
        
        self.captureLock = threading.Lock() # Sometimes used to prevent concurrent access
        self.captureThread = threading.Thread(name='get_info',target=self.get_info, args=(id,))
        self.captureThread.daemon = True
        self.captureThread.start()
        self.captureThread.join()

    
    def get_info(self,id):
        json_file = os.path.join(self.root_path, id + '.json')
        with open(json_file, encoding="utf8") as jf:
            info_json = json.load(jf)
        # print('///info_json////', info_json)
        self.name = info_json['name']
        self.gender = info_json['gender']
        self.age = info_json['age']
        self.info = info_json
        

    def get_img_path(self,id):
        path, dirs, files = next(os.walk(self.root_path))
        data_face = []
        for fn in files:
            if 'json' in fn:
                continue
            path_face_file = os.path.join(path, fn)
            self.list_img_path.append({'name': fn.replace('.jpg',''), 'path': path_face_file})
