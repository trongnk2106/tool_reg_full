# from bankcard_demo.app_v3 import output_txt
from flask import Flask, render_template, json, request, jsonify, Response, send_from_directory, copy_current_request_context, send_file, flash, redirect, url_for
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import cv2
import requests
import logging
import time
import threading
from flask_socketio import SocketIO, emit

from configparser import SafeConfigParser
from utils import rcode
import SurveillanceSystem
import Camera
from database import DB

import subprocess
import urllib.parse
import os
from zipfile import ZipFile

from multiprocessing import Queue
import base64
import json
import sys
import shutil

from functools import wraps
import memory_profiler
from memory_profiler import profile
import gc


############ MEMORY CHECKER #####################
try:
    import tracemalloc
    has_tracemalloc = True
except ImportError:
    has_tracemalloc = False

def my_profiler(func=None, stream=None, precision=1, backend='psutil'):
    """
    Decorator that will run the function and print a line-by-line profile
    """
    backend = memory_profiler.choose_backend(backend)
    if backend == 'tracemalloc' and has_tracemalloc:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    if func is not None:
        @wraps(func)
        def wrapper(*args, **kwargs):
            prof = memory_profiler.LineProfiler(backend=backend)
            val = prof(func)(*args, **kwargs)
            memory_profiler.show_results(prof, stream=stream,
                                         precision=precision)
            return val

        return wrapper
    else:
        def inner_wrapper(f):
            return profile(f, stream=stream, precision=precision,
                           backend=backend)

        return inner_wrapper

sys.path.append('src')
from datetime import datetime
import tempfile


HomeSurveillance = SurveillanceSystem.SurveillanceSystem()
datasetFace = DB()
facesUpdateThread = threading.Thread()
facesUpdateThread.daemon = False

####LOAD CONFIG####
config = SafeConfigParser()
config.read("configs/services.cfg")
LOG_PATH = str(config.get('main', 'LOG_PATH'))
SERVER_IP = str(config.get('main', 'SERVER_IP'))
SERVER_PORT = int(config.get('main', 'SERVER_PORT'))
UPLOAD_VIDEO = str(config.get('storage', 'UPLOAD_VIDEO'))
UPLOAD_FACE = str(config.get('storage', 'UPLOAD_FACE'))
DOWNLOAD_FACE = str(config.get('storage', 'DOWNLOAD_FACE'))
CAFFEMODEL = str(config.get('model', 'CAFFEMODEL'))
PROTOTXT = str(config.get('model', 'PROTOTXT'))

#####CREATE LOGGER#####
logging.basicConfig(
    filename=os.path.join(LOG_PATH,
                          str(time.time()) + ".log"),
    filemode="w",
    level=logging.DEBUG,
    format=
    '%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)
#######################################
app = Flask(__name__, template_folder="templates", static_folder="static")
cors = CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000 * 1000
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SECRET_KEY'] = os.urandom(24)  # Used for session management
async_mode = None
socketio = SocketIO(app, async_mode=async_mode)

@app.route('/memory')
def print_memory():
    return {'memory': process.memory_info().rss}


@app.route("/snapshot")
def snap():
    global s
    if not s:
        s = tracemalloc.take_snapshot()
        return "taken snapshot\n"
    else:
        lines = []
        top_stats = tracemalloc.take_snapshot().compare_to(s, 'lineno')
        for stat in top_stats[:5]:
            lines.append(str(stat))
        return "\n".join(lines)

def get_command_resp(command):
    return subprocess.Popen(command, stdout=subprocess.PIPE,
                            shell=True).communicate()


def get_command_ret(command):
    return subprocess.Popen(command, stdout=subprocess.PIPE, shell=True).wait()

def get_file_from_cmd(url, out_filename):
    # cmd = r'c:\aria2\aria2c.exe -d '+ save_dir +' -m 5 -o ' + out_filename + " "+ url
    cmd ='aria2c' ' -c -s 16 -x 16 -k 1M -j 1' + ' --out=' + out_filename + ' ' + url
    try:
        p1=subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE)
        print("---start---")
        msg_content = ''
        for line in p1.stdout:
            print(line)
            l = line.decode(encoding="utf-8", errors="ignore")
            msg_content += l
        p1.wait()
        if '(OK):download completed' in msg_content:
            print("download by aira2 successfully.")
            return True
        return False
    except Exception as e:
        print(e)
        return False

ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route('/upload_file', methods=['POST'])
# def upload_file():
#     print('/// upload file ///')
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return json.dumps({'html': 'No file part'})
#         file = request.files['file']
#         if file.filename == '':
#             flash('No selected file')
#             return json.dumps({'html': 'No selected file'})
#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(UPLOAD_VIDEO, filename))
#             SAVE_VIDEO_PATH = os.path.join(UPLOAD_VIDEO, filename)
#             print(SAVE_VIDEO_PATH)
#             return jsonify({'video_path': SAVE_VIDEO_PATH, 'name_video': str(filename)})
#         return json.dumps({'html': 'No selected file'})


@app.route('/downloadVideo', methods=['POST'])
def downloadVideo():
    inputURI = request.form['inputURI']
    inputName = request.form['inputName'] + '.mp4'
    hostname = urllib.parse.urlparse(inputURI).netloc
    error = 0
    global SAVE_VIDEO_PATH
    # id_video = inputURI.rep
    SAVE_VIDEO_PATH = os.path.join(UPLOAD_VIDEO, inputName)
    if os.path.exists(SAVE_VIDEO_PATH):
        return jsonify({'video_path': SAVE_VIDEO_PATH, 'name_video': str(inputName)})
    if hostname == 'www.youtube.com':
        command = 'youtube-dl ' + '--get-filename --format best[ext=mp4] ' + inputURI
        videoFilename = get_command_resp(command)[0].strip().decode('utf-8')
        command = 'youtube-dl ' + '--format best[ext=mp4] ' + inputURI + ' --output ' + SAVE_VIDEO_PATH
    elif 'rtsp' in inputURI:
        return jsonify({'video_path': inputURI, 'name_video': str(inputName)})
    elif '.mp4' in inputURI or '.avi' in inputURI:
        get_file_from_cmd(inputURI, SAVE_VIDEO_PATH)
    elif 'file1' in request.files or 'file2' in request.files:
        file = request.files['file1'] if 'file1' in request.files else request.files['file2']
        if file.filename == '':
            flash('No selected file')
            return json.dumps({'html': 'No selected file'})
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_VIDEO, filename))
            SAVE_VIDEO_PATH = os.path.join(UPLOAD_VIDEO, filename)
            return jsonify({'video_path': SAVE_VIDEO_PATH, 'name_video': str(filename)})
    else:
        return json.dumps({
            'html':
            '<span>Download failed with error code: ' + str(error) +
            '</span><br>'
        })

    print(command)
    # error = get_command_ret(command)
    print(error)
    if error == 0:
        print('inputName', inputName)
        return jsonify({'video_path': SAVE_VIDEO_PATH, 'name_video': str(inputName)})
    else:
        return json.dumps({'html': '<span>Please input a URI.<span><br>'})


@app.route('/home', methods=['GET', 'POST'])
# @my_profiler
def control_panel():
    if request.args.get('video_path'):
        video_path = request.args.get('video_path')
        name_video = request.args.get('name_video')
        detectionMethod = request.args.get('detectionMethod')
        dist_video = {'video_path': video_path, 'name_video': name_video, 'first': True, 'detectionMethod': detectionMethod}
        return render_template('/index.html', data=dist_video)
    app.logger.info("==== RENDER: HOME===")
    return render_template('/index.html', data={})


@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def start():
    app.logger.info("==== RENDER: START===")
    with HomeSurveillance.camerasLock:
        if len(HomeSurveillance.cameras) > 0:
            return render_template('/index.html', data={})
        else:
            return render_template('/start.html')
    
    

@app.route('/remove_face_DB', methods=['GET', 'POST'])
def remove_face_DB():
    if request.method == 'POST':
        predicted_name = request.form.get('name')
        path_face = os.path.join(UPLOAD_FACE, predicted_name)
        try:
            shutil.rmtree(path_face)
        except Exception as e:
                app.logger.error("ERROR could not remove Face DB" + e)
                pass
        data = {"face_removed_db": 'true'}
        return jsonify(data)
    return render_template('/dataset.html')

@app.route('/get_face_db_img/<name>')
def get_face_db_img(name):
    face_name, img_id = name.split("_")
    path_face = os.path.join(UPLOAD_FACE, face_name)
    name_file = face_name + '_' + img_id + '.jpg'
    while not os.path.exists(os.path.join(path_face, name_file)):
        img_id  = int(img_id) + 1
        name_file = face_name + '_' + str(img_id) + '.jpg'
    try:
        img = cv2.imread(os.path.join(path_face,name_file))
        ret, jpeg = cv2.imencode('.jpg', img)
        img = jpeg.tostring()
    except Exception as e:
        app.logger.error("Error " + e)
        img = ""
    if img == "":
        return "http://www.character-education.org.uk/images/exec/speaker-placeholder.png"
    return Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/dataset', methods=['GET', 'POST'])
def dataset():
    index = request.args.get('index')
    if request.args.get('nameSearch')  != '':
        print('nameSearch', request.args.get('nameSearch'))
        name_search = request.args.get('nameSearch')
        datasetFace.get_name(index, name_search, 0.4)
    else:
        datasetFace.get_name(index, None)
    print('datasetFace.list_path', datasetFace.page_filelist)
    print('num_page', datasetFace.num_page)
    print('num_person', datasetFace.num_person)
    data =  {'list_path': datasetFace.page_filelist, 'num_page': datasetFace.num_page, 'num_person': datasetFace.num_person}
    return render_template('/dataset.html', data=data)

@app.route('/remove_single_face_DB', methods=['GET', 'POST'])
def remove_single_face_DB():
    if request.method == 'POST':
        path_face = request.form.get('id')
        try:
            os.remove(path_face)
        except Exception as e:
                app.logger.error("ERROR could not remove single Face DB" + e)
                pass
        data = {"single_face_removed_db": 'true'}
        return jsonify(data)
    return render_template('/singlePeople.html')

@app.route('/send_result_info', methods=['GET', 'POST'])
def send_result_info():
    if request.method == 'POST':
        json_data = request.get_json(force=True)
        name = json_data['name']
        id = json_data['id']
        age = json_data['age']
        gender = json_data['gender']
        
        json_content = {'name': name, 'id': id, 'age': age, 'gender':gender}
        print(json_content)
        json_path = os.path.join(UPLOAD_FACE,id, id + '.json')
        try:
            with open(json_path, 'w') as out:
                json.dump(json_content, out)
        except Exception as e:
                app.logger.error("ERROR could not SAVE INFO DB" + e)
                pass 
        data = {"send_result_info": 'true'}
        return jsonify(data)
    return render_template('/singlePeople.html')

@app.route('/single_people')
def single_people():
    datasetFace.get_name()
    face_id = request.args.get('id')
    list_face = {'info':datasetFace.list_info[face_id], 'path_face': datasetFace.list_face_img_path[face_id]}
    app.logger.info("==== RENDER: SINGLE PEOPLE===")

    return render_template('/singlePeople.html', data=list_face)

@app.route('/downloadFile/<path>')
def downloadFile(path):
    path_face = os.path.join(UPLOAD_FACE, path)
    output_filename = os.path.join(DOWNLOAD_FACE, path)
    shutil.make_archive(output_filename, 'zip', path_face)
    return send_file(output_filename + '.zip', as_attachment=True)

@app.route('/downloadDB/')
def downloadDB():
    shutil.make_archive(DOWNLOAD_FACE + '/DB', 'zip', UPLOAD_FACE)
    return send_file(DOWNLOAD_FACE + '/DB.zip', as_attachment=True)

@app.route('/get_face_img/<name>')
# @my_profiler
def get_faceimg(name):
    key, camNum = name.split("_")
    try:
        with HomeSurveillance.cameras[int(camNum)].peopleDictLock:
            img = HomeSurveillance.cameras[int(camNum)].people[key].thumbnail
    except Exception as e:
        app.logger.error("Error " + e)
        img = ""

    if img == "":
        return "http://www.character-education.org.uk/images/exec/speaker-placeholder.png"
    return Response((b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n\r\n'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/resize_face', methods=['GET', 'POST'])
def resize_face():
    if request.method == 'POST':
        camNum = request.form.get('camNum')
        sizeFace = request.form.get('size')
        print('camNum', camNum)
        print('sizeFace', sizeFace)
        HomeSurveillance.cameras[int(camNum)].sizeFace = int(sizeFace)
    return jsonify({"resize_face": 'true'})

@app.route('/add_face', methods=['GET', 'POST'])
def add_face():
    if request.method == 'POST':
        new_name = request.form.get('new_name')
        person_id = request.form.get('person_id')
        camNum = request.form.get('camera')
        img = None

        with HomeSurveillance.cameras[int(camNum)].peopleDictLock:
            try:
                img = HomeSurveillance.cameras[int(camNum)].people[
                    person_id].face  # Gets face of person detected in cameras
                predicted_name = HomeSurveillance.cameras[int(
                    camNum)].people[person_id].identity
                del HomeSurveillance.cameras[int(camNum)].people[
                    person_id]  # Removes face from people detected in all cameras
                gc.collect()
            except Exception as e:
                app.logger.error("ERROR could not add Face" + e)

        wriitenToDir = HomeSurveillance.add_face(new_name, img)

        systemData = {
            'camNum': len(HomeSurveillance.cameras),
            'people': HomeSurveillance.peopleDB,
            'onConnect': False
        }
        socketio.emit('system_data',
                      json.dumps(systemData),
                      namespace='/surveillance')

        data = {"face_added": wriitenToDir}
        return jsonify(data)
    return render_template('index.html')


@app.route('/remove_face', methods=['GET', 'POST'])
def remove_face():
    if request.method == 'POST':
        predicted_name = request.form.get('predicted_name')
        camNum = request.form.get('camera')
        PEOPLE_DIST.pop(predicted_name)
        with HomeSurveillance.cameras[int(camNum)].peopleDictLock:
            try:
                del HomeSurveillance.cameras[int(
                    camNum)].people[predicted_name]
                gc.collect()
                app.logger.info("==== REMOVED: " + predicted_name + "===")
            except Exception as e:
                app.logger.error("ERROR could not remove Face" + e)
                pass

        data = {"face_removed": 'true'}
        return jsonify(data)
    return render_template('index.html')

from imagecluster import calc, io as icio, postproc
model = calc.get_model()
@app.route('/face_cluster', methods=['GET'])
def face_cluster():
    data = {}
    print('PEOPLE_DIST', len(PEOPLE_DIST))
    
    # for key, value in PEOPLE_DIST.items():
    #     print('key', key)
    #     print('type value', type(value))
    fingerprints = calc.fingerprints(PEOPLE_DIST, model)
    fingerprints = calc.pca(fingerprints, n_components=0.95)
    clusters = calc.cluster(fingerprints, sim=0.5, alpha=0.2)
    print('clusters', clusters)
    data['clusters'] = clusters
    data['cam'] = KEY_CAM_ID
        
    return jsonify(data)


def update_faces():
    """Used to push all detected faces to client"""
    global PEOPLE_DIST, KEY_CAM_ID
    while True:
        app.logger.info("Starting update_faces")
        peopledata = []
        persondict = {}
        PEOPLE_DIST = {}
        KEY_CAM_ID = {}
        thumbnail = None
        with HomeSurveillance.camerasLock:
            for i, camera in enumerate(HomeSurveillance.cameras):
                with HomeSurveillance.cameras[i].peopleDictLock:
                    for key, person in camera.people.items():
                        persondict = {
                            'identity': key,
                            'confidence': person.confidence,
                            'camera': i,
                            'timeD': person.time,
                            'prediction': person.identity,
                            'thumbnailNum': len(person.thumbnails)
                            # 'img_to_arr': person.images_array
                        }
                        # app.logger.info(persondict)
                        PEOPLE_DIST[key] = person.img_to_arr
                        KEY_CAM_ID[str(key)] = i
                        peopledata.append(persondict)
        # app.logger.info("Starting update_faces:", str(peopledata))
        # print("peopledata", peopledata)
        socketio.emit('people_detected',
                      json.dumps(peopledata),
                      namespace='/surveillance')
        time.sleep(1)


def gen(camera):
    while True:
        frame = camera.read_processed()  # read_jpg()  # read_processed()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'
               )  # Builds 'jpeg' data with header and payload


@app.route('/video_streamer/<camNum>')
# @my_profiler
def video_streamer(camNum):
    """Used to stream frames to client, camNum represents the camera index in the cameras array"""
    return Response(
        gen(HomeSurveillance.cameras[int(camNum)]),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )  # A stream where each part replaces the previous part the multipart/x-mixed-replace content type must be used.

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)

@app.route('/add_camera', methods=['GET', 'POST'])
# @my_profiler
def add_camera():
    """Adds camera new camera to SurveillanceSystem's cameras array"""
    
    if request.method == 'POST':
        camURL = request.form.get('camURL')
        fpsTweak = request.form.get('fpstweak')
        detectionMethod = request.form.get('detectionMethod')
        sizeFace = request.form.get('sizeFace')
        random_key_stream = request.form.get('random_key_stream')
        with HomeSurveillance.camerasLock:
            if camURL != "None":
                if HomeSurveillance.cameras == []:
                    app.logger.info("===== ADDING FIRST CAMERA: " + camURL + "=====")
                    HomeSurveillance.add_camera(
                        SurveillanceSystem.Camera.IPCamera(camURL, fpsTweak, detectionMethod, sizeFace, net))
                    HomeSurveillance.cameras[-1].random_key_stream = random_key_stream
                    data = {"camNum": len(HomeSurveillance.cameras) - 1, 'camURL': camURL, 'fpsTweak': fpsTweak, 'random_key_stream': random_key_stream}
                else:
                    for i in range(len(HomeSurveillance.cameras)):
                        if HomeSurveillance.cameras[i].url == camURL:
                            app.logger.info("===== CAMERA ALREADY EXISTS: " + camURL + "=====")
                            data = {"camNum": i, 'camURL': camURL, 'fpsTweak': fpsTweak, 'random_key_stream': HomeSurveillance.cameras[i].random_key_stream}
                        else:
                            
                            app.logger.info("===== ADDING CAMERA: " + camURL + "=====")
                            HomeSurveillance.add_camera(
                                SurveillanceSystem.Camera.IPCamera(camURL, fpsTweak, detectionMethod, sizeFace, net))
                            HomeSurveillance.cameras[-1].random_key_stream = random_key_stream
                            data = {"camNum": len(HomeSurveillance.cameras) - 1, 'camURL': camURL, 'fpsTweak': fpsTweak, 'random_key_stream': random_key_stream}
            elif len(HomeSurveillance.cameras) > 0:
                # app.logger.info("===== ADDING CAMERA: " + camURL + "=====")
                data = {"camNum": len(HomeSurveillance.cameras) - 1, 'camURL': camURL, 'fpsTweak': fpsTweak, 'random_key_stream': HomeSurveillance.cameras[-1].random_key_stream}
        
        return jsonify(data)
    return render_template('index.html')

@app.route('/remove_camera', methods = ['GET','POST'])
# @my_profiler
def remove_camera():
    if request.method == 'POST':
        camID = request.form.get('camID')
        if camID != 'all':
            app.logger.info("Removing camera: ")
            app.logger.info(camID)
            data = {"camNum": len(HomeSurveillance.cameras) - 1}
            with HomeSurveillance.camerasLock:
                HomeSurveillance.cameras[int(camID)].captureThread.stop = True
                HomeSurveillance.remove_camera(camID)
                
            app.logger.info("Removing camera number : " + camID)
        else:
            # Remove all cameras
            with HomeSurveillance.camerasLock:
                print(len(HomeSurveillance.cameras))
                for i in range(len(HomeSurveillance.cameras))[::-1]:
                    # i = len(HomeSurveillance.cameras) - i - 1
                    print(i)
                    HomeSurveillance.cameras[i].captureThread.stop = True
                    HomeSurveillance.remove_camera(i)
            app.logger.info("Removing all cameras")

        data = {"alert_status": "removed"}
        return jsonify(data)
    return render_template('index.html')

@app.route('/stop_camera', methods = ['GET','POST'])
def stop_camera():
    if request.method == 'POST':
        camID = request.form.get('camID')
        app.logger.info("Stoping camera: ")
        app.logger.info(camID)
        data = {"camNum": len(HomeSurveillance.cameras) - 1}
        with HomeSurveillance.camerasLock:
            HomeSurveillance.cameras[int(camID)].paused = True

        app.logger.info("Stoping camera number : " + camID)
        data = {"alert_status": "Stop"}
        return jsonify(data)
    return render_template('index.html')

@app.route('/restart_camera', methods = ['GET','POST'])
def restart_camera():
    if request.method == 'POST':
        camID = request.form.get('camID')
        app.logger.info("Restart camera: ")
        app.logger.info(camID)
        data = {"camNum": len(HomeSurveillance.cameras) - 1}
        with HomeSurveillance.camerasLock:
            HomeSurveillance.cameras[int(camID)].paused = False
            
        app.logger.info("Restart camera number : " + camID)
        data = {"alert_status": "Restart"}
        return jsonify(data)
    return render_template('index.html')

@socketio.on('connect', namespace='/surveillance')
# @my_profiler
def connect():
    global facesUpdateThread
    global datasetFace

    app.logger.info("client connected")
    if not facesUpdateThread.isAlive():
        app.logger.info("Starting facesUpdateThread")
        facesUpdateThread = threading.Thread(name='websocket_process_thread_',
                                             target=update_faces,
                                             args=())
        facesUpdateThread.start()

    cameraData = {}
    cameras = []

    with HomeSurveillance.camerasLock:
        for i, camera in enumerate(HomeSurveillance.cameras):
            with HomeSurveillance.cameras[i].peopleDictLock:
                cameraData = {'camNum': i, 'url': camera.url, 'random_key_stream': camera.random_key_stream}
                #print cameraData
                app.logger.info(cameraData)
                cameras.append(cameraData)
    systemData = {
        'camNum': len(HomeSurveillance.cameras),
        'cameras': cameras,
        'onConnect': True
    }
    socketio.emit('system_data',
                  json.dumps(systemData),
                  namespace='/surveillance')


if __name__ == "__main__":
    global SAVE_VIDEO_PATH, PEOPLE_DIST, KEY_CAM_ID
    # app.run(debug=True,host=SERVER_IP, port=SERVER_PORT)
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    socketio.run(app, host=SERVER_IP, port=SERVER_PORT, debug=True)
