#!/usr/bin/python
import os
import time
import atexit
import cv2
import math
import numpy as np
import sys
import params
import argparse
import array
from multiprocessing import Process, Lock, Array
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

from PIL import Image, ImageDraw
import input_stream
import json
import logging

import signal
import sys
import shutil

##########################################################
# import deeppicar's sensor/actuator modules
##########################################################
camera   = __import__(params.camera)
actuator = __import__(params.actuator)

##########################################################
# global variable initialization
##########################################################
use_dnn = False
use_thread = True
view_video = False
fpv_video = False
enable_record = False

cfg_cam_res = (320, 240)
cfg_cam_fps = 30
cfg_throttle = 50 # 50% power.

max_throttle = cfg_throttle # Automatically set to throttle unless modified
min_throttle = cfg_throttle # Automatically set to throttle unless modified

frame_id = 0
angle = 0.0
period = 0.05 # sec (=50ms)

interpreter = None
input_index = None
output_index = None
finish = False

# Web stream and file handling
class stream_handler(BaseHTTPRequestHandler):
    global cfg_cam_fps
    streaming = True
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS, POST')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Location', '/index.html')
            self.end_headers()
        if self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            period = 1./cfg_cam_fps
            end_time = time.time() + period

            try:
                while stream_handler.streaming:
                    frame = camera.read_frame()
                    ret, frame = cv2.imencode('.jpg', frame)
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')

                    tdiff = end_time - time.time()
                    if tdiff > 0:
                        time.sleep(tdiff)
                    end_time += period
                    print('streaming')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        elif self.path == '/download':
            shutil.make_archive('./Dataset', 'zip', './data')
            f = open('./Dataset.zip', 'rb')
            self.send_response(200)
            self.send_header('Content-Type', 'application/zip')
            self.end_headers()
            self.wfile.write(f.read())
            f.close()
        else:
            self.send_error(404)
            self.end_headers()

    def do_POST(self):
        global new_inp_type
        if self.path == '/stream.mjpg':
            self.send_response(201)
            self.end_headers()
            self.data_string = self.rfile.read(int(self.headers['Content-Length']))
            data = json.loads(self.data_string)
            print(data)
            stream_handler.streaming = data['params']['streaming']
        elif self.path == '/upload':
            filename = "large-200x66x3.tflite"
            file_length = int(self.headers['Content-Length'])
            read = 0
            with open('./models/'+filename, 'wb') as output_file:
                output_file.write(self.rfile.read(file_length))
            self.send_response(201, 'Created')
            self.end_headers()
            reply_body = 'Saved "%s"\n' % filename
            self.wfile.write(reply_body.encode('utf-8'))
            load_model()
        elif self.path == '/input_switch':
            self.send_response(301)
            self.end_headers()
            self.data_string = self.rfile.read(int(self.headers['Content-Length']))
            data = json.loads(self.data_string)
            new_inp_type = int(data['params']['input_type'])
        else:
            self.send_error(404)
            self.end_headers()

        
##########################################################
# local functions
##########################################################

def deg2rad(deg):
    return deg * math.pi / 180.0

def rad2deg(rad):
    return 180.0 * rad / math.pi

def g_tick():
    t = time.time()
    count = 0
    while True:
        count += 1
        yield max(t + count*period - time.time(),0)

def turn_off():
    print('Finishing...')
    stream_handler.streaming = False
    server.server_close()
    actuator.stop()
    camera.stop()
    cur_inp_stream.stop()

def preprocess(img):
    img = img[img.shape[0]//2:]
    img = cv2.resize(img, (params.img_width, params.img_height))
    # Convert to grayscale and readd channel dimension
    if params.img_channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.reshape(img, (params.img_height, params.img_width, params.img_channels))
    img = img / 255.
    return img

def overlay_image(l_img, s_img, x_offset, y_offset):
    assert y_offset + s_img.shape[0] <= l_img.shape[0]
    assert x_offset + s_img.shape[1] <= l_img.shape[1]

    l_img = l_img.copy()
    for c in range(0, 3):
        l_img[y_offset:y_offset+s_img.shape[0],
              x_offset:x_offset+s_img.shape[1], c] = (
                  s_img[:,:,c] * (s_img[:,:,3]/255.0) +
                  l_img[y_offset:y_offset+s_img.shape[0],
                        x_offset:x_offset+s_img.shape[1], c] *
                  (1.0 - s_img[:,:,3]/255.0))
    return l_img


def load_model():
    global interpreter
    global input_index
    global output_index
    ##########################################################
    # import deeppicar's DNN model
    ##########################################################
    print ("Loading model: " + params.model_file)
    try:
        # Import TFLite interpreter from tflite_runtime package if it's available.
        from tflite_runtime.interpreter import Interpreter
        interpreter = Interpreter(params.model_file+'.tflite', num_threads=args.ncpu)
    except ImportError:
        # If not, fallback to use the TFLite interpreter from the full TF package.
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=params.model_file+'.tflite', num_threads=args.ncpu)

    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

def signal_handler(sig, frame):
    global finish
    finish = True

signal.signal(signal.SIGINT, signal_handler)


##########################################################
# program begins
##########################################################
if __name__ == '__main__':    

    parser = argparse.ArgumentParser(description='DeepPicar main')
    parser.add_argument("-d", "--dnn", help="Enable DNN", action="store_true")
    parser.add_argument("-t", "--throttle", help="throttle percent. [0-100]%", type=int)
    parser.add_argument("-n", "--ncpu", help="number of cores to use.", type=int, default=1)
    parser.add_argument("-f", "--hz", help="control frequnecy", type=int)
    parser.add_argument("-min", "--min", help="throttle minimum", type=int)
    parser.add_argument("-g", "--gamepad", help="Use gamepad", action="store_true")
    parser.add_argument("-w", "--web", help="Use webpage based input", action="store_true")
    parser.add_argument("--fpvvideo", help="Take FPV video of DNN driving", action="store_true")
    args = parser.parse_args()


    if args.dnn:
        print ("DNN is on")
        use_dnn = True
    if args.throttle:
        max_throttle = args.throttle
        cfg_throttle = args.throttle
        print ("throttle = %d pct" % (cfg_throttle))
    if args.hz:
        period = 1.0/args.hz
        print("new period: ", period)
    if args.fpvvideo:
        fpv_video = True
        print("FPV video of DNN driving is on")
    if args.min:
        min_throttle = args.min


    load_model()
        
    if args.gamepad:
        cur_inp_type= input_stream.input_type.GAMEPAD
    elif args.web:
        cur_inp_type= input_stream.input_type.WEB
    else:
        cur_inp_type= input_stream.input_type.KEYBOARD
    new_inp_type=cur_inp_type
    cur_inp_stream= input_stream.instantiate_inp_stream(cur_inp_type, cfg_throttle)

    address = ('', 8001)
    server = ThreadingHTTPServer(address, stream_handler)
    server.timeout = 0

    # initlaize deeppicar modules

    actuator.init(cfg_throttle)
    camera.init(res=cfg_cam_res, fps=cfg_cam_fps, threading=use_thread)

    g = g_tick()
    start_ts = time.time()

    frame_arr = []
    angle_arr = []
    # enter main loop
    while not finish:
        if use_thread:
            time.sleep(next(g))
        frame = camera.read_frame()
        ts = time.time()

        if new_inp_type != cur_inp_type:
            del cur_inp_stream
            cur_inp_type= new_inp_type
            cur_inp_stream= input_stream.instantiate_inp_stream(cur_inp_type, speed)

        if view_video:
            cv2.imshow('frame', frame)
            ch = cv2.waitKey(1) & 0xFF
        else:
            command, direction, speed = cur_inp_stream.read_inp()
            
        actuator.set_speed(speed)

        if command == 'a':
            actuator.ffw()
            print ("accel")
        elif command == 's':
            actuator.stop()
            print ("stop")
        elif command == 'z':
            actuator.rew()
            print ("reverse")
        elif command == 'r':
            print ("toggle record mode")
            if enable_record:
                keyfile.close()
                vidfile.release()
                frame_id= 0
            enable_record = not enable_record
        elif command == 't':
            print ("toggle video mode")
            view_video = not view_video
        elif command == 'd':
            print ("toggle DNN mode")
            use_dnn = not use_dnn
        elif command == 'q':
            finish = True
            break

        if use_dnn:
            # 1. machine input
            img = preprocess(frame)
            img = np.expand_dims(img, axis=0).astype(np.float32)
            interpreter.set_tensor(input_index, img)
            interpreter.invoke()
            angle = interpreter.get_tensor(output_index)[0][0]
            action_limit = 10
            print(f"Angle: {rad2deg(angle)}, possible decreased throttle: {max_throttle - rad2deg(angle * 5) // 2} ")

            if rad2deg(angle) < -action_limit:
                if ((max_throttle - abs(rad2deg(angle * 5) // 2))) < min_throttle:
                    cfg_throttle = min_throttle
                else:
                    cfg_throttle = (max_throttle - abs(rad2deg(angle * 5) // 2))
                actuator.left()
                print ("left (CPU)")
            elif rad2deg(angle) >= -action_limit and rad2deg(angle) <= action_limit:
                cfg_throttle = args.throttle
                actuator.center()
                print ("center (CPU)")
            elif rad2deg(angle) > action_limit:
                if ((max_throttle - abs(rad2deg(angle * 5) // 2))) < min_throttle:
                    cfg_throttle = min_throttle
                else:
                    cfg_throttle = (max_throttle - abs(rad2deg(angle * 5) // 2))
                actuator.right()
                print ("right (CPU)")
        else:
            if direction < 0:
                angle = deg2rad(direction * 30)
                actuator.left(direction)
                print ("left")
            elif direction > 0:
                angle = deg2rad(direction * 30)
                actuator.right(direction)
                print ("right")
            else:
                angle=0.
                actuator.center()
                print ("center")

        dur = time.time() - ts
        if dur > period:
            print("%.3f: took %d ms - deadline miss."
                % (ts - start_ts, int(dur * 1000)))
        else:
            print("%.3f: took %d ms" % (ts - start_ts, int(dur * 1000)))

        if enable_record == True and frame_id == 0:
            # ensure directory exists
            os.makedirs(params.data_dir, exist_ok=True)
            # create files for data recording
            keyfile = open(params.rec_csv_file, 'w+')
            keyfile.write("ts_micro,frame,wheel\n")
            try:
                fourcc = cv2.cv.CV_FOURCC(*'XVID')
            except AttributeError as e:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            vidfile = cv2.VideoWriter(params.rec_vid_file, fourcc,
                                    cfg_cam_fps, cfg_cam_res)
        if enable_record == True and frame is not None:
            # increase frame_id
            frame_id += 1

            # write input (angle)
            str = "{},{},{},{}\n".format(int(ts*1000), frame_id, angle, int(cfg_throttle))
            keyfile.write(str)

            if use_dnn and fpv_video:
                textColor = (255,255,255)
                bgColor = (0,0,0)
                newImage = Image.new('RGBA', (100, 20), bgColor)
                drawer = ImageDraw.Draw(newImage)
                drawer.text((0, 0), "Frame #{}".format(frame_id), fill=textColor)
                drawer.text((0, 10), "Angle:{}".format(angle), fill=textColor)
                newImage = cv2.cvtColor(np.array(newImage), cv2.COLOR_BGR2RGBA)
                frame = overlay_image(frame,
                                        newImage,
                                        x_offset = 0, y_offset = 0)
            # write video stream
            vidfile.write(frame)
            #img_name = "cal_images/opencv_frame_{}.png".format(frame_id)
            #cv2.imwrite(img_name, frame)
            #if frame_id >= 1000:
            #    print ("recorded 1000 frames")
            #    break
            print ("%.3f %d %.3f %d(ms)" %
            (ts, frame_id, angle, int((time.time() - ts)*1000)))

        server.handle_request()
    turn_off()
