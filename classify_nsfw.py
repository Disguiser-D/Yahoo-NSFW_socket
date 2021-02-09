#!/usr/bin/env python
import sys
import argparse
import os
import configparser
import logging
import logging.handlers

import tensorflow as tf
import numpy as np
import socket as s

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader
from socket import *
from multiprocessing import Pool
import multiprocessing

IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

cf = configparser.ConfigParser()
cf.read("config.ini")
IP = cf.get("Socket-info", "HOST")
PORT = int(cf.get("Socket-info", "PORT"))
BUFSIZ = int(cf.get("Socket-info", "BUFSIZ"))
PROCESS_NUMBER = int(cf.get("Operational-configuration", "PROCESS_NUMBER"))
INPUT_TYPE = cf.get("Operational-configuration", "INPUT_TYPE")
FN_LOAD_IMAGE = cf.get("Operational-configuration", "FN_LOAD_IMAGE")
LOGGER_LEVEL = cf.get("LOGGER", "LEVEL")
LOGGER_FILENAME = cf.get("LOGGER", "FILENAME")

level = None
if LOGGER_LEVEL.upper() == 'NOTSET':
    level = 0
elif LOGGER_LEVEL.upper() == 'DEBUG':
    level = 10
elif LOGGER_LEVEL.upper() == 'INFO':
    level = 20
elif LOGGER_LEVEL.upper() == 'WARNING':
    level = 30
elif LOGGER_LEVEL.upper() == 'ERROR':
    level = 40
elif LOGGER_LEVEL.upper() == 'CRITICAL':
    level = 50
else:
    print("parameter error")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handler = logging.handlers.RotatingFileHandler(
              LOGGER_FILENAME, maxBytes=1024*1024, backupCount=5)
logging.basicConfig(filename=LOGGER_FILENAME, level=level, format=LOG_FORMAT)
logging.getLogger('').addHandler(handler)

fn_load_image = None
if INPUT_TYPE == 'TENSOR':
    if FN_LOAD_IMAGE == IMAGE_LOADER_TENSORFLOW:
        fn_load_image = create_tensorflow_image_loader(tf.Session(graph=tf.Graph()))
    elif FN_LOAD_IMAGE == IMAGE_LOADER_YAHOO:
        fn_load_image = create_yahoo_image_loader()
    else:
        print("parameter error")
elif INPUT_TYPE == 'BASE64_JPEG':
    if FN_LOAD_IMAGE == IMAGE_LOADER_YAHOO:
        print("parameter mismatch")
    elif FN_LOAD_IMAGE == IMAGE_LOADER_TENSORFLOW:
        import base64
        fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])
    else:
        print("parameter error")
else:
    print("parameter error")

model = OpenNsfwModel()
input_type = InputType[INPUT_TYPE]
model.build(weights_path="data/open_nsfw-weights.npy", input_type=input_type)
sess = tf.Session()


def process_start(conn, addr):
    logging.info("Connection address:" + str(addr))
    rec_d = bytes([])
    try:
        while True:
            data = conn.recv(BUFSIZ)
            if not data or len(data) == 0:
                break
            else:
                rec_d = rec_d + data
    except Exception as e:
        logging.error(str(e))
    sess.run(tf.global_variables_initializer())
    image = fn_load_image(rec_d)
    predictions = \
        sess.run(model.predictions,
                 feed_dict={model.input: image})
    conn.send("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]).encode())
    conn.shutdown(s.SHUT_RDWR)
    conn.close()


def main():
    processes = None
    if PROCESS_NUMBER <= 0:
        processes = Pool(multiprocessing.cpu_count())
    else:
        processes = Pool(PROCESS_NUMBER)
    server = socket()
    ip_port = (IP, PORT)
    server.bind(ip_port)
    server.listen(5)
    while True:
        logging.info("waiting for connection...")
        conn, addr = server.accept()
        processes.apply_async(process_start, (conn, addr))
    conn.close()
    sess.close()


if __name__ == "__main__":
    main()
