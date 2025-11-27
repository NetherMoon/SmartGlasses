#V1 Which works with cool black-white

import sys
import time
import logging
import cv2
import numpy as np
import socket
import pickle
import struct
from PIL import Image

sys.path.append("..")
from lib import LCD_2inch

# Configuration
COMPUTER_IP = "10.0.0.57"
COMPUTER_PORT = 5000

logging.basicConfig(level=logging.INFO)

try:
    disp = LCD_2inch.LCD_2inch()
    disp.Init()
    disp.clear()
    WIDTH = disp.height
    HEIGHT = disp.width

    logging.info("Opening camera stream...")
    cap = cv2.VideoCapture("libcamerasrc ! videoconvert ! videoscale ! video/x-raw,width=240,height=320 ! appsink", cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        logging.error("Failed to open camera stream.")
        exit()

    logging.info(f"Connecting to computer at {COMPUTER_IP}:{COMPUTER_PORT}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((COMPUTER_IP, COMPUTER_PORT))
    logging.info("Connected to computer!")

    logging.info("Streaming to computer and LCD. Press Ctrl+C to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("No frame received.")
            continue

        data = pickle.dumps(frame, protocol=2)
        message_size = struct.pack("!I", len(data))
        
        try:
            client_socket.sendall(message_size + data)
            
            size_data = b""
            while len(size_data) < 4:
                packet = client_socket.recv(4 - len(size_data))
                if not packet:
                    raise ConnectionError("Connection lost")
                size_data += packet
            
            packed_msg_size = struct.unpack("!I", size_data)[0]
            
            frame_data = b""
            while len(frame_data) < packed_msg_size:
                packet = client_socket.recv(4096)
                if not packet:
                    raise ConnectionError("Connection lost")
                frame_data += packet
            
            processed_frame = pickle.loads(frame_data)
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.rotate(180)
            disp.ShowImage(img)
            
        except (ConnectionError, BrokenPipeError) as e:
            logging.error(f"Connection error: {e}")
            break

except KeyboardInterrupt:
    logging.info("Interrupted by user.")
except Exception as e:
    logging.error(e)
    import traceback
    traceback.print_exc()
finally:
    cap.release()
    if 'client_socket' in locals():
        client_socket.close()
    disp.module_exit()
    logging.info("Camera stream closed. Exiting.")