#V2 With voice command support - fixed sample rate

import sys
import time
import logging
import cv2
import numpy as np
import socket
import pickle
import struct
import threading
import pyaudio
from PIL import Image

sys.path.append("..")
from lib import LCD_2inch

COMPUTER_IP = "10.0.0.57"
COMPUTER_PORT = 5000
AUDIO_PORT = 5001

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 3

logging.basicConfig(level=logging.INFO)

current_mode = "canny"
audio_socket = None
mode_lock = threading.Lock()

def get_supported_sample_rate(p, device_index):
    """Find a supported sample rate for the device"""
    rates_to_try = [44100, 48000, 22050, 16000, 8000]
    
    for rate in rates_to_try:
        try:
            if p.is_format_supported(rate,
                                     input_device=device_index,
                                     input_channels=CHANNELS,
                                     input_format=FORMAT):
                return rate
        except ValueError:
            continue
    return None

def record_and_send_audio():
    global audio_socket, current_mode
    try:
        p = pyaudio.PyAudio()
        
        mic_index = None
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                name = dev['name']
                logging.info(f"Found input device {i}: {name}")
                if 'usb' in name.lower() or 'hw:1' in name.lower():
                    mic_index = i
                    break
                elif mic_index is None:
                    mic_index = i
        
        if mic_index is None:
            logging.error("No microphone found!")
            return
        
        sample_rate = get_supported_sample_rate(p, mic_index)
        if sample_rate is None:
            dev_info = p.get_device_info_by_index(mic_index)
            sample_rate = int(dev_info['defaultSampleRate'])
        
        logging.info(f"Using mic index {mic_index} with sample rate {sample_rate}")
        logging.info("Listening for voice commands...")
        
        while True:
            try:
                stream = p.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=sample_rate,
                              input=True,
                              input_device_index=mic_index,
                              frames_per_buffer=CHUNK)
                
                frames = []
                for _ in range(0, int(sample_rate / CHUNK * RECORD_SECONDS)):
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                
                stream.stop_stream()
                stream.close()
                
                audio_data = b''.join(frames)
                
                if audio_socket:
                    try:
                        header = struct.pack("!II", sample_rate, len(audio_data))
                        audio_socket.sendall(header + audio_data)
                        
                        response = audio_socket.recv(1024).decode()
                        if response.startswith("MODE:"):
                            new_mode = response.split(":")[1]
                            with mode_lock:
                                current_mode = new_mode
                            logging.info(f"Mode changed to: {new_mode}")
                    except Exception as e:
                        logging.error(f"Error sending audio: {e}")
                
                time.sleep(0.5)
                
            except Exception as e:
                logging.error(f"Recording error: {e}")
                time.sleep(1)
                
    except Exception as e:
        logging.error(f"Audio thread error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        p.terminate()

try:
    # Connect to audio server
    logging.info(f"Connecting to audio server at {COMPUTER_IP}:{AUDIO_PORT}...")
    audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    audio_socket.connect((COMPUTER_IP, AUDIO_PORT))
    logging.info("Connected to audio server!")
    
    # Start audio thread
    audio_thread = threading.Thread(target=record_and_send_audio, daemon=True)
    audio_thread.start()
    
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

    logging.info(f"Connecting to video server at {COMPUTER_IP}:{COMPUTER_PORT}...")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((COMPUTER_IP, COMPUTER_PORT))
    logging.info("Connected to video server!")

    logging.info("Streaming to computer. Say 'camera mode canny' or 'camera mode normal' to switch.")
    
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
    if 'audio_socket' in locals() and audio_socket:
        audio_socket.close()
    disp.module_exit()
    logging.info("Camera stream closed. Exiting.")