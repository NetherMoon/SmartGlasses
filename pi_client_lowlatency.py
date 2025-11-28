#V3 Low-latency version - UDP + JPEG compression + minimal buffering

import sys
import time
import logging
import cv2
import numpy as np
import socket
import struct
import threading
from PIL import Image

sys.path.append("..")
from lib import LCD_2inch

# Configuration
COMPUTER_IP = "10.0.0.57"
VIDEO_PORT = 5000
RECEIVE_PORT = 5002  # Port to receive processed frames on Pi

# JPEG quality (lower = smaller = faster, but lower quality)
# 50-70 is good balance for low latency
JPEG_QUALITY = 60

logging.basicConfig(level=logging.INFO)

def create_udp_sender():
    """Create UDP socket for sending frames"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)
    return sock

def create_udp_receiver(port):
    """Create UDP socket for receiving frames"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65535)
    sock.bind(('0.0.0.0', port))
    sock.settimeout(0.1)  # 100ms timeout
    return sock

# Frame receiver thread
latest_frame = None
frame_lock = threading.Lock()

def receive_frames():
    """Receive processed frames from computer"""
    global latest_frame
    
    recv_sock = create_udp_receiver(RECEIVE_PORT)
    logging.info(f"Listening for processed frames on port {RECEIVE_PORT}")
    
    buffer = {}
    
    while True:
        try:
            # Receive packet
            data, addr = recv_sock.recvfrom(65535)
            
            if len(data) < 12:
                continue
            
            # Parse header: frame_id (4), chunk_id (2), total_chunks (2), data_len (4)
            frame_id, chunk_id, total_chunks, data_len = struct.unpack('!IHHI', data[:12])
            chunk_data = data[12:]
            
            # Store chunk
            if frame_id not in buffer:
                buffer[frame_id] = {}
            buffer[frame_id][chunk_id] = chunk_data
            
            # Check if frame complete
            if len(buffer[frame_id]) == total_chunks:
                # Reassemble frame
                frame_data = b''.join(buffer[frame_id][i] for i in range(total_chunks))
                
                # Decode JPEG
                nparr = np.frombuffer(frame_data[:data_len], np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    with frame_lock:
                        latest_frame = frame
                
                # Clean up old frames
                old_frames = [fid for fid in buffer if fid < frame_id - 2]
                for fid in old_frames:
                    del buffer[fid]
                del buffer[frame_id]
                
        except socket.timeout:
            continue
        except Exception as e:
            logging.error(f"Receive error: {e}")

def send_frame_udp(sock, frame, frame_id, dest):
    """Send frame via UDP with chunking"""
    # Encode as JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    _, encoded = cv2.imencode('.jpg', frame, encode_params)
    data = encoded.tobytes()
    
    # Chunk size (leave room for header)
    MAX_CHUNK = 60000
    total_chunks = (len(data) + MAX_CHUNK - 1) // MAX_CHUNK
    
    for i in range(total_chunks):
        start = i * MAX_CHUNK
        end = min(start + MAX_CHUNK, len(data))
        chunk = data[start:end]
        
        # Header: frame_id (4), chunk_id (2), total_chunks (2), total_data_len (4)
        header = struct.pack('!IHHI', frame_id, i, total_chunks, len(data))
        sock.sendto(header + chunk, dest)

try:
    # Initialize display
    disp = LCD_2inch.LCD_2inch()
    disp.Init()
    disp.clear()
    
    logging.info("Opening camera with low-latency settings...")
    
    # Low-latency GStreamer pipeline
    # - Lower resolution for speed
    # - Minimal buffering
    pipeline = (
        "libcamerasrc ! "
        "video/x-raw,width=320,height=240,framerate=30/1 ! "
        "videoconvert ! "
        "appsink drop=true max-buffers=1 sync=false"
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        # Fallback pipeline
        logging.warning("Trying fallback pipeline...")
        pipeline = (
            "libcamerasrc ! "
            "videoconvert ! videoscale ! "
            "video/x-raw,width=320,height=240 ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        logging.error("Failed to open camera")
        exit()
    
    # Set OpenCV buffer size to minimum
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Create UDP sender
    send_sock = create_udp_sender()
    dest = (COMPUTER_IP, VIDEO_PORT)
    
    # Start receiver thread
    recv_thread = threading.Thread(target=receive_frames, daemon=True)
    recv_thread.start()
    
    logging.info(f"Streaming to {COMPUTER_IP}:{VIDEO_PORT} via UDP")
    logging.info("Low-latency mode active!")
    
    frame_id = 0
    last_display_time = 0
    
    while True:
        # Grab frame (non-blocking style)
        ret = cap.grab()
        if not ret:
            continue
        
        ret, frame = cap.retrieve()
        if not ret or frame is None:
            continue
        
        # Send frame via UDP
        send_frame_udp(send_sock, frame, frame_id, dest)
        frame_id = (frame_id + 1) % 65536
        
        # Display latest processed frame (or raw if none received yet)
        with frame_lock:
            display_frame = latest_frame if latest_frame is not None else frame
        
        # Convert and display
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Resize to display size if needed
        frame_rgb = cv2.resize(frame_rgb, (240, 320))
        
        img = Image.fromarray(frame_rgb)
        img = img.rotate(180)
        disp.ShowImage(img)

except KeyboardInterrupt:
    logging.info("Interrupted by user.")
except Exception as e:
    logging.error(e)
    import traceback
    traceback.print_exc()
finally:
    if 'cap' in locals():
        cap.release()
    if 'send_sock' in locals():
        send_sock.close()
    if 'disp' in locals():
        disp.module_exit()
    logging.info("Exiting.")
