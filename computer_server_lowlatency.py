#V5 Low-latency UDP server with minimal buffering

import socket
import cv2
import struct
import logging
import numpy as np
import threading
from collections import deque
import time

# Configuration
HOST = '0.0.0.0'
VIDEO_PORT = 5000       # Receive frames from Pi
PI_IP = "10.0.0.60"     # Pi's IP address
PI_PORT = 5002          # Send processed frames to Pi

# Processing mode
current_mode = "normal"
mode_lock = threading.Lock()

# JPEG quality for sending back
JPEG_QUALITY = 60

logging.basicConfig(level=logging.INFO)

# ============== PROCESSING FUNCTIONS ==============

frame_count = 0

def process_normal(frame):
    """Normal with minimal overlay"""
    global frame_count
    frame_count += 1
    h, w = frame.shape[:2]
    
    # Minimal overlay - just a small indicator
    cv2.circle(frame, (w - 10, 10), 5, (0, 255, 0), -1)
    return frame

def process_canny(frame):
    """Fast edge detection"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

def process_night(frame):
    """Night vision - optimized"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    result = np.zeros_like(frame)
    result[:, :, 1] = enhanced
    return result

def process_thermal(frame):
    """Thermal view"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

def process_frame(frame):
    """Route to processor"""
    with mode_lock:
        mode = current_mode
    
    processors = {
        "normal": process_normal,
        "canny": process_canny,
        "night": process_night,
        "thermal": process_thermal,
    }
    return processors.get(mode, process_normal)(frame)

# ============== UDP NETWORKING ==============

def send_frame_udp(sock, frame, frame_id, dest):
    """Send processed frame back to Pi via UDP"""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    _, encoded = cv2.imencode('.jpg', frame, encode_params)
    data = encoded.tobytes()
    
    MAX_CHUNK = 60000
    total_chunks = (len(data) + MAX_CHUNK - 1) // MAX_CHUNK
    
    for i in range(total_chunks):
        start = i * MAX_CHUNK
        end = min(start + MAX_CHUNK, len(data))
        chunk = data[start:end]
        
        header = struct.pack('!IHHI', frame_id, i, total_chunks, len(data))
        try:
            sock.sendto(header + chunk, dest)
        except Exception as e:
            logging.error(f"Send error: {e}")

# ============== KEYBOARD CONTROL ==============

def keyboard_control():
    """Handle keyboard input for mode switching"""
    global current_mode
    
    print("\n=== Keyboard Controls ===")
    print("1 = normal")
    print("2 = canny (edges)")
    print("3 = night vision")
    print("4 = thermal")
    print("q = quit")
    print("=========================\n")
    
    while True:
        try:
            key = input()
            with mode_lock:
                if key == '1':
                    current_mode = "normal"
                    print(f"Mode: {current_mode}")
                elif key == '2':
                    current_mode = "canny"
                    print(f"Mode: {current_mode}")
                elif key == '3':
                    current_mode = "night"
                    print(f"Mode: {current_mode}")
                elif key == '4':
                    current_mode = "thermal"
                    print(f"Mode: {current_mode}")
                elif key == 'q':
                    break
        except:
            break

# ============== MAIN ==============

def main():
    # Create UDP sockets
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
    recv_sock.bind((HOST, VIDEO_PORT))
    recv_sock.settimeout(1.0)
    
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 262144)
    
    pi_dest = (PI_IP, PI_PORT)
    
    logging.info(f"UDP Server listening on port {VIDEO_PORT}")
    logging.info(f"Sending processed frames to {PI_IP}:{PI_PORT}")
    logging.info(f"Mode: {current_mode}")
    logging.info("Press 1-4 to change modes, q to quit")
    
    # Start keyboard control thread
    kb_thread = threading.Thread(target=keyboard_control, daemon=True)
    kb_thread.start()
    
    # Frame reassembly buffer
    buffer = {}
    frame_times = deque(maxlen=30)
    last_fps_print = time.time()
    
    try:
        while True:
            try:
                data, addr = recv_sock.recvfrom(65535)
                
                if len(data) < 12:
                    continue
                
                # Parse header
                frame_id, chunk_id, total_chunks, data_len = struct.unpack('!IHHI', data[:12])
                chunk_data = data[12:]
                
                # Store chunk
                if frame_id not in buffer:
                    buffer[frame_id] = {'chunks': {}, 'time': time.time()}
                buffer[frame_id]['chunks'][chunk_id] = chunk_data
                
                # Check if complete
                if len(buffer[frame_id]['chunks']) == total_chunks:
                    start_time = time.time()
                    
                    # Reassemble
                    frame_data = b''.join(buffer[frame_id]['chunks'][i] for i in range(total_chunks))
                    
                    # Decode JPEG
                    nparr = np.frombuffer(frame_data[:data_len], np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Process frame
                        processed = process_frame(frame)
                        
                        # Send back to Pi
                        send_frame_udp(send_sock, processed, frame_id, pi_dest)
                        
                        # FPS tracking
                        frame_times.append(time.time())
                        
                        # Print FPS every 2 seconds
                        if time.time() - last_fps_print > 2:
                            if len(frame_times) > 1:
                                fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                                process_time = (time.time() - start_time) * 1000
                                logging.info(f"FPS: {fps:.1f} | Process: {process_time:.1f}ms | Mode: {current_mode}")
                            last_fps_print = time.time()
                    
                    # Cleanup
                    del buffer[frame_id]
                    
                    # Remove old incomplete frames
                    old_frames = [fid for fid in buffer if fid < frame_id - 5]
                    for fid in old_frames:
                        del buffer[fid]
                
            except socket.timeout:
                continue
            except Exception as e:
                logging.error(f"Error: {e}")
                
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        recv_sock.close()
        send_sock.close()

if __name__ == "__main__":
    main()
