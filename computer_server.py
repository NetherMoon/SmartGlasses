#V2 With processing modes


import socket
import cv2
import pickle
import struct
import logging
import numpy as np
from datetime import datetime

# Configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5000

# ============== MODE SELECTION ==============
# Available modes:
#   "canny"  - Canny edge detection (black & white edges)
#   "normal" - Normal camera with overlay text
MODE = "canny"
# ============================================

logging.basicConfig(level=logging.INFO)

# Frame counter for overlay
frame_count = 0

def process_canny(frame):
    """
    Canny edge detection - black & white edge view
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_bgr

def process_normal(frame):
    """
    Normal camera view with overlay to show server is processing
    """
    global frame_count
    frame_count += 1
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Add semi-transparent overlay bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 30), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Add "SERVER PROCESSING" text
    cv2.putText(frame, "SERVER", (5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Add frame counter
    cv2.putText(frame, f"#{frame_count}", (w - 60, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Add small corner indicator (green dot = connected)
    cv2.circle(frame, (w - 10, h - 10), 5, (0, 255, 0), -1)
    
    return frame

def process_frame(frame):
    """
    Route to the appropriate processing function based on MODE
    """
    if MODE == "canny":
        return process_canny(frame)
    elif MODE == "normal":
        return process_normal(frame)
    else:
        logging.warning(f"Unknown mode: {MODE}, defaulting to normal")
        return process_normal(frame)

def main():
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    logging.info(f"Server listening on {HOST}:{PORT}")
    logging.info(f"Processing mode: {MODE}")
    logging.info("Waiting for Raspberry Pi connection...")
    
    try:
        while True:
            # Accept connection from Raspberry Pi
            conn, addr = server_socket.accept()
            logging.info(f"Connection from {addr}")
            
            data = b""
            payload_size = 4  # Fixed 4 bytes for size (using "!I" format)
            
            try:
                while True:
                    # Receive message size
                    while len(data) < payload_size:
                        packet = conn.recv(4096)
                        if not packet:
                            raise ConnectionError("Connection closed by client")
                        data += packet
                    
                    packed_msg_size = data[:payload_size]
                    data = data[payload_size:]
                    msg_size = struct.unpack("!I", packed_msg_size)[0]
                    
                    # Receive frame data
                    while len(data) < msg_size:
                        packet = conn.recv(4096)
                        if not packet:
                            raise ConnectionError("Connection closed by client")
                        data += packet
                    
                    frame_data = data[:msg_size]
                    data = data[msg_size:]
                    
                    # Deserialize frame
                    frame = pickle.loads(frame_data)
                    
                    # Process the frame
                    processed_frame = process_frame(frame)
                    
                    # Serialize processed frame
                    processed_data = pickle.dumps(processed_frame, protocol=2)
                    message_size = struct.pack("!I", len(processed_data))
                    
                    # Send processed frame back to Pi
                    conn.sendall(message_size + processed_data)
                    
                    logging.info(f"Processed frame: {frame.shape}")
                    
            except (ConnectionError, BrokenPipeError, EOFError) as e:
                logging.warning(f"Connection lost: {e}")
            finally:
                conn.close()
                logging.info("Client disconnected. Waiting for new connection...")
                
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        server_socket.close()
        logging.info("Server shut down.")

if __name__ == "__main__":
    main()
