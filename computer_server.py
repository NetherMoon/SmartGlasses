#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import socket
import cv2
import pickle
import struct
import logging
import numpy as np

# Configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5000

logging.basicConfig(level=logging.INFO)

def process_frame(frame):
    """
    Simple image processing - applies Canny edge detection
    You can modify this function to do any processing you want
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Convert back to BGR for consistency
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    return edges_bgr

def main():
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    logging.info(f"Server listening on {HOST}:{PORT}")
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
