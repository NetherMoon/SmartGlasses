#V4 With multiple creative modes


import socket
import cv2
import pickle
import struct
import logging
import numpy as np
import threading
import speech_recognition as sr
import io
import wave
from datetime import datetime

# Configuration
HOST = '0.0.0.0'  # Listen on all network interfaces
PORT = 5000
AUDIO_PORT = 5001  # Separate port for audio

# ============== MODE SELECTION ==============
# Available modes:
#   "normal"    - Normal camera with overlay
#   "canny"     - Canny edge detection (black & white edges)
#   "night"     - Night vision (green tint + brightness boost)
#   "thermal"   - Fake thermal camera effect
# 
# Voice commands: "mode <number>" or "mode <name>"
current_mode = "normal"
mode_lock = threading.Lock()
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

def process_night(frame):
    """
    Night vision effect - green tint with enhanced brightness
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast and brightness
    enhanced = cv2.equalizeHist(gray)
    
    # Apply slight blur to reduce noise
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Create green-tinted image
    night_vision = np.zeros_like(frame)
    night_vision[:, :, 1] = enhanced  # Green channel only
    
    # Add scan lines for effect
    for i in range(0, frame.shape[0], 3):
        night_vision[i, :, :] = night_vision[i, :, :] * 0.7
    
    return night_vision

def process_thermal(frame):
    """
    Fake thermal camera effect using color mapping
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply thermal colormap
    thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    
    # Add slight blur for thermal camera look
    thermal = cv2.GaussianBlur(thermal, (5, 5), 0)
    
    return thermal

def process_frame(frame):
    """
    Route to the appropriate processing function based on current_mode
    """
    global current_mode
    with mode_lock:
        mode = current_mode
    
    processors = {
        "normal": process_normal,
        "canny": process_canny,
        "night": process_night,
        "thermal": process_thermal,
    }
    
    processor = processors.get(mode, process_normal)
    return processor(frame)


def parse_voice_command(text):
    """
    Parse voice command and return new mode if valid
    """
    text = text.lower().strip()
    logging.info(f"Heard: '{text}'")
    
    # Check for mode change commands
    if "mode" in text or "camera" in text or any(str(i) in text for i in range(1, 5)):
        # Map voice words to modes (including number alternatives)
        mode_keywords = {
            "normal": ["normal", "regular", "standard", "default", "1", "one", "won"],
            "canny": ["canny", "edge", "edges", "candy", "2", "two", "to", "too"],
            "night": ["night", "night vision", "green", "dark", "3", "three", "tree"],
            "thermal": ["thermal", "heat", "infrared", "thermo", "4", "four", "for"],
        }
        
        for mode, keywords in mode_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    return mode
    
    return None


def audio_server_thread():
    """
    Separate thread to handle audio input and speech recognition
    """
    global current_mode
    
    recognizer = sr.Recognizer()
    
    audio_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    audio_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    audio_socket.bind((HOST, AUDIO_PORT))
    audio_socket.listen(1)
    
    logging.info(f"Audio server listening on {HOST}:{AUDIO_PORT}")
    
    while True:
        try:
            conn, addr = audio_socket.accept()
            logging.info(f"Audio connection from {addr}")
            
            while True:
                # Receive header: sample_rate (4 bytes) + audio_size (4 bytes)
                header_data = b""
                while len(header_data) < 8:
                    packet = conn.recv(8 - len(header_data))
                    if not packet:
                        break
                    header_data += packet
                
                if len(header_data) < 8:
                    break
                
                sample_rate, audio_size = struct.unpack("!II", header_data)
                
                # Receive audio data
                audio_data = b""
                while len(audio_data) < audio_size:
                    packet = conn.recv(4096)
                    if not packet:
                        break
                    audio_data += packet
                
                if len(audio_data) < audio_size:
                    break
                
                # Convert to AudioData for speech recognition
                try:
                    # Audio is sent as raw PCM data with dynamic sample rate
                    # sample_width=2 for paInt16 (16 bits = 2 bytes)
                    audio = sr.AudioData(audio_data, sample_rate, 2)
                    
                    # Use Google Speech Recognition (free, no API key needed)
                    text = recognizer.recognize_google(audio)
                    
                    # Parse command
                    new_mode = parse_voice_command(text)
                    if new_mode:
                        with mode_lock:
                            old_mode = current_mode
                            current_mode = new_mode
                        logging.info(f"Mode changed: {old_mode} -> {new_mode}")
                        # Send confirmation back
                        conn.sendall(f"MODE:{new_mode}".encode())
                    else:
                        conn.sendall(b"OK")
                        
                except sr.UnknownValueError:
                    conn.sendall(b"OK")  # Could not understand audio
                except sr.RequestError as e:
                    logging.error(f"Speech recognition error: {e}")
                    conn.sendall(b"OK")
                    
        except Exception as e:
            logging.error(f"Audio server error: {e}")
        finally:
            conn.close()

def main():
    # Start audio server in separate thread
    audio_thread = threading.Thread(target=audio_server_thread, daemon=True)
    audio_thread.start()
    
    # Create socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    
    logging.info(f"Server listening on {HOST}:{PORT}")
    logging.info(f"Processing mode: {current_mode}")
    logging.info("Modes: 1=normal, 2=canny, 3=night, 4=thermal")
    logging.info("Voice command: 'mode <number>' or 'mode <name>'")
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
