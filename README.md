# Smart Glasses Video Streaming Setup

This project enables streaming video from a Raspberry Pi Zero 2W with camera and displays to a computer for processing.

## Files

- `pi_client.py` - Run this on your Raspberry Pi Zero 2W
- `computer_server.py` - Run this on your computer

## Setup Instructions

### 1. On Your Computer

First, find your computer's IP address:
- **Windows**: Open Command Prompt and run `ipconfig`, look for "IPv4 Address"
- **Mac/Linux**: Open Terminal and run `ifconfig` or `ip addr`, look for inet address

Then run the server:
```bash
python computer_server.py
```

### 2. On Your Raspberry Pi

Edit `pi_client.py` and change line 17 to your computer's IP address:
```python
COMPUTER_IP = "192.168.1.100"  # Change this to your computer's IP address
```

Then run:
```bash
python pi_client.py
```

## How It Works

1. **Raspberry Pi** captures video from the camera
2. Sends each frame over the network to the **Computer**
3. **Computer** processes the frame (currently applies Canny edge detection)
4. Sends the processed frame back to the **Raspberry Pi**
5. **Raspberry Pi** displays the processed video on the LCD screens

## Customizing the Processing

To change what processing is done, edit the `process_frame()` function in `computer_server.py`. 

Current processing: Canny edge detection (shows edges in the video)

Example alternatives:
- Face detection
- Object detection
- Color filters
- Blur effects
- Text overlay
- etc.

## Troubleshooting

- Make sure both devices are on the same network
- Check firewall settings if connection fails
- Ensure port 5000 is not blocked
- Verify the IP address is correct in `pi_client.py`
