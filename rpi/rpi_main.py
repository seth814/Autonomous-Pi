"""
start computer_main (server) first.
Once images arrive, the computer launches a client to complete the message connection.

Reference:
PiCamera documentation
https://picamera.readthedocs.io/en/release-1.13/recipes2.html#rapid-capture-and-streaming
picamera acts as a client and streams using split frames.
split frames is recommended (30 fps). the first method topped out at 18 fps for me.
an enable signal is being set from enable.py to turn on motors.
this is optional according to your needs.
"""

import io
import socket
import struct
import time
import picamera
import threading
from gpiozero import DigitalOutputDevice as GPIO
from subprocess import call
from Queue import Queue
import serial

call('sudo pkill -9 -f enable.py', shell=True)
time.sleep(0.1)
GPIO(pin=26).off()
RUN = True
q = Queue(10)

class SplitFrames(object):
    def __init__(self, connection):
        self.connection = connection
        self.stream = io.BytesIO()
        self.count = 0

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; send the old one's length
            # then the data
            size = self.stream.tell()
            if size > 0:
                self.connection.write(struct.pack('<L', size))
                self.connection.flush()
                self.stream.seek(0)
                self.connection.write(self.stream.read(size))
                self.count += 1
                self.stream.seek(0)
        self.stream.write(buf)

def server(host, port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(0)
    connection, client_address = server_socket.accept()

    ser = serial.Serial('/dev/ttyS0', baudrate=57600)

    msg_start = time.time()

    global RUN

    try:
        while RUN:
            current = time.time()
            #100 hertz update rate
            if (current - msg_start) > 0.01:
                msg = connection.recv(1024)
                if msg:
                    ser.write(msg)
                msg_start = current

    finally:
        ser.write('L0R0E')
        ser.close()
        connection.close()
        server_socket.close()


def client_split(host, port):
    'recommended picamera streaming method. high fps is achievable.'
    client_socket = socket.socket()
    client_socket.connect((host, port))
    connection = client_socket.makefile('wb')
    try:
        output = SplitFrames(connection)
        with picamera.PiCamera(resolution=(200,66), framerate=30) as camera:
            time.sleep(2)
            camera.start_recording(output, format='mjpeg')
            camera.wait_recording(float('inf'))
            camera.stop_recording()
            # Write the terminating 0-length to the connection to let the
            # server know we're done
            connection.write(struct.pack('<L', 0))
            
    except socket.error:
        global RUN
        RUN = False
        connection.close()
        client_socket.close()
        
def client(host, port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    connection = client_socket.makefile('wb')

    try:
        with picamera.PiCamera() as camera:
            camera.resolution = (200, 66)       # pi camera resolution
            camera.framerate = 10               # frames/sec
            time.sleep(2)                       # give 2 secs for camera to initilize
            stream = io.BytesIO()
        
            # send jpeg format video stream
            for _ in camera.capture_continuous(stream, 'jpeg', use_video_port=True):
                connection.write(struct.pack('<L', stream.tell()))
                connection.flush()
                stream.seek(0)
                connection.write(stream.read())
                stream.seek(0)
                stream.truncate()
            connection.write(struct.pack('<L', 0))

    except socket.error:
        global RUN
        RUN = False
        connection.close()
        client_socket.close()

if __name__ == '__main__':

    video_thread   = threading.Thread(target=client_split, args=('computer_ip_address', 8000))
    master_thread  = threading.Thread(target=server, args=('rpi_ip_address', 8001))
    
    master_thread.start()
    video_thread.start()
    
    master_thread.join()
    video_thread.join()
    
    call('sudo python enable.py &', shell=True)
