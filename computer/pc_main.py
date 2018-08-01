import numpy as np
import cv2
import pygame
import threading
import socketserver
import socket
from queue import Queue
import time
import pandas as pd
from keras.models import model_from_yaml
import os

RUN = True
COLLECT = False
q = Queue(100)

class Control(object):
    'keeps track of steering during training'
    'parses steering state from neural network'
    def __init__(self):
        self.mag   = 300
        self.rate  = self.mag / 2
        self.third = self.mag / 3
        self.range = 2 * (self.mag - self.third)
        self.left  = self.mag
        self.right = self.mag
        
        self.colors = []
        
    def left_pressed(self):
        if self.right < self.mag:
            self.right += 1
        elif self.left > self.third:
            self.left -= 1
    
    def right_pressed(self):
        if self.left < self.mag:
            self.left += 1
        elif self.right > self.third:
            self.right -= 1
    
    def get_target(self):
        if self.left < self.mag:
            return self.left
        else:
            diff = self.mag - self.right
            return self.mag + diff
        
    def set_rates(self, y):
        if y < 0.0:
            y = 0
        elif y > 1.0:
            y = 1
        else :
            y = (y * self.range) + self.third
        
        if y < self.mag:
            self.left = int(y)
            self.right = self.mag
        if y == self.mag:
            self.left = self.mag
            self.right = self.mag
        if y > self.mag:
            diff = int(y) - self.mag
            self.left = self.mag
            self.right = self.mag - diff

class VideoStreamHandler(socketserver.StreamRequestHandler):
    'parses images as they arrive and pushes them into a queue'
    def handle(self):
        
        global RUN, q
        fps = []
        
        print("Streaming...")
        
        start_time = time.time()
        stream_bytes = bytearray()
        
        while RUN:
            
            stream_bytes += self.rfile.read(1024)
            first = stream_bytes.find(b'\xff\xd8')
            last = stream_bytes.find(b'\xff\xd9')
            
            if first != -1 and last != -1:
                delta_time = time.time() - start_time
                fps.append(delta_time)
                if len(fps) > 30:
                    fps.pop(0)
                rate = 1/(np.sum(fps)/len(fps))
                #print(int(rate))
                start_time = time.time()
                
                jpg = stream_bytes[first:last + 2]
                stream_bytes = stream_bytes[last + 2:]
                arr = np.asarray(jpg, dtype=np.uint8)
                image = cv2.imdecode(arr, 1)
                q.put(image)

class ImageHandler(object):
    'processes images as soon as they arrive'
    def __init__(self):
        
        self.control = Control()
        self.cnt = 0
        self.image = np.zeros((200,66))
        self.target = []
        
        if not COLLECT:
            self.load_models()
    
    def load_models(self):
        os.chdir('results')        
        yaml_file = open('nvidia.yaml', 'r')
        model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(model_yaml)
        model.load_weights('nvidia.h5')
        model.compile(loss='mse', optimizer='adam')
        self.model = model
        self.model._make_predict_function()
        
        os.chdir('..')
    
    def create_csv(self):
        df = pd.DataFrame(self.target, columns=['target'])
        df.to_csv('images/target.csv', index=False)
        
    def process_image(self):
        'receives images. either stores images or evaluates using nn model'
        if q.empty() is False:
            image = q.get()
            #image is bgr format
            image = np.rot90(image, k=2)
            self.image = image
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            blue_lower = np.array([ 80,  0,  0])
            blue_upper = np.array([120,255,255])
            mask = cv2.inRange(hsv, blue_lower, blue_upper)
            blue_image = cv2.bitwise_and(image, image, mask=mask)
            
            if COLLECT:
                
                cv2.imwrite('images/color/image_'+str(self.cnt)+'.jpeg', image)
                cv2.imwrite('images/gray/image_'+str(self.cnt)+'.jpeg', gray)
                cv2.imwrite('images/blue/image_'+str(self.cnt)+'.jpeg', blue_image)
                self.target.append(self.control.get_target())
                self.cnt += 1
                
            else:
                
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                X = rgb.astype(np.float32).reshape(1, 66, 200, 3)
                y = self.model.predict(X)[0][0]
                self.control.set_rates(y)
                
            return True
        
        else:
            return False

    def query_keyboard(self):
        'checks which arrow keys are pressed and updates steering'
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.control.left_pressed()
        if keys[pygame.K_RIGHT]:
            self.control.right_pressed()
            
        left_rate = str(self.control.left)
        right_rate = str(self.control.right)
        
        msg = 'L'+left_rate+'R'+right_rate+'E'
        self.client_socket.sendall(str.encode(msg))
        
    def update_loop(self, host, port):
        'checks for new images and pushes to pygame windows'
        'queries keyboard and sends relevant messages to rpi'
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((host, port))
        pygame.init()
        screen = pygame.display.set_mode((200,66))
        key_time = time.time()
        img_time = time.time()
        
        global RUN, q
        
        try:
        
            while RUN:
                
                current = time.time()
                
                'checks for new images and updates display: 100 hertz'
                if current - img_time > 1/100:
                    
                    new_image = self.process_image()
                    if new_image:
                        screen.fill([0,0,0])
                        frame = self.image
                        frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                        frame = cv2.flip(frame, 0)
                        frame = np.rot90(frame, k=3)
                        frame = pygame.surfarray.make_surface(frame)
                        screen.blit(frame, (0,0))
                        pygame.display.update()
                    img_time = current
                
                'updates drive state: 150 hertz'
                if current - key_time > 1/self.control.rate:
                    
                    self.query_keyboard()
                    key_time = current
                    
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            RUN = False
                    
        finally:
            if COLLECT:
                self.create_csv()
            self.client_socket.close()
            cv2.destroyAllWindows()
            pygame.quit()

class ManageTCPServer(object):
    'allows for tcp server to be shutdown from main'
    def setup_server(self, host, port):
        self.server = socketserver.TCPServer(server_address=(host, port),
                                        RequestHandlerClass=VideoStreamHandler)
        self.server.serve_forever()

    def shutdown_server(self):
        self.server.shutdown()

if __name__ == '__main__':
    manager = ManageTCPServer()
    video_thread  = threading.Thread(target=manager.setup_server, args=('computer_ip_address', 8000))
    video_thread.start()
    ih = ImageHandler()
    
    while True:
        if q.empty() is False:
            master_thread = threading.Thread(target=ih.update_loop, args=('rpi_ip_address', 8001))
            master_thread.start()
            break
        
    master_thread.join()
    manager.shutdown_server()
