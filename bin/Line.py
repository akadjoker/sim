import socket
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from datetime import datetime
import time
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
from collections import deque

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LaneFollower')

def _clamp(value, limits):
    lower, upper = limits
    if value is None:
        return None
    elif (upper is not None) and (value > upper):
        return upper
    elif (lower is not None) and (value < lower):
        return lower
    return value


DATASET_DIR = "dataset_unity"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

class DatasetRecorder:
    def __init__(self, base_dir="dataset_unity"):
        self.base_dir = base_dir
        self.recording = False
        self.frame_counter = 0
        self.session_dir = None
        
        # Criar diretório base se não existir
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    
    def start_recording(self):
        """Inicia uma nova sessão de gravação"""
        if not self.recording:
            # Criar diretório para a sessão atual com timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = os.path.join(self.base_dir, f"session_{timestamp}")
            os.makedirs(self.session_dir)
            
            # Criar arquivo para armazenar os comandos
            self.labels_file = open(os.path.join(self.session_dir, "labels.csv"), "w")
            self.labels_file.write("frame,speed,steering_angle\n")
            
            self.recording = True
            self.frame_counter = 0
            logger.info(f"Iniciando gravação na sessão: {self.session_dir}")
    
    def stop_recording(self):
        """Para a gravação atual"""
        if self.recording:
            self.labels_file.close()
            self.recording = False
            logger.info(f"Gravação finalizada. Total de frames: {self.frame_counter}")
    
    def save_frame(self, frame, speed, steering_angle):
        """Salva um frame e seus comandos associados"""
        if self.recording:
            # Salvar imagem
            frame_path = os.path.join(self.session_dir, f"frame_{self.frame_counter:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Salvar comandos no arquivo CSV
            self.labels_file.write(f"{self.frame_counter:06d},{speed},{steering_angle }\n")
            self.labels_file.flush()  # Garantir que os dados sejam escritos
            
            self.frame_counter += 1

class PID(object):
    """A simple PID controller."""

    def __init__(
        self,
        Kp=1.0,
        Ki=0.0,
        Kd=0.0,
        setpoint=0,
        sample_time=0.01,
        output_limits=(None, None),
        auto_mode=True,
        proportional_on_measurement=False,
        differential_on_measurement=True,
        error_map=None,
        time_fn=None,
        starting_output=0.0,
    ):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.sample_time = sample_time

        self._min_output, self._max_output = None, None
        self._auto_mode = auto_mode
        self.proportional_on_measurement = proportional_on_measurement
        self.differential_on_measurement = differential_on_measurement
        self.error_map = error_map

        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._last_time = None
        self._last_output = None
        self._last_error = None
        self._last_input = None

        if time_fn is not None:
            # Use the user supplied time function
            self.time_fn = time_fn
        else:
            import time

            try:
                # Get monotonic time to ensure that time deltas are always positive
                self.time_fn = time.monotonic
            except AttributeError:
                # time.monotonic() not available (using python < 3.3), fallback to time.time()
                self.time_fn = time.time

        self.output_limits = output_limits
        self.reset()

        # Set initial state of the controller
        self._integral = _clamp(starting_output, output_limits)

    def __call__(self, input_, dt=None):
        if not self.auto_mode:
            return self._last_output

        now = self.time_fn()
        if dt is None:
            dt = now - self._last_time if (now - self._last_time) else 1e-16
        elif dt <= 0:
            raise ValueError('dt has negative value {}, must be positive'.format(dt))

        if self.sample_time is not None and dt < self.sample_time and self._last_output is not None:
            # Only update every sample_time seconds
            return self._last_output

        # Compute error terms
        error = self.setpoint - input_
        d_input = input_ - (self._last_input if (self._last_input is not None) else input_)
        d_error = error - (self._last_error if (self._last_error is not None) else error)

        # Check if must map the error
        if self.error_map is not None:
            error = self.error_map(error)

        # Compute the proportional term
        if not self.proportional_on_measurement:
            # Regular proportional-on-error, simply set the proportional term
            self._proportional = self.Kp * error
        else:
            # Add the proportional error on measurement to error_sum
            self._proportional -= self.Kp * d_input

        # Compute integral and derivative terms
        self._integral += self.Ki * error * dt
        self._integral = _clamp(self._integral, self.output_limits)  # Avoid integral windup

        if self.differential_on_measurement:
            self._derivative = -self.Kd * d_input / dt
        else:
            self._derivative = self.Kd * d_error / dt

        # Compute final output
        output = self._proportional + self._integral + self._derivative
        output = _clamp(output, self.output_limits)

        # Keep track of state
        self._last_output = output
        self._last_input = input_
        self._last_error = error
        self._last_time = now

        return output

    def __repr__(self):
        return (
            '{self.__class__.__name__}('
            'Kp={self.Kp!r}, Ki={self.Ki!r}, Kd={self.Kd!r}, '
            'setpoint={self.setpoint!r}, sample_time={self.sample_time!r}, '
            'output_limits={self.output_limits!r}, auto_mode={self.auto_mode!r}, '
            'proportional_on_measurement={self.proportional_on_measurement!r}, '
            'differential_on_measurement={self.differential_on_measurement!r}, '
            'error_map={self.error_map!r}'
            ')'
        ).format(self=self)

    @property
    def components(self):
  
        return self._proportional, self._integral, self._derivative

    @property
    def tunings(self):
        """The tunings used by the controller as a tuple: (Kp, Ki, Kd)."""
        return self.Kp, self.Ki, self.Kd

    @tunings.setter
    def tunings(self, tunings):
        """Set the PID tunings."""
        self.Kp, self.Ki, self.Kd = tunings

    @property
    def auto_mode(self):
        """Whether the controller is currently enabled (in auto mode) or not."""
        return self._auto_mode

    @auto_mode.setter
    def auto_mode(self, enabled):
        """Enable or disable the PID controller."""
        self.set_auto_mode(enabled)

    def set_auto_mode(self, enabled, last_output=None):
        if enabled and not self._auto_mode:
            # Switching from manual mode to auto, reset
            self.reset()

            self._integral = last_output if (last_output is not None) else 0
            self._integral = _clamp(self._integral, self.output_limits)

        self._auto_mode = enabled

    @property
    def output_limits(self):
        return self._min_output, self._max_output

    @output_limits.setter
    def output_limits(self, limits):
        """Set the output limits."""
        if limits is None:
            self._min_output, self._max_output = None, None
            return

        min_output, max_output = limits

        if (None not in limits) and (max_output < min_output):
            raise ValueError('lower limit must be less than upper limit')

        self._min_output = min_output
        self._max_output = max_output

        self._integral = _clamp(self._integral, self.output_limits)
        self._last_output = _clamp(self._last_output, self.output_limits)

    def reset(self):
        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._integral = _clamp(self._integral, self.output_limits)

        self._last_time = self.time_fn()
        self._last_output = None
        self._last_input = None
        self._last_error = None



class LineFollower:
    '''
    OpenCV based controller
    This controller takes a horizontal slice of the image at a set Y coordinate.
    Then it converts to HSV and does a color thresh hold to find the yellow pixels.
    It does a histogram to find the pixel of maximum yellow. Then is uses that iPxel
    to guid a PID controller which seeks to maintain the max yellow at the same point
    in the image.
    '''
    def __init__(self, pid, cfg):
        self.overlay_image = cfg.OVERLAY_IMAGE
        self.scan_y = cfg.SCAN_Y   # num pixels from the top to start horiz scan
        self.scan_height = cfg.SCAN_HEIGHT  # num pixels high to grab from horiz scan
        self.color_thr_low = np.asarray(cfg.COLOR_THRESHOLD_LOW)  # hsv dark yellow
        self.color_thr_hi = np.asarray(cfg.COLOR_THRESHOLD_HIGH)  # hsv light yellow
        self.target_pixel = cfg.TARGET_PIXEL  # of the N slots above, which is the ideal relationship target
        self.target_threshold = cfg.TARGET_THRESHOLD # minimum distance from target_pixel before a steering change is made.
        self.confidence_threshold = cfg.CONFIDENCE_THRESHOLD  # percentage of yellow pixels that must be in target_pixel slice
        self.steering = 0.0 # from -1 to 1
        self.throttle = cfg.THROTTLE_INITIAL # from -1 to 1
        self.delta_th = cfg.THROTTLE_STEP  # how much to change throttle when off
        self.throttle_max = cfg.THROTTLE_MAX
        self.throttle_min = cfg.THROTTLE_MIN

        self.pid_st = pid


    def get_i_color(self, cam_img):
        '''
        get the horizontal index of the color at the given slice of the image
        input: cam_image, an RGB numpy array
        output: index of max color, value of cumulative color at that index, and mask of pixels in range
        '''
        # take a horizontal slice of the image
        iSlice = self.scan_y
        scan_line = cam_img[iSlice : iSlice + self.scan_height, :, :]

        # convert to HSV color space
        img_hsv = cv2.cvtColor(scan_line, cv2.COLOR_RGB2HSV)

        # make a mask of the colors in our range we are looking for
        mask = cv2.inRange(img_hsv, self.color_thr_low, self.color_thr_hi)

        # which index of the range has the highest amount of yellow?
        hist = np.sum(mask, axis=0)
        max_yellow = np.argmax(hist)

        return max_yellow, hist[max_yellow], mask


    def run(self, cam_img):
        '''
        main runloop of the CV controller
        input: cam_image, an RGB numpy array
        output: steering, throttle, and the image.
        If overlay_image is True, then the output image
        includes and overlay that shows how the 
        algorithm is working; otherwise the image
        is just passed-through untouched. 
        '''
        if cam_img is None:
            return 0, 0, False, None

        max_yellow, confidence, mask = self.get_i_color(cam_img)
        conf_thresh = 0.001

        if self.target_pixel is None:
            # Use the first run of get_i_color to set our relationship with the yellow line.
            # You could optionally init the target_pixel with the desired value.
            self.target_pixel = max_yellow
            logger.info(f"Automatically chosen line position = {self.target_pixel}")

        if self.pid_st.setpoint != self.target_pixel:
            # this is the target of our steering PID controller
            self.pid_st.setpoint = self.target_pixel

        if confidence >= self.confidence_threshold:
            # invoke the controller with the current yellow line position
            # get the new steering value as it chases the ideal
            self.steering = self.pid_st(max_yellow)

            # slow down linearly when away from ideal, and speed up when close
            if abs(max_yellow - self.target_pixel) > self.target_threshold:
                # we will be turning, so slow down
                if self.throttle > self.throttle_min:
                    self.throttle -= self.delta_th
                if self.throttle < self.throttle_min:
                    self.throttle = self.throttle_min
            else:
                # we are going straight, so speed up
                if self.throttle < self.throttle_max:
                    self.throttle += self.delta_th
                if self.throttle > self.throttle_max:
                    self.throttle = self.throttle_max
        else:
            logger.info(f"No line detected: confidence {confidence} < {self.confidence_threshold}")

        # show some diagnostics
        if self.overlay_image:
            cam_img = self.overlay_display(cam_img, mask, max_yellow, confidence)

        return self.steering, self.throttle, cam_img

    def overlay_display(self, cam_img, mask, max_yellow, confidense):

        mask_exp = np.stack((mask, ) * 3, axis=-1)
        iSlice = self.scan_y
        img = np.copy(cam_img)
        img[iSlice : iSlice + self.scan_height, :, :] = mask_exp
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        display_str = []
        display_str.append("STEERING:{:.1f}".format(self.steering))
        display_str.append("THROTTLE:{:.2f}".format(self.throttle))
        display_str.append("I YELLOW:{:d}".format(max_yellow))
        display_str.append("CONF:{:.2f}".format(confidense))

        y = 10
        x = 10

        for s in display_str:
            cv2.putText(img, s, color=(200, 200, 200), org=(x ,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4)
            y += 10

        return img        

class UnityBridge:
    def __init__(self, 
                 unity_ip='127.0.0.1', 
                 image_port=5000, 
                 command_port=5001):
        
        # Configurações de rede
        self.unity_ip = unity_ip
        self.image_port = image_port
        self.command_port = command_port
        
        # Socket para receber imagens
        self.image_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.image_socket.bind(('0.0.0.0', image_port))
        self.image_socket.settimeout(1.0)
        
        # Socket para enviar comandos
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.steering = 0.0  
        self.throttle = 0.0   
        

        # Configurar PID e LineFollower
        pid = PID(Kp=0.5, Ki=0.01, Kd=0.1, output_limits=(-1, 1))

        self.dataset_recorder = DatasetRecorder()
        self.dataset_recorder.start_recording()
        
 
        class Config:
            OVERLAY_IMAGE = True
            SCAN_Y = 60  # Posição da linha a ser detetada na imagem
            SCAN_HEIGHT = 10
    

            COLOR_THRESHOLD_LOW  = (10, 172, 192)
            COLOR_THRESHOLD_HIGH  = (30, 255, 255)

            TARGET_PIXEL = None
            TARGET_THRESHOLD = 5
            CONFIDENCE_THRESHOLD = 100
            THROTTLE_INITIAL = 0.3
            THROTTLE_STEP = 0.02
            THROTTLE_MAX = 1.0
            THROTTLE_MIN = 0.2
        
        self.line_follower = LineFollower(pid, Config())
        
        # Estado do sistema
        self.current_frame = None
        self.running = True
        self.last_command_time = time.time()
        self.command_rate_limit = 0.01
        
        # Estatísticas
        self.frames_processed = 0
        self.start_time = time.time()
        self.last_fps_update = self.start_time
        self.current_fps = 0
        
        # Iniciar thread
       
        self.receive_thread = threading.Thread(target=self.receive_images)
        self.receive_thread.daemon = True
        self.receive_thread.start()
    
 
    def show_images(self,images, cmap=None):
        cols = 2
        rows = (len(images)+1)//cols
        
        plt.figure(figsize=(10, 11))
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i+1)
            # use gray scale color map if there is only one channel
            cmap = 'gray' if len(image.shape)==2 else cmap
            plt.imshow(image, cmap=cmap)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        plt.show()

    def select_rgb_white_yellow(self,image): 
        # white color mask
        lower = np.uint8([200, 200, 200])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(image, lower, upper)
        # yellow color mask
        lower = np.uint8([190, 190,   0])
        upper = np.uint8([255, 255, 255])
        yellow_mask = cv2.inRange(image, lower, upper)
        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        masked = cv2.bitwise_and(image, image, mask = mask)
        return masked
    
    def convert_hsv(self,image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    def convert_hls(self,image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    
    def select_white_yellow(self,image):
        converted = self.convert_hls(image)
        # white color mask
        lower = np.uint8([  0, 200,   0])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(converted, lower, upper)
        # yellow color mask
        lower = np.uint8([ 10,   0, 100])
        upper = np.uint8([ 40, 255, 255])
        yellow_mask = cv2.inRange(converted, lower, upper)
        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return cv2.bitwise_and(image, image, mask = mask)    
    
    def select_orange_yellow(self,image):
        converted = self.convert_hsv(image)
        # white color mask
        lower = np.array([5, 120, 150])
        upper = np.array([25, 255, 255])
        white_mask = cv2.inRange(converted, lower, upper)
        # yellow color mask
        lower = np.uint8([ 10,   100, 100])
        upper = np.uint8([ 20, 255, 255])
        yellow_mask = cv2.inRange(converted, lower, upper)
        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return cv2.bitwise_and(image, image, mask = mask)   

    def apply_smoothing(self,image, kernel_size=15):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    def detect_edges(self,image, low_threshold=50, high_threshold=150):
        return cv2.Canny(image, low_threshold, high_threshold)

    def filter_region(self,image, vertices):
        mask = np.zeros_like(image)
        if len(mask.shape)==2:
            cv2.fillPoly(mask, vertices, 255)
        else:
            cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
        return cv2.bitwise_and(image, mask)

    def hough_lines(self,image):
        return cv2.HoughLinesP(image, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

    def draw_lines(self,image, lines, color=[255, 0, 0], thickness=2, make_copy=True):
        # the lines returned by cv2.HoughLinesP has the shape (-1, 1, 4)
        if make_copy:
            image = np.copy(image) # don't want to modify the original
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        return image

    def average_slope_intercept(self,lines):
        left_lines    = [] # (slope, intercept)
        left_weights  = [] # (length,)
        right_lines   = [] # (slope, intercept)
        right_weights = [] # (length,)
        
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2==x1:
                    continue # ignore a vertical line
                slope = (y2-y1)/(x2-x1)
                intercept = y1 - slope*x1
                length = np.sqrt((y2-y1)**2+(x2-x1)**2)
                if slope < 0: # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))
        
        # add more weight to longer lines    
        left_lane  = np.dot(left_weights,  left_lines) /np.sum(left_weights)  if len(left_weights) >0 else None
        right_lane = np.dot(right_weights, right_lines)/np.sum(right_weights) if len(right_weights)>0 else None
    
        return left_lane, right_lane # (slope, intercept), (slope, intercept)

    def make_line_points(self,y1, y2, line):

        if line is None:
            return None
        
        slope, intercept = line
        
        # make sure everything is integer as cv2.line requires it
        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)
        y1 = int(y1)
        y2 = int(y2)
        
        return ((x1, y1), (x2, y2))
    
    def lane_lines(self,image, lines):
        left_lane, right_lane = self.average_slope_intercept(lines)
        
        y1 = image.shape[0] # bottom of the image
        y2 = y1*0.6         # slightly lower than the middle

        left_line  = self.make_line_points(y1, y2, left_lane)
        right_line = self.make_line_points(y1, y2, right_lane)
        
        return left_line, right_line

        
    def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
        # make a separate image to draw lines and combine with the orignal later
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line,  color, thickness)
        # image1 * α + image2 * β + λ
        # image1 and image2 must be the same shape.
        return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

    def select_region(self,image):
        # first, define the polygon by vertices
        rows, cols = image.shape[:2]
        bottom_left  = [cols*0.1, rows*0.95]
        top_left     = [cols*0.4, rows*0.6]
        bottom_right = [cols*0.9, rows*0.95]
        top_right    = [cols*0.6, rows*0.6] 
        # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        return self.filter_region(image, vertices)


    def image_oragenge_mask(self, frame):
            if frame is None:
                return None, None
                
            height, width = frame.shape[:2]
            
            # 1. Detecção principal por cor (HSV)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Detecção da cor laranja (mais flexível com dois intervalos)
            lower_orange1 = np.array([5, 120, 150])
            upper_orange1 = np.array([25, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
            
            # Operações morfológicas para melhorar a máscara
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask1, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            return mask
    def receive_images(self):
        """Thread para receber e processar imagens da simulação Unity"""
        frame_count = 0
        frame_skip = 2
        
        while self.running:
            try:
                # Receber frame
                data, _ = self.image_socket.recvfrom(65507)
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning("Recebido frame vazio ou corrompido")
                    continue
                    
                frame_count += 1
                self.frames_processed += 1

                #steering, throttle, processed_frame = self.line_follower.run(frame)
                



                # Atualiza FPS a cada segundo
                current_time = time.time()
                if current_time - self.last_fps_update >= 0.1:
                    self.current_fps = self.frames_processed / (current_time - self.last_fps_update)
                    self.frames_processed = 0
                    self.last_fps_update = current_time
                
                    #self.send_command(throttle, steering)
                
          
            
                self.current_frame = frame
              
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # Tecla ESC para sair
                    print("Saindo...")
                    self.running = False
                    break
                elif key == ord('w'):  # Acelerar
                    self.throttle = min(self.throttle + 0.05, 1.0)
                elif key == ord('s'):  # Travar/marcha-atrás
                    self.throttle = max(self.throttle - 0.05, -1.0)
                elif key == ord('a'):  # Virar à esquerda
                    self.steering = max(self.steering - 0.1, -1.0)
                elif key == ord('d'):  # Virar à direita
                    self.steering = min(self.steering + 0.1, 1.0)
                elif key == ord('r'):  # Reset do steering
                    self.steering = 0.0
                elif key == ord('t'):  # Reset da aceleração
                    self.throttle = 0.3
                elif key == ord('g'):  # Guardar frame e comando atual
                    #save_data(frame, steering, throttle)
                    #print(f"Guardado: {csv_path}")
                    pass 

                if key!=255:
                    self.send_command(self.throttle, self.steering)
                    self.dataset_recorder.save_frame(frame, self.throttle, self.steering)
                    

                mask = self.select_region(self.apply_smoothing(self.select_white_yellow(frame)))
                list_of_lines = self.hough_lines(self.detect_edges(mask))

                line = self.draw_lines(frame, list_of_lines)
                

                cv2.imshow('Linha', frame)
                cv2.imshow('Mask', mask)
                cv2.imshow('Line', line)
    
                #cv2.imshow('Processada', processed_frame)
                

                
            except socket.timeout:
                logger.warning("Timeout ao receber imagem")
                continue
            except Exception as e:
                logger.error(f"Erro ao processar imagem: {e}")
           
    def send_command(self, speed: float, rotation: float):
        try:
            import struct
            command_bytes = struct.pack('ff', float(speed), float(rotation))
            self.command_socket.sendto(command_bytes, (self.unity_ip, self.command_port))
        except Exception as e:
            logger.error(f"Erro ao enviar comando: {e}")
    
    def close(self):
        logger.info("Encerrando comunicação com Unity")
        self.running = False
        try:

            self.send_command(0, 0)
            
            self.dataset_recorder.stop_recording()
            self.receive_thread.join(timeout=2.0)

             

            self.image_socket.close()
            self.command_socket.close()
            

            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Erro ao encerrar: {e}")


def main():
  
   

 
    bridge = UnityBridge(
        unity_ip="127.0.0.1", 
        image_port=5000, 
        command_port=5001 
    )
    
    try:
      
 
        while bridge.running:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Interrupção de teclado detectada")
    except Exception as e:
        logger.error(f"Erro não tratado: {e}")
    finally:
        bridge.close()
        logger.info("Programa encerrado.")


if __name__ == "__main__":
    main()