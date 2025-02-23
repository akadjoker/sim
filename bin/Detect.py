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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
import math
import cv2


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

def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    # reshape lines to a 2d matrix
    lines = lines.reshape(lines.shape[0], lines.shape[2])
    # create array of slopes
    slopes = (lines[:,3] - lines[:,1]) /(lines[:,2] - lines[:,0])
    # remove junk from lists
    lines = lines[~np.isnan(lines) & ~np.isinf(lines)]
    slopes = slopes[~np.isnan(slopes) & ~np.isinf(slopes)]
    # convert lines into list of points
    lines.shape = (lines.shape[0]//2,2)

    # Right lane
    # move all points with negative slopes into right "lane"
    right_slopes = slopes[slopes < 0]
    right_lines = np.array(list(filter(lambda x: x[0] > (img.shape[1]/2), lines)))
    max_right_x, max_right_y = right_lines.max(axis=0)
    min_right_x, min_right_y = right_lines.min(axis=0)

    # Left lane
    # all positive  slopes go into left "lane"
    left_slopes = slopes[slopes > 0]
    left_lines = np.array(list(filter(lambda x: x[0] < (img.shape[1]/2), lines)))
    max_left_x, max_left_y = left_lines.max(axis=0)
    min_left_x, min_left_y = left_lines.min(axis=0)

    # Curve fitting approach
    # calculate polynomial fit for the points in right lane
    right_curve = np.poly1d(np.polyfit(right_lines[:,1], right_lines[:,0], 2))
    left_curve  = np.poly1d(np.polyfit(left_lines[:,1], left_lines[:,0], 2))

    # shared ceiling on the horizon for both lines
    min_y = min(min_left_y, min_right_y)

    # use new curve function f(y) to calculate x values
    max_right_x = int(right_curve(img.shape[0]))
    min_right_x = int(right_curve(min_right_y))

    min_left_x = int(left_curve(img.shape[0]))

    r1 = (min_right_x, min_y)
    r2 = (max_right_x, img.shape[0])
    print('Right points r1 and r2,', r1, r2)
    cv2.line(img, r1, r2, color, thickness)

    l1 = (max_left_x, min_y)
    l2 = (min_left_x, img.shape[0])
    print('Left points l1 and l2,', l1, l2)
    cv2.line(img, l1, l2, color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Takes in a single frame or an image and returns a marked image
def mark_lanes(image):
    if image is None: raise ValueError("no image given to mark_lanes")
    # grayscale the image to make finding gradients clearer
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges_img = cv2.Canny(np.uint8(blur_gray), low_threshold, high_threshold)


    imshape = image.shape
    vertices = np.array([[(0, imshape[0]),
                          (450, 320),
                          (490, 320),
                          (imshape[1], imshape[0]) ]],
                          dtype=np.int32)

    masked_edges = region_of_interest(edges_img, vertices )


    # Define the Hough transform parameters
    rho             = 2           # distance resolution in pixels of the Hough grid
    theta           = np.pi/180   # angular resolution in radians of the Hough grid
    threshold       = 15       # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20       # minimum number of pixels making up a line
    max_line_gap    = 20       # maximum gap in pixels between connectable line segments

    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # Draw the lines on the edge image
    # initial_img * α + img * β + λ
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lines_edges

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
        

     

        self.dataset_recorder = DatasetRecorder()
        self.dataset_recorder.start_recording()
        
 

        
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
                    
                edges = mark_lanes(frame)
                

                cv2.imshow('Linha', frame)

                cv2.imshow('Edges', edges)
    
    
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