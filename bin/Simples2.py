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

import cv2
import numpy as np

def detectar_faixas_e_preencher(imagem):
    """
    Detecta a faixa amarela (esquerda) e branca (direita) e preenche a área entre elas
    """
    # Converter para HSV
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    
    # Definir ranges para cor amarela (faixa esquerda)
    amarelo_baixo = np.array([20, 100, 100])
    amarelo_alto = np.array([30, 255, 255])
    mascara_amarela = cv2.inRange(hsv, amarelo_baixo, amarelo_alto)
    
    # Definir ranges para cor branca (faixa direita)
    branco_baixo = np.array([0, 0, 200])
    branco_alto = np.array([180, 30, 255])
    mascara_branca = cv2.inRange(hsv, branco_baixo, branco_alto)
    
    # Aplicar blur e detectar bordas para cada cor separadamente
    blur_amarelo = cv2.GaussianBlur(mascara_amarela, (5, 5), 0)
    blur_branco = cv2.GaussianBlur(mascara_branca, (5, 5), 0)
    
    bordas_amarelo = cv2.Canny(blur_amarelo, 50, 150)
    bordas_branco = cv2.Canny(blur_branco, 50, 150)
    
    # Definir ROI (região de interesse)
    altura, largura = bordas_amarelo.shape
    mascara_roi = np.zeros_like(bordas_amarelo)
    
    # Ajustar pontos do polígono para focar nas laterais
    poligono = np.array([[
        (0, altura),
        (largura//2 - 100, altura//2),
        (largura//2 + 100, altura//2),
        (largura, altura)
    ]], np.int32)
    
    cv2.fillPoly(mascara_roi, poligono, 255)
    
    # Aplicar ROI às bordas
    roi_amarelo = cv2.bitwise_and(bordas_amarelo, mascara_roi)
    roi_branco = cv2.bitwise_and(bordas_branco, mascara_roi)
    
    # Detectar linhas para cada cor
    linhas_amarelas = cv2.HoughLinesP(roi_amarelo, 1, np.pi/180, 50,
                                     minLineLength=100, maxLineGap=50)
    linhas_brancas = cv2.HoughLinesP(roi_branco, 1, np.pi/180, 50,
                                    minLineLength=100, maxLineGap=50)
    
    # Criar imagem para o resultado
    imagem_final = imagem.copy()
    
    # Pontos para criar o polígono entre as faixas
    pontos_pista = []
    
    # Processar linha amarela (esquerda)
    if linhas_amarelas is not None:
        for linha in linhas_amarelas:
            x1, y1, x2, y2 = linha[0]
            cv2.line(imagem_final, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # Guardar pontos para o polígono
            pontos_pista.extend([(x1, y1), (x2, y2)])
    
    # Processar linha branca (direita)
    if linhas_brancas is not None:
        for linha in linhas_brancas:
            x1, y1, x2, y2 = linha[0]
            cv2.line(imagem_final, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # Guardar pontos para o polígono
            pontos_pista.extend([(x1, y1), (x2, y2)])
    
    # Preencher área entre as faixas se detectou ambas as linhas
    if pontos_pista and len(pontos_pista) >= 4:
        # Organizar pontos para formar um polígono fechado
        pontos_pista = np.array(pontos_pista, np.int32)
        hull = cv2.convexHull(pontos_pista)
        # Preencher a área com uma cor semi-transparente
        overlay = imagem_final.copy()
        cv2.fillPoly(overlay, [hull], (0, 255, 0))  # Verde
        cv2.addWeighted(overlay, 0.3, imagem_final, 0.7, 0, imagem_final)
    
    return imagem_final



def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
def select_rgb_white_yellow(image): 
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

def select_white_yellow(image):
    converted = convert_hls(image)
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
def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_smoothing(image, kernel_size=15):
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

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

                #if key!=255:
                
                self.send_command(self.throttle, self.steering)
              

                

                cv2.imshow('Linha', frame)
                resultado = detect_edges(apply_smoothing(convert_gray_scale(select_white_yellow(frame))))
        
                # Mostrar resultados
                cv2.imshow('Mask', resultado)
             
            
    
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