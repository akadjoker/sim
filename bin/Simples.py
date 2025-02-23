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

def detectar_faixas(imagem):
    """
    Detecta faixas brancas e amarelas em uma imagem de estrada
    """
    # Converter para HSV para melhor detecção de cores
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    
    # Definir ranges para cor amarela
    amarelo_baixo = np.array([20, 100, 100])
    amarelo_alto = np.array([30, 255, 255])
    
    # Máscara para cor amarela
    mascara_amarela = cv2.inRange(hsv, amarelo_baixo, amarelo_alto)
    
    # Definir range para cor branca
    branco_baixo = np.array([0, 0, 200])
    branco_alto = np.array([180, 30, 255])
    
    # Máscara para cor branca
    mascara_branca = cv2.inRange(hsv, branco_baixo, branco_alto)
    
    # Combinar as máscaras
    mascara_combinada = cv2.bitwise_or(mascara_branca, mascara_amarela)
    
    # Aplicar um blur para reduzir ruído
    blur = cv2.GaussianBlur(mascara_combinada, (5, 5), 0)
    
    # Detectar bordas usando Canny
    bordas = cv2.Canny(blur, 50, 150)
    
    # Definir região de interesse (ROI)
    altura, largura = bordas.shape
    mascara = np.zeros_like(bordas)
    
    # Definir os pontos do polígono para ROI
    poligono = np.array([[
        (0, altura),
        (largura//2 - 50, altura//2),
        (largura//2 + 50, altura//2),
        (largura, altura)
    ]], np.int32)
    
    cv2.fillPoly(mascara, poligono, 255)
    roi = cv2.bitwise_and(bordas, mascara)
    
    # Detectar linhas usando transformada de Hough
    linhas = cv2.HoughLinesP(roi, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    
    # Desenhar as linhas detectadas
    imagem_final = imagem.copy()
    if linhas is not None:
        for linha in linhas:
            x1, y1, x2, y2 = linha[0]
            cv2.line(imagem_final, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return imagem_final, mascara_combinada, bordas

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
                    

                

                #cv2.imshow('Linha', frame)
                resultado, mascara, bordas = detectar_faixas(frame)
        
                # Mostrar resultados
                cv2.imshow('Detecção de Faixas', resultado)
                cv2.imshow('Máscara', mascara)
                cv2.imshow('Bordas', bordas)
            
    
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