import socket
import os
import cv2
import numpy as np
import threading
import time
from tensorflow.keras.models import load_model
import logging
from dataclasses import dataclass

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('AutonomousDriver')

@dataclass
class CarConfig:
    base_speed: float = 1.0
    max_steering_angle: float = 25.0
    steering_smoothing: float = 0.75
    min_speed: float = 0.5

class PilotNetController:
    def __init__(self, model_path="pilotnet_unity.keras"):
        logger.info(f"Carregando modelo do PilotNet de {model_path}")
        self.model = load_model(model_path)
        self.last_prediction = 0
        self.smoothing_factor = 0.6  #suavização para as predições

    def process_image(self, frame):
        """Processa a imagem para o formato esperado pelo modelo"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (200, 66))  # Tamanho esperado pelo PilotNet
        img = img / 255.0  # Normalização
        return np.expand_dims(img, axis=0)

    def predict(self, frame):
        """Faz a predição do ângulo de direção"""
        processed_image = self.process_image(frame)
        prediction = self.model.predict(processed_image, verbose=0)[0][0]
        
        # Suavização da predição
        smoothed_prediction = (prediction * self.smoothing_factor +  self.last_prediction * (1 - self.smoothing_factor))
        self.last_prediction = smoothed_prediction
        
        return smoothed_prediction

class AutonomousDriver:
    def __init__(self, 
                 unity_ip='127.0.0.1', 
                 image_port=5000, 
                 command_port=5001,
                 car_config=CarConfig()):
        
        # Configurações de rede
        self.unity_ip = unity_ip
        self.image_port = image_port
        self.command_port = command_port
        
        # Sockets
        self.image_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.image_socket.bind(('0.0.0.0', image_port))
        self.image_socket.settimeout(1.0)
        
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Configurações do carro
        self.car_config = car_config
        
        # Controlador neural
        self.controller = PilotNetController()
        
        # Estado do sistema
        self.running = True
        self.autonomous_mode = True
        self.current_speed = car_config.base_speed
        self.current_steering = 0
        
        # Métricas
        self.frames_processed = 0
        self.start_time = time.time()
        self.last_fps_update = self.start_time
        self.current_fps = 0
        
        # Interface
        #cv2.namedWindow('Autonomous Driver', cv2.WINDOW_NORMAL)
        
        # Iniciar thread de processamento
        self.receive_thread = threading.Thread(target=self.process_loop)
        self.receive_thread.daemon = True
        self.receive_thread.start()
        
        logger.info("Sistema de direção autônoma iniciado")

    def process_loop(self):
        """Loop principal de processamento"""
        while self.running:
            try:
                # Receber frame
                data, _ = self.image_socket.recvfrom(65507)
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                
                # Atualizar FPS
                self.frames_processed += 1
                current_time = time.time()
                #if current_time - self.last_fps_update >= 0.1:
                self.current_fps = self.frames_processed
                self.frames_processed = 0
                self.last_fps_update = current_time
            
            
                # Obter predição do modelo
                steering_correction = self.controller.predict(frame)
                self.current_steering = steering_correction * self.car_config.max_steering_angle
                
                # Ajustar velocidade baseado na curva
                turn_factor = abs(steering_correction)
                self.current_speed = max(
                    self.car_config.min_speed,
                    self.car_config.base_speed * (1 - (turn_factor * self.car_config.steering_smoothing))
                )
                
                # Enviar comandos
                self.send_command(self.current_speed, self.current_steering)
                
                # Visualização
                self.draw_interface(frame)
                
                # Processar teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break

                
            except socket.timeout:
                logger.warning("Timeout ao receber imagem")
                continue
            except Exception as e:
                logger.error(f"Erro ao processar imagem: {e}")
                continue
    
    def draw_interface(self, frame):
        """Desenha a interface do usuário"""
        # Informações básicas
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {self.current_speed:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Direction: {self.current_steering:.1f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        
        # Visualizar
        cv2.imshow('Driver', frame)
    
    def send_command(self, speed, steering):
        """Envia comandos para o Unity"""
        try:
            import struct
            command_bytes = struct.pack('ff', float(speed), float(steering))
            self.command_socket.sendto(command_bytes, (self.unity_ip, self.command_port))
        except Exception as e:
            logger.error(f"Erro ao enviar comando: {e}")
    
    def close(self):
        """Encerra o sistema"""
        logger.info("Encerrando sistema...")
        self.running = False
        self.send_command(0, 0)  # Parar o carro
        time.sleep(0.5)
        self.image_socket.close()
        self.command_socket.close()
        cv2.destroyAllWindows()

def main():
  
    car_config = CarConfig(
        base_speed=0.9,
        max_steering_angle=15.0,
        steering_smoothing=0.55,
        min_speed=0.3
    )
    
    # Iniciar sistema
    driver = AutonomousDriver(
        unity_ip="127.0.0.1",
        car_config=car_config
    )
    
    try:
    
        while driver.running:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nEncerrando...")
    finally:
        driver.close()

if __name__ == "__main__":
    main()