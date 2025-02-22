import socket
import cv2
import numpy as np
import threading
import time
import logging
from dataclasses import dataclass

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('LaneFollower')

@dataclass
class PIDConfig:
    kp: float = 0.6  # Aumentado para reagir mais rápido
    ki: float = 0.02
    kd: float = 1.0  # Reduzido para diminuir oscilações

@dataclass
class CarConfig:
    base_speed: float = 0.7
    max_steering_angle: float = 1.0
    steering_smoothing: float = 0.65  # Reduzido para reações mais rápidas

def clamp(value: float, min_value: float = -1.0, max_value: float = 1.0) -> float:
    return max(min(value, max_value), min_value)

class PID:
    def __init__(self, config: PIDConfig = PIDConfig()):
        self.config = config
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.need_reset = False
        logger.info(f"PID inicializado com kp={config.kp}, ki={config.ki}, kd={config.kd}")

    def compute(self, error: float) -> float:
        if self.need_reset:
            self.previous_error = error
            self.integral = 0.0
            self.need_reset = False

        self.integral += error
        self.integral = clamp(self.integral, -1.0, 1.0)

        current_time = time.time()
        dt = max(current_time - self.last_time, 0.01)

        derivative = (error - self.previous_error) / dt

        self.previous_error = error
        self.last_time = current_time

        output = clamp(
            self.config.kp * error +
            self.config.ki * self.integral +
            self.config.kd * derivative
        )

        return output

class ImageProcessor:
    @staticmethod
    def detect_lane(frame):
        if frame is None:
            return None, None

        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([10, 120, 120])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) != 2:
            return None, mask

        M1 = cv2.moments(contours[0])
        M2 = cv2.moments(contours[1])

        if M1["m00"] > 0 and M2["m00"] > 0:
            cx1 = int(M1["m10"] / M1["m00"])
            cx2 = int(M2["m10"] / M2["m00"])
            center_x = (cx1 + cx2) // 2
            deviation = (center_x - (width // 2)) / (width // 2)
            return deviation, mask
        
        return None, mask
    
class UnityBridge:
    def __init__(self, 
                 unity_ip='127.0.0.1', 
                 image_port=5000, 
                 command_port=5001,
                 camera=None,
                 car_config=CarConfig(),
                 pid_config=PIDConfig()):
        
        # Configurações permanecem as mesmas...
        self.unity_ip = unity_ip
        self.image_port = image_port
        self.command_port = command_port
        self.image_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.image_socket.bind(('0.0.0.0', image_port))
        self.image_socket.settimeout(1.0)
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.car_config = car_config
        self.pid = PID(pid_config)
        self.current_frame = None
        self.running = True
        self.car = camera
        self.last_command_time = time.time()
        self.command_rate_limit = 0.1
        self.frames_processed = 0
        self.start_time = time.time()
        self.last_fps_update = self.start_time
        self.current_fps = 0
        
        logger.info(f"Iniciando bridge com Unity (IP: {unity_ip}, Porta de imagem: {image_port}, Porta de comando: {command_port})")
        self.receive_thread = threading.Thread(target=self.receive_images)
        self.receive_thread.daemon = True
        self.receive_thread.start()
    
    def receive_images(self):
        frame_count = 0
        frame_skip = 2  
        
        while self.running:
            try:
                data, _ = self.image_socket.recvfrom(65507)
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning("Recebido frame vazio ou corrompido")
                    continue
                    
                frame_count += 1
                self.frames_processed += 1
                
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.current_fps = self.frames_processed / (current_time - self.last_fps_update)
                    self.frames_processed = 0
                    self.last_fps_update = current_time

                deviation, mask = ImageProcessor.detect_lane(frame)
                
                if deviation is not None:
  
                    deviation = clamp(deviation)
                    
 
                    steering_correction = self.pid.compute(deviation)
                    steering_angle = clamp(steering_correction) 
                    
 
                    turn_factor = abs(steering_angle)
                    adjusted_speed = clamp(
                        self.car_config.base_speed * (1 - (turn_factor * self.car_config.steering_smoothing)),
                        0.3,  # Velocidade mínima
                        self.car_config.base_speed  # Velocidade máxima
                    )

                    #if frame_count % frame_skip == 0 or current_time - self.last_command_time >= self.command_rate_limit:
                    if current_time - self.last_command_time >= self.command_rate_limit:
                        self.send_command(adjusted_speed, steering_angle)
                        self.last_command_time = current_time

                    self._draw_visualization(frame, deviation, steering_angle, adjusted_speed)
                else:
                    self.send_command(0, 0)
                    self.pid.need_reset = True
                    cv2.putText(frame, "Linha perdida - Parando", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                self.current_frame = frame
                cv2.imshow('Base', frame)
                cv2.imshow('Mask', mask)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    self.running = False
                
            except socket.timeout:
                logger.warning("Timeout ao receber imagem")
                continue
            except Exception as e:
                logger.error(f"Erro ao processar imagem: {e}")
                continue
    
    def send_command(self, speed: float, rotation: float):
        try:
            speed = clamp(speed)
            rotation = clamp(rotation)
            
            import struct
            command_bytes = struct.pack('ff', float(speed), float(rotation))
            self.command_socket.sendto(command_bytes, (self.unity_ip, self.command_port))
        except Exception as e:
            logger.error(f"Erro ao enviar comando: {e}")

    def _draw_visualization(self, frame, deviation, steering_angle, speed):
        height, width = frame.shape[:2]
        center_x = int(width/2)
        center_y = height - 50
        
        # Linha de direção
        steer_x = int(center_x + steering_angle * width/180)
        cv2.line(frame, (center_x, center_y), (steer_x, center_y-50), (255, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Marcador de desvio
        target_x = width // 4
        cv2.circle(frame, (target_x, height - 20), 5, (255, 0, 0), -1)
        cx = int(target_x + (deviation * width // 2))
        cv2.line(frame, (cx, height), (cx, height - 50), (0, 255, 0), 2)
        
        # Informações de telemetria
        cv2.putText(frame, f"Angle: {steering_angle}",     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {speed:.1f}",          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Deviation: {deviation:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    
    def close(self):
        logger.info("Encerrando comunicação com Unity")
        self.running = False
        try:
            self.send_command(0, 0)
            self.receive_thread.join(timeout=2.0)
            self.image_socket.close()
            self.command_socket.close()
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Erro ao encerrar: {e}")


def main():
    car_config = CarConfig(base_speed=0.8, steering_smoothing=0.75)
    pid_config = PIDConfig(kp=0.35, ki=0.01, kd=1.5)

    bridge = UnityBridge(
        unity_ip='127.0.0.1', 
        image_port=5000, 
        command_port=5001,
        car_config=car_config,
        pid_config=pid_config
    )
    
    try:
        logger.info("Sistema iniciado. Pressione Ctrl+C para sair.")
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
