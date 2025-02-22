import socket
import cv2
import numpy as np
import threading
import time
import numpy as np
import time
import cv2

class PID:
    def __init__(self, kp=0.25, ki=0.006, kd=2.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        self.need_reset = False

    def compute(self, error):
        if self.need_reset:
            self.previous_error = error
            self.integral = 0.0
            self.need_reset = False
            
        if (error > 0 and self.previous_error < 0) or (error < 0 and self.previous_error > 0):
            self.integral = 0.0
            
        if -30 < self.integral < 30:
            self.integral += error
            
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.1
            
        derivative = (error - self.previous_error) / dt
        
        self.previous_error = error
        self.last_time = current_time
        
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class UnityBridge:
    def __init__(self, 
                 unity_ip='127.0.0.1', 
                 image_port=5000, 
                 command_port=5001,
                 camera=None):
        # Configuração para receber imagens
        self.image_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.image_socket.bind(('0.0.0.0', image_port))
        
        # Configuração para enviar comandos
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_ip = unity_ip
        self.command_port = command_port
        
        # Configurações de controle
        self.pid = PID()
        self.BASE_SPEED = 1
        self.DIFF = 0.6
        
        # Última imagem recebida
        self.current_frame = None
        self.running = True
        

        self.car = camera
        
        # Iniciar thread para receber imagens
        self.receive_thread = threading.Thread(target=self.receive_images)
        self.receive_thread.start()
    
    def process_image_center(self, frame):
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([10, 150, 150])
        upper_orange = np.array([25, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        roi_height = int(height * 0.3)
        roi = mask[height-roi_height:height, :]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, mask
        
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            return None, mask
        
        cx = int(M["m10"] / M["m00"])
        center_deviation = (cx - (width / 2)) / (width / 2)
        
        return center_deviation, mask
    
    def receive_images(self):
        frame_count = 0
        frame_skip = 2 
        
        while self.running:
            try:
                # Receber frame
                data, _ = self.image_socket.recvfrom(65507)
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                frame_count += 1
   
                
                if frame is not None:
                    # Processamento de imagem e controle
                    deviation, mask = self.process_image_center(frame)
                    
                    if deviation is not None:
                        # Calcular ângulo de direção usando PID
                        steering_correction = self.pid.compute(deviation)
                        steering_angle = int(steering_correction * self.DIFF)
                        
                        # Ajustar velocidade baseado na severidade da curva
                        turn_factor = abs(steering_angle) / self.DIFF
                        adjusted_speed = self.BASE_SPEED * (1 - (turn_factor * 0.75))
                        
           
                        
                        if frame_count % frame_skip != 0:
                            self.send_command(adjusted_speed, steering_angle)
                           
                        # Visualização
                        height, width = frame.shape[:2]
               
                        center_x = int(width/2)
                        center_y = height - 50
                        steer_x = int(center_x + steering_angle * width/180)
                        
                        cv2.line(frame, (center_x, center_y), (steer_x, center_y-50), (0, 255, 0), 2)
                        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        target_x = width // 4
                        cv2.circle(frame, (target_x, height - 20), 5, (255, 0, 0), -1)
                        cx = int(target_x + (deviation * width // 2))
                        cv2.line(frame, (cx, height), (cx, height - 50), (0, 255, 0), 2)
                        
                        cv2.putText(frame, f"Angle: {steering_angle}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Speed: {adjusted_speed:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        # Parar quando nenhuma linha é detectada
                        self.send_command(0, 0)
                        self.pid.need_reset = True
                        cv2.putText(frame, "Line Lost - Stopping", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Mostrar frames
                    cv2.imshow('Line Following', frame)
                    cv2.imshow('Mask', mask)
                    
                    # Verificar tecla ESC para sair
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        self.running = False
                
            except Exception as e:
                print(f"Erro ao processar imagem: {e}")
    
    def send_command(self, speed, rotation):
        """Envia comandos de velocidade e rotação para Unity como bytes"""
        try:
            #print(" Speed: ",speed," Steering: " , rotation)
            import struct
            command_bytes = struct.pack('ff', float(speed), float(rotation))
            self.command_socket.sendto(command_bytes, (self.unity_ip, self.command_port))
        except Exception as e:
            print(f"Erro ao enviar comando: {e}")
    
    def close(self):
        self.running = False
        self.receive_thread.join()
        self.image_socket.close()
        self.command_socket.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    
    bridge = UnityBridge()
    
    try:
        while bridge.running:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        print("Programa encerrado.")
