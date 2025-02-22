import socket
import cv2
import numpy as np
import threading
import time
import numpy as np
import time
import cv2
from WhiteController import TrackController

class Car:
    def __init__(self):
        self.speed_history = [] 
        self.steering_history = []  
        self.max_history = 3  
        self.smoothing_threshold = 0.01  
        self.speed=0
        self.steering=0

    def smooth_value(self, history, new_value):
        """Adiciona um novo valor ao histórico e suaviza oscilações."""
        if history and abs(history[-1] - new_value) < self.smoothing_threshold:
            return history[-1]  # Mantém o último valor se a mudança for pequena
        
        history.append(new_value)
        if len(history) > self.max_history:
            history.pop(0)  # Remove o valor mais antigo
        return sum(history) / len(history)  # Retorna a média dos últimos valores

    def set_steer(self, steer):
        smoothed_steer = self.smooth_value(self.steering_history, steer*100)
        self.steering = smoothed_steer

    def set_speed(self, speed):
        smoothed_speed = self.smooth_value(self.speed_history, speed)
        self.speed = smoothed_speed
       



KP = 0.3
KI = 0.005
KD = 1.5

def update_KP(val):
    global KP
    KP = val / 1000.0  # Converte para float entre 0.0 e 1.0
    print(f"KP atualizado: {KP}")

def updateKI(val):
    global KI
    KI = val / 1000.0   
    print(f"KI atualizado: {KI}")

def updateKD(val):
    global KD
    KD = val / 100.0 
    print(f"KD atualizado: {KD}")






class UnityBridge:
    def __init__(self, 
                 unity_ip='127.0.0.1', 
                 image_port=5000, 
                 command_port=5001):
        # Configuração para receber imagens
        self.image_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.image_socket.bind(('0.0.0.0', image_port))
        
        # Configuração para enviar comandos
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.unity_ip = unity_ip
        self.command_port = command_port
        
        self.car = Car()
        self.controller = TrackController(self.car)
        
        # Última imagem recebida
        self.current_frame = None
        self.running = True
        
        
 
        self.receive_thread = threading.Thread(target=self.process)
        self.receive_thread.start()
    

    
    def process(self):
        frame_count = 0
        frame_skip = 8
        global KI,KD,KP
        cv2.namedWindow("Controle")
        cv2.createTrackbar("KP", "Controle", int(KP * 1000.0), 1000, update_KP)  # 0 a 100
        cv2.createTrackbar("KI", "Controle", int(KI * 1000.0), 1000, updateKI)  
        cv2.createTrackbar("KD", "Controle", int(KD * 1000.0), 1000, updateKI)  


        
        while self.running:
            try:
                data, _ = self.image_socket.recvfrom(65507)
                nparr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                #self.controller.KI = KI
                # self.controller.KD = KD
                # self.controller.KP = KP
                
                
                frame_count += 1
                self.current_frame = frame
                self.controller.update(frame)
                debugFrame = self.controller.visualization(frame)
                cv2.imshow('Controle', frame)
                cv2.imshow('Debug', debugFrame)
                # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # lower = np.array([15, 150, 150])#laranja
                # upper = np.array([25, 255, 255])


      
                # yellow_mask = cv2.inRange(hsv, lower, upper)
                # cv2.imshow('Yellow Mask', yellow_mask)
               

                if frame_count % frame_skip != 0:
                    #continue
                    self.send_command(self.car.speed,self.car.steering)
                
                
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
    while bridge.current_frame is None and bridge.running:
        time.sleep(0.1)
    try:
        while bridge.running:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        print("Programa encerrado.")
