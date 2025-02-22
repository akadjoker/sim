import socket
import os
import cv2
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

@dataclass
class CarConfig:
    base_speed: float = 1.0
    max_steering_angle: float = 1.0
    steering_smoothing: float = 0.75

# ==================== ALTERNATIVA 1: CONTROLADOR FUZZY ====================
class FuzzyController:
    """
    Implementação de um controlador Fuzzy simplificado para seguimento de linha.
    Utiliza lógica Fuzzy para determinar o ângulo de direção baseado no erro.
    """
    def __init__(self):
        self.last_output = 0.0
        logger.info("Controlador Fuzzy inicializado")
        
    def compute(self, error: float) -> float:
        """
        Calcula a saída do controlador Fuzzy baseado no erro atual
        Args:
            error: Desvio normalizado do centro da faixa (-1.0 a 1.0)
        Returns:
            Fator de correção para o ângulo de direção
        """
        # Fuzzificação - Categorizando o erro em conjuntos difusos
        error_abs = abs(error)
        
        # Regras de pertinência (membership)
        small_error = max(0, 1 - error_abs * 5)      # Erro pequeno (0 a 0.2)
        medium_error = max(0, min(error_abs * 5 - 1, 3 - error_abs * 5))  # Erro médio (0.2 a 0.6)
        large_error = max(0, error_abs * 5 - 3)      # Erro grande (0.6 a 1.0)
        
        # Base de regras e inferência
        # - Se erro é pequeno: correção pequena
        # - Se erro é médio: correção média
        # - Se erro é grande: correção grande
        small_correction = 0.2
        medium_correction = 0.6
        large_correction = 1.0
        
        # Agregação e defuzzificação (centro de gravidade)
        numerator = (small_error * small_correction + 
                     medium_error * medium_correction + 
                     large_error * large_correction)
        denominator = small_error + medium_error + large_error
        
        if denominator == 0:
            output = 0
        else:
            output = numerator / denominator
            
        # Aplica sinal do erro e suavização
        output = output * (1.0 if error >= 0 else -1.0)
        output = 0.7 * output + 0.3 * self.last_output  # Suavização
        
        self.last_output = output
        return output

# ==================== ALTERNATIVA 2: CONTROLADOR PREDITIVO ====================
class PredictiveController:
    """
    Controlador Preditivo que estima a trajetória futura da faixa
    e aplica correções antecipadas baseadas em previsão.
    """
    def __init__(self, history_size=10, prediction_horizon=5):
        self.error_history = deque(maxlen=history_size)
        self.history_size = history_size
        self.prediction_horizon = prediction_horizon
        self.last_output = 0.0
        logger.info(f"Controlador Preditivo inicializado (histórico={history_size}, horizonte={prediction_horizon})")
        
    def compute(self, error: float) -> float:
        """
        Calcula a correção de direção com base na previsão de trajetória
        Args:
            error: Desvio normalizado do centro da faixa (-1.0 a 1.0)
        Returns:
            Fator de correção para o ângulo de direção
        """
        # Armazena o erro atual no histórico
        self.error_history.append(error)
        
        # Se não temos histórico suficiente, retorna resposta proporcional
        if len(self.error_history) < 3:
            return error * 0.5
            
        # Calcula a derivada da tendência (taxa de mudança do erro)
        errors = list(self.error_history)
        error_trend = np.polyfit(range(len(errors)), errors, 1)[0]
        
        # Prevê o erro futuro baseado na tendência atual
        predicted_error = error + error_trend * self.prediction_horizon
        
        # Calcula a saída combinando o erro atual e a previsão
        current_weight = 0.6
        prediction_weight = 0.4
        
        output = current_weight * error + prediction_weight * predicted_error
        
        # Aplica suavização para evitar mudanças bruscas
        smoothed_output = 0.7 * output + 0.3 * self.last_output
        self.last_output = smoothed_output
        
        return smoothed_output
        
# ==================== ALTERNATIVA 3: CONTROLADOR BASEADO EM REDES NEURAIS ====================
class NeuralController:
    """
    Controlador baseado em uma pequena rede neural para aprendizado de comportamento.
    Implementação simplificada que simula o comportamento de uma rede treinada.
    """
    def __init__(self):
        # Em uma implementação real, carregaríamos pesos pré-treinados
        # Aqui simulamos o comportamento de uma rede treinada com uma função de resposta
        self.errors = deque(maxlen=5)
        logger.info("Controlador Neural inicializado")
        
    def compute(self, error: float) -> float:
        """
        Calcula a correção de direção usando um modelo neural simplificado
        Args:
            error: Desvio normalizado do centro da faixa (-1.0 a 1.0)
        Returns:
            Fator de correção para o ângulo de direção
        """
        # Armazena histórico de erros para análise temporal
        self.errors.append(error)
        
        # Função de ativação sigmoide para não-linearidade
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        # Em um cenário real, teríamos camadas e pesos da rede
        # Aqui simulamos um comportamento aprendido com base na experiência
        
        error_abs = abs(error)
        error_sign = np.sign(error)
        
        # Características temporais (diferença entre erros consecutivos)
        error_diff = 0
        if len(self.errors) >= 2:
            error_diff = self.errors[-1] - self.errors[-2]
        
        # Camada 1: Feature extraction (simulada)
        f1 = sigmoid(3 * error)
        f2 = sigmoid(2 * error_diff)
        f3 = error_abs * error_abs  # Resposta quadrática
        
        # Camada 2: Combinação (simulada)
        output_linear = 0.5 * f1 + 0.3 * f2 + 0.2 * f3
        
        # Resposta não-linear
        output = error_sign * (1 - np.exp(-3 * error_abs))
        
        # Ajuste adicional baseado na tendência da faixa
        if len(self.errors) >= 3:
            trend = np.mean([self.errors[-1] - self.errors[-2], 
                            self.errors[-2] - self.errors[-3]])
            output += 0.2 * trend
        
        return output

# ==================== ALTERNATIVA 4: CONTROLADOR HÍBRIDO ====================
class HybridController:
    """
    Controlador híbrido que combina diferentes estratégias de controle
    dependendo da situação de direção.
    """
    def __init__(self):
        # Componentes de controladores individuais
        self.predictive = PredictiveController(history_size=8, prediction_horizon=3)
        self.fuzzy = FuzzyController()
        
        # Parâmetros PID para situações específicas
        self.pid_params = {
            'straight': {'kp': 0.2, 'ki': 0.005, 'kd': 1.0},
            'curve': {'kp': 0.3, 'ki': 0.001, 'kd': 2.5}
        }
        
        # Estado do controlador
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
        self.last_output = 0.0
        self.error_history = deque(maxlen=20)
        
        logger.info("Controlador Híbrido inicializado")
        
    def compute(self, error: float) -> float:
        """
        Seleciona e combina diferentes estratégias de controle baseado na situação atual
        Args:
            error: Desvio normalizado do centro da faixa (-1.0 a 1.0)
        Returns:
            Fator de correção para o ângulo de direção
        """
        # Atualiza histórico de erros
        self.error_history.append(error)
        
        # Determina se estamos em curva ou reta baseado no histórico de erros
        is_curve = False
        if len(self.error_history) >= 5:
            errors = list(self.error_history)[-5:]
            variance = np.var(errors)
            is_curve = variance > 0.05 or abs(error) > 0.3
            
        # Seleciona parâmetros PID baseado na situação
        pid_config = self.pid_params['curve'] if is_curve else self.pid_params['straight']
        
        # Cálculo do PID básico
        dt = time.time() - self.last_time
        if dt <= 0: dt = 0.01
        
        # Componente proporcional
        p_term = pid_config['kp'] * error
        
        # Componente integral (com anti-windup)
        if abs(self.integral) < 30:
            self.integral += error * dt
        i_term = pid_config['ki'] * self.integral
        
        # Componente derivativa
        d_term = pid_config['kd'] * (error - self.last_error) / dt
        
        # Saída PID
        pid_output = p_term + i_term + d_term
        
        # Combina PID com outros controladores
        if is_curve:
            # Em curvas, damos mais peso ao PID e controlador preditivo
            predictive_output = self.predictive.compute(error)
            output = 0.6 * pid_output + 0.4 * predictive_output
        else:
            # Em retas, mais peso ao controle fuzzy para suavidade
            fuzzy_output = self.fuzzy.compute(error)
            output = 0.4 * pid_output + 0.6 * fuzzy_output
        
        # Suavização da resposta
        smoothed_output = 0.7 * output + 0.3 * self.last_output
        
        # Atualiza estado para próxima iteração
        self.last_error = error
        self.last_time = time.time()
        self.last_output = smoothed_output
        
        return smoothed_output

# ==================== DETECTOR DE FAIXA APRIMORADO ====================
class AdvancedLaneDetector:
    """
    Detector de faixa mais avançado que utiliza múltiplas técnicas para
    detecção mais robusta da linha.
    """
    def __init__(self):
        # Parâmetros de detecção
        self.roi_height_ratio = 0.4  # Proporção da altura para ROI
        self.last_lane_position = None
        self.confidence = 0.0
        self.history = deque(maxlen=5)
        
    def detect_lane(self, frame):
        """
        Detecta a posição da faixa usando múltiplas técnicas
        Args:
            frame: Imagem da câmera
        Returns:
            (desvio_normalizado, máscara_processada) ou (None, mask) se não detectado
        """
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
        
        # 2. Região de interesse (ROI) dinâmica baseada em histórico
        roi_height = int(height * self.roi_height_ratio)
        roi_y_start = height - roi_height
        
        # Ajuste da ROI baseado no último ponto detectado (se disponível)
        if self.last_lane_position is not None and self.confidence > 0.5:
            last_x = int(width/2 + self.last_lane_position * width/2)
            # Centraliza a ROI lateralmente no último ponto conhecido
            roi_width = int(width * 1.0)  # 70% da largura
            roi_x_start = max(0, last_x - roi_width//2)
            roi_x_end = min(width, roi_x_start + roi_width)
            
            # ROI adaptativa
            roi = mask[roi_y_start:height, roi_x_start:roi_x_end]
            roi_offset_x = roi_x_start  # Para ajustar coordenadas depois
        else:
            # ROI completa se não temos histórico confiável
            roi = mask[roi_y_start:height, :]
            roi_offset_x = 0
            
        # Encontrar contornos na ROI
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Tenta uma abordagem alternativa com gradientes se método principal falhar
            edges = cv2.Canny(frame[roi_y_start:height, :], 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=40, maxLineGap=25)
            
            if lines is not None:
                # Calcula centro médio das linhas
                x_positions = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    x_center = (x1 + x2) / 2
                    x_positions.append(x_center)
                
                if x_positions:
                    mean_x = np.mean(x_positions)
                    center_deviation = (mean_x - (width / 2)) / (width / 2)
                    self.confidence = 0.3  # Confiança baixa para detecção por Hough
                    self.last_lane_position = center_deviation
                    self.history.append(center_deviation)
                    return center_deviation, edges
            
            # Se ainda falhar, usa histórico se disponível
            if self.history:
                avg_position = np.mean(list(self.history))
                self.confidence *= 0.8  # Reduz confiança progressivamente
                return avg_position, mask
                
            self.confidence = 0
            return None, mask
        
        # Encontra o contorno mais relevante (maior ou mais próximo do centro)
        if len(contours) > 1:
            # Prioriza contornos maiores e mais próximos do centro
            def contour_score(cnt):
                area = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    return 0
                cx = int(M["m10"] / M["m00"]) + roi_offset_x
                center_distance = abs(cx - width/2)
                # Fórmula que balanceia área e proximidade ao centro
                return area - (center_distance * 0.5)
                
            contours = sorted(contours, key=contour_score, reverse=True)
            
        largest_contour = contours[0]
        M = cv2.moments(largest_contour)
        
        if M["m00"] == 0:
            self.confidence *= 0.8
            if self.history:
                return np.mean(list(self.history)), mask
            return None, mask
        
        # Calcula o centro do contorno
        cx = int(M["m10"] / M["m00"]) + roi_offset_x
        center_deviation = (cx - (width / 2)) / (width / 2)
        
        # Avalia qualidade da detecção
        contour_area = cv2.contourArea(largest_contour)
        min_expected_area = 100  # Área mínima esperada
        max_expected_area = width * height * 0.1  # 10% da imagem
        
        if min_expected_area <= contour_area <= max_expected_area:
            self.confidence = min(1.0, self.confidence + 0.2)
        else:
            self.confidence = max(0.0, self.confidence - 0.1)
        
        # Armazena posição para usos futuros
        self.last_lane_position = center_deviation
        self.history.append(center_deviation)
        
        # Visualização: desenha o contorno na máscara
        vis_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_mask[roi_y_start:height, :], 
                          [np.array(largest_contour) + np.array([roi_offset_x, 0])], 
                          -1, (0, 255, 0), 2)
        
        return center_deviation, vis_mask

# ==================== CLASSE BRIDGE UNITY MODIFICADA ====================
class UnityBridge:
    def __init__(self, 
                 unity_ip='127.0.0.1', 
                 image_port=5000, 
                 command_port=5001,
                 controller_type='hybrid',
                 car_config=CarConfig()):
        
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
        
        # Configurações do carro
        self.car_config = car_config
        
        # Seleção do controlador
        self.controller_type = controller_type
        self.controller = self._create_controller(controller_type)
        
        # Detector de faixa avançado
        self.lane_detector = AdvancedLaneDetector()
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
        logger.info(f"Iniciando bridge com Unity usando controlador {controller_type}")
        self.receive_thread = threading.Thread(target=self.receive_images)
        self.receive_thread.daemon = True
        self.receive_thread.start()
    
    def _create_controller(self, controller_type):
        """Cria o controlador baseado no tipo especificado"""
        controllers = {
            'fuzzy': FuzzyController(),
            'predictive': PredictiveController(),
            'neural': NeuralController(),
            'hybrid': HybridController()
        }
        
        if controller_type not in controllers:
            logger.warning(f"Tipo de controlador '{controller_type}' não reconhecido. Usando híbrido.")
            return controllers['hybrid']
            
        return controllers[controller_type]
    
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
                
                # Atualiza FPS a cada segundo
                current_time = time.time()
                if current_time - self.last_fps_update >= 0.1:
                    self.current_fps = self.frames_processed / (current_time - self.last_fps_update)
                    self.frames_processed = 0
                    self.last_fps_update = current_time
                
                # Detecção de faixa usando detector avançado
                deviation, mask = self.lane_detector.detect_lane(frame)
                
                if deviation is not None:
                    # Calcular ângulo de direção usando o controlador selecionado
                    steering_correction = self.controller.compute(deviation)
                    steering_angle = int(steering_correction * self.car_config.max_steering_angle)
                    
                    # Ajustar velocidade baseado na severidade da curva
                    turn_factor = abs(steering_angle) / self.car_config.max_steering_angle
                    adjusted_speed = self.car_config.base_speed * (1 - (turn_factor * self.car_config.steering_smoothing))
                    
                    
                    # Enviar comandos com controle de taxa
                    #if frame_count % frame_skip == 0 or current_time - self.last_command_time >= self.command_rate_limit:
                    if current_time - self.last_command_time >= self.command_rate_limit:
                        self.send_command(adjusted_speed, steering_angle)
                        self.dataset_recorder.save_frame(frame, adjusted_speed, steering_angle)
                        self.last_command_time = current_time
                       
                    # Visualização
                    self._draw_visualization(frame, deviation, steering_angle, adjusted_speed)
                else:
                    self.send_command(0, 0)
                    cv2.putText(frame, "Linha perdida - Parando", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Mostrar frames
                self.current_frame = frame
                cv2.imshow('Linha', frame)
                cv2.imshow('Mask', mask)
                
                # Verificar tecla ESC para sair
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    self.running = False
                
            except socket.timeout:
                logger.warning("Timeout ao receber imagem")
                continue
            except Exception as e:
                logger.error(f"Erro ao processar imagem: {e}")
                continue
    
    def _draw_visualization(self, frame, deviation, steering_angle, speed):
        """Desenha elementos visuais no frame para visualização"""
        height, width = frame.shape[:2]
        center_x = int(width/2)
        center_y = height - 50
        

        steer_x = int(center_x + steering_angle * width/180)
        cv2.line(frame, (center_x, center_y), (steer_x, center_y-50), (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        

        target_x = width // 4
        cv2.circle(frame, (target_x, height - 20), 5, (255, 0, 0), -1)
        cx = int(target_x + (deviation * width // 2))
        cv2.line(frame, (cx, height), (cx, height - 50), (0, 255, 0), 2)
        

        cv2.putText(frame, f"Controlador: {self.controller_type}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(frame, f"Angle: {steering_angle}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Speed: {speed:.1f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidance: {self.lane_detector.confidence:.2f}", (10, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
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
            

            self.receive_thread.join(timeout=2.0)

            if self.dataset_recorder.recording:
                self.dataset_recorder.stop_recording()
            

            self.image_socket.close()
            self.command_socket.close()
            

            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Erro ao encerrar: {e}")


def main():
    """Função principal do programa"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Seguimento de linha com diferentes controladores')
    parser.add_argument('--controller', type=str, default='hybrid',
                      choices=['fuzzy', 'predictive', 'neural', 'hybrid'],
                      help='Tipo de controlador a ser usado (default: hybrid)')
    parser.add_argument('--ip', type=str, default='127.0.0.1',
                      help='Endereço IP do Unity (default: 127.0.0.1)')
    parser.add_argument('--speed', type=float, default=1.5,
                      help='Velocidade base do carro (default: 1.5)')
    
    args = parser.parse_args()
    
    # Configurações do carro
    car_config = CarConfig(
        base_speed=args.speed,
        max_steering_angle=15.0,
        steering_smoothing=0.7
    )

    # Inicializa a bridge com o controlador escolhido
    bridge = UnityBridge(
        unity_ip=args.ip, 
        image_port=5000, 
        command_port=5001,
        controller_type=args.controller,
        car_config=car_config
    )
    
    try:
        logger.info(f"Sistema iniciado com controlador {args.controller}. Pressione Ctrl+C para sair.")
        # Loop principal
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