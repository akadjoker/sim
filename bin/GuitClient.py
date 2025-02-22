import sys
import zmq
import struct
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QSlider, QLabel
from PyQt5.QtCore import Qt

class ZeroMQQtClient(QWidget):
    def __init__(self):
        super().__init__()

        # Configurar ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:4455")  

        # Configurar interface
        self.setWindowTitle("Controle do Carro (ZeroMQ)")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        # Slider de Speed
        self.label_speed = QLabel("Speed: 0.0")
        layout.addWidget(self.label_speed)

        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setMinimum(-100)
        self.slider_speed.setMaximum(100)
        self.slider_speed.setValue(0)
        self.slider_speed.valueChanged.connect(self.update_speed_label)
        self.slider_speed.sliderReleased.connect(self.send_command)  # Enviar comando ao soltar o slider
        layout.addWidget(self.slider_speed)

        # Slider de Steering
        self.label_steering = QLabel("Steering: 0.0")
        layout.addWidget(self.label_steering)

        self.slider_steering = QSlider(Qt.Horizontal)
        self.slider_steering.setMinimum(-100)
        self.slider_steering.setMaximum(100)
        self.slider_steering.setValue(0)
        self.slider_steering.valueChanged.connect(self.update_steering_label)
        self.slider_steering.sliderReleased.connect(self.send_command)  # Enviar comando ao soltar o slider
        layout.addWidget(self.slider_steering)

        # Botão para reset
        self.button_reset = QPushButton("Reset")
        self.button_reset.clicked.connect(self.reset_values)
        layout.addWidget(self.button_reset)

        self.setLayout(layout)

    def update_speed_label(self):
        value = self.slider_speed.value() #/ 100.0  # Normalizar para -1 a 1
        self.label_speed.setText(f"Speed: {value:}")

    def update_steering_label(self):
        value = self.slider_steering.value() #/ 100.0  # Normalizar para -1 a 1
        self.label_steering.setText(f"Steering: {value:}")

    def send_command(self):
        speed = self.slider_speed.value() #3/ 100.0  # Normalizar para -1 a 1
        steering = self.slider_steering.value() #/ 100.0  # Normalizar para -1 a 1

        data = struct.pack("i", speed)
        self.socket.send(data)
        response = self.socket.recv()  # Espera resposta do Unity
        print(f"Enviado Speed: {speed}, Steering: {steering} | Resposta: {response.decode()}")

    def reset_values(self):
        self.slider_speed.setValue(0)
        self.slider_steering.setValue(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ZeroMQQtClient()
    window.show()
    sys.exit(app.exec_())

