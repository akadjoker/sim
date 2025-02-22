import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Carregar modelo
model = load_model("pilotnet_unity.keras")

def test_model():
    img = cv2.imread("/home/djoker/sim/bin/dataset_unity/session_20250221_172823/frame_000048.jpg")  # Usa uma imagem da pista real
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 66))  # Ajusta para o formato do modelo
    img = img / 255.0  # Normaliza
    img = np.expand_dims(img, axis=0)  # Adiciona dimensão do batch
    
    prediction = model.predict(img, verbose=0)
    print(f"Steering Angle previsto: {prediction[0][0]}")

test_model()
