import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
import cv2
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configurações
DATASET_DIR = "dataset_unity"
IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005

def load_session_data(session_dir):
    """
    Carrega dados de uma sessão específica do dataset
    """
    images = []
    steering_angles = []
    
    # Carregar arquivo de labels
    labels_df = pd.read_csv(os.path.join(session_dir, "labels.csv"))
    
    # Carregar cada imagem e seu ângulo correspondente
    for _, row in labels_df.iterrows():
        # Converter o número do frame para inteiro
        frame_num = int(row['frame'])
        frame_path = os.path.join(session_dir, f"frame_{frame_num:06d}.jpg")
        
        if os.path.exists(frame_path):
            try:
                # Carregar e preprocessar imagem
                img = cv2.imread(frame_path)
                if img is None:
                    print(f"Erro ao carregar imagem: {frame_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                
                images.append(img)
                steering_angles.append(row['steering_angle'])
            except Exception as e:
                print(f"Erro ao processar {frame_path}: {e}")
    
    if not images:
        print(f"Nenhuma imagem carregada do diretório: {session_dir}")
        return np.array([]), np.array([])
    
    return np.array(images), np.array(steering_angles)

def load_complete_dataset():
    """
    Carrega todas as sessões do dataset
    """
    all_images = []
    all_angles = []
    
    # Percorrer todas as sessões
    for session_dir in os.listdir(DATASET_DIR):
        session_path = os.path.join(DATASET_DIR, session_dir)
        if os.path.isdir(session_path):
            print(f"Carregando sessão: {session_dir}")
            images, angles = load_session_data(session_path)
            
            if len(images) > 0:
                all_images.extend(images)
                all_angles.extend(angles)
    
    if not all_images:
        raise ValueError("Nenhuma imagem foi carregada do dataset!")
    
    return np.array(all_images), np.array(all_angles)

def augment_data(images, angles):
    """
    Aplica técnicas de data augmentation para melhorar o dataset
    """
    augmented_images = []
    augmented_angles = []
    
    for image, angle in zip(images, angles):
        augmented_images.append(image)
        augmented_angles.append(angle)
        
        # Espelhar horizontalmente
        flipped_image = cv2.flip(image, 1)
        flipped_angle = -angle
        augmented_images.append(flipped_image)
        augmented_angles.append(flipped_angle)
        
        # Adicionar ruído aleatório
        noisy_image = image + np.random.normal(0, 10, image.shape)
        noisy_image = np.clip(noisy_image, 0, 255)
        augmented_images.append(noisy_image)
        augmented_angles.append(angle)
    
    return np.array(augmented_images), np.array(augmented_angles)

def create_pilotnet_model():
    """
    Cria o modelo PilotNet
    """
    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), activation="relu", 
               input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
        Conv2D(36, (5, 5), strides=(2, 2), activation="relu"),
        Conv2D(48, (5, 5), strides=(2, 2), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        Flatten(),
        Dense(100, activation="relu"),
        Dense(50, activation="relu"),
        Dense(10, activation="relu"),
        Dense(1)
    ])
    
    return model

def plot_training_history(history):
    """
    Plota o histórico de treinamento
    """
    plt.figure(figsize=(12, 4))
    
    # Plot do loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Salvar o gráfico
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Carregar dataset
    print("Carregando dataset...")
    X, y = load_complete_dataset()
    
    print(f"Dataset carregado: {len(X)} imagens")
    
    # Normalizar imagens
    X = X.astype(np.float32) / 255.0
    
    # Data augmentation
    print("Aplicando data augmentation...")
    X_aug, y_aug = augment_data(X, y)
    print(f"Dataset após augmentation: {len(X_aug)} imagens")
    
    # Dividir em treino e validação
    print("Dividindo dataset...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_aug, test_size=0.2, random_state=42
    )
    
    # Criar e compilar modelo
    print("Criando modelo...")
    model = create_pilotnet_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'pilotnet_unity_checkpoint.keras',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Treinar modelo
    print("Iniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Plotar histórico
    plot_training_history(history)
    
    # Salvar modelo final
    model.save("pilotnet_unity.keras")
    print("Treinamento concluído! Modelo salvo como 'pilotnet_unity.keras'")

if __name__ == "__main__":
    main()