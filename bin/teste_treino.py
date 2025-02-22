import tensorflow as tf
import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob

def load_test_data(dataset_dir, num_samples=5):
    """
    Carrega imagens aleatórias e seus ângulos correspondentes do dataset
    """
    # Lista para armazenar dados de teste
    test_data = []
    
    # Encontrar todas as sessões
    sessions = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    for session in sessions:
        session_path = os.path.join(dataset_dir, session)
        labels_path = os.path.join(session_path, "labels.csv")
        
        if os.path.exists(labels_path):
            # Carregar labels
            labels_df = pd.read_csv(labels_path)
            
            # Selecionar algumas linhas aleatórias
            sample_rows = labels_df.sample(n=min(num_samples, len(labels_df)))
            
            for _, row in sample_rows.iterrows():
                frame_num = int(row['frame'])
                frame_path = os.path.join(session_path, f"frame_{frame_num:06d}.jpg")
                
                if os.path.exists(frame_path):
                    test_data.append({
                        'image_path': frame_path,
                        'steering_angle': row['steering_angle']
                    })
    
    # Embaralhar dados
    random.shuffle(test_data)
    return test_data[:num_samples]

def test_model(model_path="pilotnet_unity.keras", dataset_dir="dataset_unity", num_tests=5):
    """
    Testa o modelo em imagens aleatórias e visualiza os resultados
    """
    # Carregar modelo
    print("Carregando modelo...")
    model = tf.keras.models.load_model(model_path)
    
    # Carregar dados de teste
    print("Carregando dados de teste...")
    test_data = load_test_data(dataset_dir, num_tests)
    
    if not test_data:
        print("Erro: Nenhuma imagem de teste encontrada!")
        return
    
    # Configurar visualização
    fig, axes = plt.subplots(2, num_tests, figsize=(20, 8))
    
    for i, data in enumerate(test_data):
        # Carregar e processar imagem
        img = cv2.imread(data['image_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionar para visualização
        img_display = img.copy()
        
        # Processar para predição
        img_model = cv2.resize(img, (200, 66)) / 255.0
        
        # Fazer predição
        predicted_angle = model.predict(np.expand_dims(img_model, axis=0), verbose=0)[0][0]
        real_angle = data['steering_angle']
        
        # Visualizar imagem original
        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"Original\nReal: {real_angle:.1f}°\nPrevisto: {predicted_angle:.1f}°")
        axes[0, i].axis("off")
        
        # Visualizar imagem processada (como o modelo vê)
        axes[1, i].imshow(img_model)
        axes[1, i].set_title("Processada (entrada do modelo)")
        axes[1, i].axis("off")
        
        # Calcular erro
        error = abs(predicted_angle - real_angle)
        print(f"\nImagem {i+1}:")
        print(f"Ângulo real: {real_angle:.1f}°")
        print(f"Ângulo previsto: {predicted_angle:.1f}°")
        print(f"Erro absoluto: {error:.1f}°")
    
    plt.tight_layout()
    plt.savefig('test_results.png')
    plt.show()
    
    return test_data

def main():
    # Configurações
    MODEL_PATH = "pilotnet_unity.keras"
    DATASET_DIR = "dataset_unity"
    NUM_TESTS = 5
    
    # Verificar se o modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")
        return
    
    # Verificar se o dataset existe
    if not os.path.exists(DATASET_DIR):
        print(f"Erro: Dataset não encontrado em {DATASET_DIR}")
        return
    
    try:
        test_data = test_model(MODEL_PATH, DATASET_DIR, NUM_TESTS)
        
        if test_data:
            print("\nTeste concluído com sucesso!")
            print(f"Resultados salvos em 'test_results.png'")
    except Exception as e:
        print(f"Erro durante o teste: {e}")

if __name__ == "__main__":
    main()