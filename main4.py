from PIL import ImageGrab
import cv2
import numpy as np
import time

# Função para detectar objetos de cores específicas em um frame
def detectar_cores(frame, lower, upper):
    # Converte o frame para o espaço de cores BGR se não estiver no formato correto
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    # Converte o frame para o espaço de cores HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define a faixa de cores a ser detectada
    mask = cv2.inRange(hsv, lower, upper)
    
    # Encontra contornos na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Inicializa uma lista para armazenar as localizações dos objetos
    object_locations = []

    # Itera sobre os contornos encontrados
    for contour in contours:
        # Calcula o centro do contorno
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            object_locations.append((cX, cY))
            
            # Desenha um círculo vermelho ao redor do objeto
            cv2.circle(frame, (cX, cY), 10, (0, 0, 255), -1)  # (0, 0, 255) é a cor vermelha

    return object_locations

# Inicializa o contador de FPS
fps_start_time = time.time()
fps_counter = 0

while True:
    # Captura a tela do monitor principal
    frame = np.array(ImageGrab.grab(bbox=(0, 0, 1920, 1080)))  # Ajuste as dimensões conforme necessário
    
    # Define as faixas de cores para a detecção (nesse caso, é azul)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])

    # Chama a função de detecção de cores
    locations = detectar_cores(frame, lower_blue, upper_blue)

    # Exibe as localizações dos objetos
    for loc in locations:
        print(f"Objeto encontrado em x={loc[0]}, y={loc[1]}")

    # Exibe o frame com as áreas detectadas
    cv2.imshow('Detecção de Cores', frame)

    # Calcula o FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1:
        print(f"FPS: {fps_counter}")
        fps_counter = 0
        fps_start_time = time.time()

    # Encerra o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fecha as janelas
cv2.destroyAllWindows()
