import cv2
import numpy as np

# Função para detectar objetos de cores específicas em uma imagem
def detectar_cores(frame, lower, upper):
    # Converte o quadro para o espaço de cores HSV
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

    return object_locations

# Inicializa a captura de vídeo do monitor principal (0 para o monitor principal)
cap = cv2.VideoCapture(0)

while True:
    # Captura um quadro do vídeo
    ret, frame = cap.read()

    if not ret:
        break

    # Define as faixas de cores para a detecção (nesse caso, é azul)
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])

    # Chama a função de detecção de cores
    locations = detectar_cores(frame, lower_blue, upper_blue)

    # Exibe as localizações dos objetos
    for loc in locations:
        print(f"Objeto encontrado em x={loc[0]}, y={loc[1]}")

    # Exibe o quadro com as áreas detectadas
    cv2.imshow('Detecção de Cores', frame)

    # Encerra o loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()
