import cv2  # Importa la librería OpenCV
import numpy as np  # Importa la librería NumPy para manejo de matrices y arreglos

# Inicializa la captura de video desde la cámara (índice 0 para la cámara predeterminada)
cap = cv2.VideoCapture(0)

# Crea un objeto Background Subtractor MOG2 para detectar el fondo
fgbg = cv2.createBackgroundSubtractorMOG2()

# Crea un kernel de elemento estructurante para operaciones morfológicas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Ciclo principal para procesar cada fotograma del video
while True:
    # Captura un fotograma desde la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Convierte el fotograma a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Dibuja un rectángulo negro en la parte superior para mostrar el estado
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    color = (0, 255, 0)
    texto_estado = "Estado: No se ha detectado movimiento"

    # Definición de los puntos extremos del área de análisis (original)
    area_pts = np.array([[450, 380], [590, 380], [590, frame.shape[0]-55], [450, frame.shape[0]-55]])

    # Creación de una máscara de área para la detección de movimiento
    imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(gray, gray, mask=imAux)

    # Aplicación del Background Subtractor MOG2 y operaciones morfológicas
    fgmask = fgbg.apply(image_area)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    # Detección de contornos en la máscara
    cnts, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            texto_estado = "Estado 1: Alerta Movimiento Detectado!"
            color = (0, 0, 255)

    # Dibuja el área de análisis y el estado en el fotograma
    cv2.drawContours(frame, [area_pts], -1, color, 2)
    cv2.putText(frame, texto_estado, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Definición de los puntos extremos del área de análisis desplazada
    area_pts_shifted = area_pts + [280, 0]  # Desplazamiento de 400 píxeles a la derecha

    # Creación de una máscara de área para la segunda detección de movimiento
    imAux_shifted = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux_shifted = cv2.drawContours(imAux_shifted, [area_pts_shifted], -1, (255), -1)
    image_area_shifted = cv2.bitwise_and(gray, gray, mask=imAux_shifted)

    # Aplicación del Background Subtractor MOG2 y operaciones morfológicas
    fgmask_shifted = fgbg.apply(image_area_shifted)
    fgmask_shifted = cv2.morphologyEx(fgmask_shifted, cv2.MORPH_OPEN, kernel)
    fgmask_shifted = cv2.dilate(fgmask_shifted, None, iterations=2)

    # Detección de contornos en la máscara desplazada
    cnts_shifted, _ = cv2.findContours(fgmask_shifted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts_shifted:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            texto_estado = "Estado 2: Alerta Movimiento Detectado! (Desplazado)"
            color = (0, 0, 255)

    # Dibuja el área de análisis desplazada y el estado en el fotograma
    cv2.drawContours(frame, [area_pts_shifted], -1, color, 2)
    cv2.putText(frame, texto_estado, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Muestra las máscaras y el fotograma en ventanas
    cv2.imshow('fgmask', fgmask)
    cv2.imshow('fgmask_shifted', fgmask_shifted)
    cv2.imshow("frame", frame)

    # Espera por la tecla ESC para salir del ciclo
    k = cv2.waitKey(70) & 0xFF
    if k == 27:
        break

# Libera la captura de video y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
