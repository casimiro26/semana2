import cv2
import numpy as np
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Crear carpeta para guardar capturas
carpeta_capturas = "capturas"
os.makedirs(carpeta_capturas, exist_ok=True)

# Emociones simuladas
emotions = ['Enojado', 'Asco', 'Miedo', 'Feliz', 'Triste', 'Sorprendido', 'Neutral']

# Colores únicos para cada emoción
colores_emociones = {
    'Enojado': '#FF0000',      # Rojo
    'Asco': '#8B008B',         # Morado oscuro
    'Miedo': '#800000',        # Marrón oscuro
    'Feliz': '#FFD700',        # Amarillo
    'Triste': '#1E90FF',       # Azul
    'Sorprendido': '#FFA500',  # Naranja
    'Neutral': '#808080'       # Gris
}

# Cargar clasificador de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    print("Error al cargar Haar Cascade.")
    exit()

# Abrir cámara
camara = cv2.VideoCapture(0)
if not camara.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

ultima_emocion = None
conteo_emociones = {e: 0 for e in emotions}

# Configurar gráfico en ventana propia
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
colores = [colores_emociones[e] for e in emotions]
barras = ax.bar(conteo_emociones.keys(), conteo_emociones.values(), color=colores)
ax.set_ylim(0, 10)
ax.set_ylabel('Cantidad de veces detectada')
ax.set_title('Conteo de emociones detectadas')
plt.tight_layout()
fig.canvas.manager.set_window_title("Gráfico de Emociones")

frame_counter = 0

while True:
    ret, frame = camara.read()
    if not ret:
        print("No se pudo leer el frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        # Simular emoción
        emotion = random.choice(emotions)

        # Solo capturar si cambia la emoción y no es Neutral
        if emotion != ultima_emocion and emotion != 'Neutral':
            nombre_archivo = f"{carpeta_capturas}/rostro_{emotion}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            rostro_color = frame[y:y+h, x:x+w]
            cv2.imwrite(nombre_archivo, rostro_color)
            print(f"Captura guardada: {nombre_archivo}")
            ultima_emocion = emotion

        # Aumentar conteo
        conteo_emociones[emotion] += 1

        # Mostrar emoción detectada
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar ventana de cámara
    cv2.imshow('Detector de Emociones - Presiona "q" para salir', frame)

    # Actualizar gráfico cada 10 frames
    frame_counter += 1
    if frame_counter % 10 == 0:
        for barra, valor in zip(barras, conteo_emociones.values()):
            barra.set_height(valor)
        ax.set_ylim(0, max(10, max(conteo_emociones.values()) + 2))
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
