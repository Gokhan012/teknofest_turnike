import cv2
import numpy as np
import os
import time
from insightface.app import FaceAnalysis

# 1. Ayarlar ve Hazırlık
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
last_analysis_time = 0
analysis_interval = 0.5  # 0.5 saniyede bir analiz yap
faces = [] # Analiz yapılmayan karelerde kutuların kaybolmaması için

# 'buffalo_s' hem SCRFD (hizalama) hem ArcFace (embedding) içerir
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

# 2. Kamera Bağlantısı
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Teknofest Turnike AI Modülü Hazır...")
print("SCRFD + ArcFace (Buffalo_S) aktif.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    
    # 3. Akıllı Analiz Döngüsü
    # Sadece belirlenen aralıklarla AI modelini çalıştır (CPU'yu korur)
    if current_time - last_analysis_time > analysis_interval:
        faces = app.get(frame)
        last_analysis_time = current_time

    # 4. Görselleştirme (Her karede çizim yapılır, böylece görüntü akıcı kalır)
    for face in faces:
        # SCRFD Bounding Box
        bbox = face.bbox.astype(int)
        
        # ArcFace 512-d Embedding (Bu veriyi SQLite'a kaydedeceğiz)
        embedding = face.normed_embedding 

        # Çizimler
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        score = f"Score: {face.det_score:.2f}"
        cv2.putText(frame, score, (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 5. Ekranda Göster
    cv2.imshow('Teknofest Proje - SCRFD & ArcFace', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
