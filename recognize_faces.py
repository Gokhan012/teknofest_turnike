import cv2
import sqlite3
import numpy as np
import os
from insightface.app import FaceAnalysis

# 1. Hazırlık ve Modeller
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320))

# 2. Veritabanındaki Kullanıcıları Belleğe Yükle
def load_known_faces():
    known_faces = []
    try:
        conn = sqlite3.connect('turnike.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name, embedding FROM users")
        rows = cursor.fetchall()
        for row in rows:
            name = row[0]
            # BLOB verisini tekrar numpy dizisine çevir
            embedding = np.frombuffer(row[1], dtype=np.float32)
            known_faces.append({"name": name, "embedding": embedding})
        conn.close()
    except Exception as e:
        print(f"Veritabanı hatası: {e}")
    return known_faces

known_users = load_known_faces()
THRESHOLD = 0.45  # Benzerlik eşiği (0.4 - 0.6 arası idealdir)

# 3. Kamera Döngüsü
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    faces = app.get(frame)
    
    for face in faces:
        bbox = face.bbox.astype(int)
        current_embedding = face.normed_embedding
        
        best_match = "Bilinmeyen"
        max_similarity = -1

        # Her bir kayıtlı kullanıcı ile karşılaştır
        for user in known_users:
            # Kosinüs Benzerliği (İki vektör de normize olduğu için dot product yeterli)
            similarity = np.dot(current_embedding, user["embedding"])
            
            if similarity > max_similarity:
                max_similarity = similarity
                if similarity > THRESHOLD:
                    best_match = user["name"]

        # Ekrana Yazdır
        color = (0, 255, 0) if best_match != "Bilinmeyen" else (0, 0, 255)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        label = f"{best_match} ({max_similarity:.2f})"
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Teknofest Turnike - Tanima Modu', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
