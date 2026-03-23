import cv2
import sqlite3
import numpy as np
import os
from insightface.app import FaceAnalysis

# --- VERİTABANI GARANTİLEME ---
def db_check():
    conn = sqlite3.connect('turnike.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            student_id TEXT UNIQUE,
            embedding BLOB NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("✅ Veritabanı ve tablo kontrol edildi, hazır.")

db_check() # Script başlar başlamaz çalıştır

# --- MODEL VE KAMERA HAZIRLIĞI ---
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'
app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(160, 160))

cap = cv2.VideoCapture(0)

user_name = input("Ad Soyad: ")
user_id = input("Numara: ")

while True:
    ret, frame = cap.read()
    if not ret: continue

    faces = app.get(frame)
    if len(faces) > 0:
        face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
        cv2.putText(frame, "Kaydetmek icin 'S' basin", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('Kayit Ekrani', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s') and len(faces) > 0:
        embedding = face.normed_embedding.tobytes()
        conn = sqlite3.connect('turnike.db')
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (name, student_id, embedding) VALUES (?, ?, ?)", 
                           (user_name, user_id, embedding))
            conn.commit()
            print(f"✅ {user_name} başarıyla eklendi!")
            break
        except Exception as e:
            print(f"❌ Kayıt hatası: {e}")
        finally:
            conn.close()

cap.release()
cv2.destroyAllWindows()
