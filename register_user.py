import cv2
import sqlite3
import numpy as np
import os
import time
import logging
from insightface.app import FaceAnalysis
from PIL import ImageFont, ImageDraw, Image

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────
DB_PATH             = "turnike.db"
DETECT_PERIOD       = 0.3          

# ─────────────────────────────────────────────
#  FONT YARDIMCISI
# ─────────────────────────────────────────────
_font_cache: dict = {}

def _get_font(size: int):
    if size not in _font_cache:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                _font_cache[size] = ImageFont.truetype(path, size)
                return _font_cache[size]
        _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def put_text(img: np.ndarray, text: str, pos: tuple,
             size: int = 20, color: tuple = (255, 255, 255)) -> np.ndarray:
    """BGR görüntüye Türkçe destekli metin yazar."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(pos, text, font=_get_font(size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ─────────────────────────────────────────────
#  VERİTABANI
# ─────────────────────────────────────────────
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT NOT NULL,
                student_id TEXT UNIQUE,
                embedding  BLOB NOT NULL
            )
        """)
        conn.commit()
    logger.info("✅ Veritabanı hazır.")

def student_id_exists(student_id: str) -> bool:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT 1 FROM users WHERE student_id = ?", (student_id,)
        ).fetchone()
    return row is not None

def save_user(name: str, student_id: str, embedding: bytes) -> bool:
    """Kullanıcıyı kaydeder. Başarılıysa True döner."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO users (name, student_id, embedding) VALUES (?, ?, ?)",
                (name, student_id, embedding),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        logger.error("❌ Bu numara zaten kayıtlı.")
        return False
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası: {e}")
        return False

# ─────────────────────────────────────────────
#  KULLANICI GİRİŞİ
# ─────────────────────────────────────────────
def get_user_input() -> tuple[str, str]:
    """Boş bırakılamayan ve çakışmayan giriş alır."""
    while True:
        name = input("Ad Soyad : ").strip()
        if not name:
            print("⚠  Ad Soyad boş bırakılamaz.")
            continue

        student_id = input("Numara   : ").strip()
        if not student_id:
            print("⚠  Numara boş bırakılamaz.")
            continue

        if student_id_exists(student_id):
            print(f"⚠  '{student_id}' numarası zaten kayıtlı. Farklı bir numara girin.")
            continue

        return name, student_id

# ─────────────────────────────────────────────
#  EN BÜYÜK YÜZÜ SEÇ
# ─────────────────────────────────────────────
def largest_face(faces: list):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

# ─────────────────────────────────────────────
#  ANA FONKSİYON
# ─────────────────────────────────────────────
def main() -> None:
    init_db()

    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(160, 160))

    print("\n─── Kullanıcı Bilgileri ───")
    user_name, user_id = get_user_input()
    print(f"\n👤 Kayıt edilecek: {user_name} | {user_id}")
    print("📷 Kamera açılıyor... Yüzünüzü çerçeveye alın ve [S] tuşuna basın. Çıkış: [Q]\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.critical("❌ Kamera açılamadı. Program sonlandırılıyor.")
        return

    detected_faces: list = []
    last_detect: float   = 0.0
    saved: bool          = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠ Kamera çerçevesi alınamadı.")
                continue

            now = time.time()
            h, w = frame.shape[:2]

            if now - last_detect >= DETECT_PERIOD:
                detected_faces = face_app.get(frame)
                last_detect = now

            display = frame.copy()

            if detected_faces:
                face = largest_face(detected_faces)
                x1, y1, x2, y2 = (
                    max(0, int(face.bbox[0])),
                    max(0, int(face.bbox[1])),
                    min(w, int(face.bbox[2])),
                    min(h, int(face.bbox[3])),
                )
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 255), 2)
                display = put_text(display, f"{user_name}", (x1, max(0, y1 - 30)), 20, (0, 200, 255))
                display = put_text(display, "Kaydetmek için [S] basın", (10, 10), 20, (0, 255, 0))
            else:
                display = put_text(display, "Yüz algılanamadı...", (10, 10), 20, (0, 0, 255))

            cv2.imshow("Kayıt Ekranı", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                if not detected_faces:
                    logger.warning("⚠ Yüz algılanmadı, önce kameraya bakın.")
                else:
                    face      = largest_face(detected_faces)
                    embedding = face.normed_embedding.astype(np.float32).tobytes()
                    if save_user(user_name, user_id, embedding):
                        logger.info(f"✅ '{user_name}' başarıyla kaydedildi!")
                        saved = True
                        confirm = frame.copy()
                        confirm = put_text(confirm, f"✅ {user_name} kaydedildi!", (10, 10), 24, (0, 255, 0))
                        cv2.imshow("Kayıt Ekranı", confirm)
                        cv2.waitKey(1500)
                        break

            elif key == ord("q"):
                logger.info("⛔ Kullanıcı kayıt iptal etti.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not saved:
        print("ℹ️  Kayıt tamamlanmadı.")

if __name__ == "__main__":
    main()
