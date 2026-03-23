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
DB_PATH        = "turnike.db"
THRESHOLD      = 0.45   # Benzerlik eşiği (0.4–0.6 arası ideal)
DETECT_PERIOD  = 0.3    # saniye: her karede algılama yapmak yerine throttle
RELOAD_PERIOD  = 5.0    # saniye: yeni kayıt varsa otomatik yeniden yükle

# ─────────────────────────────────────────────
#  FONT ÖNBELLEĞİ
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
def load_known_faces() -> list[dict]:
    """Veritabanındaki tüm kullanıcıları belleğe yükler."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT name, embedding FROM users").fetchall()
        users = [
            {"name": r[0], "emb": np.frombuffer(r[1], dtype=np.float32)}
            for r in rows
        ]
        logger.info(f"✅ {len(users)} kullanıcı yüklendi.")
        return users
    except sqlite3.Error as e:
        logger.error(f"❌ Veritabanı hatası: {e}")
        return []

def get_user_count() -> int:
    """Veritabanındaki mevcut kullanıcı sayısını döner."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    except sqlite3.Error:
        return 0

# ─────────────────────────────────────────────
#  TANIMA YARDIMCISI
# ─────────────────────────────────────────────
def recognize(embedding: np.ndarray, known_users: list[dict]) -> tuple[str, float]:
    """
    Verilen embedding'e en yakın kullanıcıyı bulur.
    Döner: (isim, benzerlik_skoru)
    """
    best_name = "Bilinmeyen"
    best_sim  = 0.0          # BUG FIX: -1 yerine 0.0 → negatif skor asla eşik geçmez

    for user in known_users:
        sim = float(np.dot(embedding, user["emb"]))
        if sim > best_sim:
            best_sim = sim
            best_name = user["name"] if sim >= THRESHOLD else "Bilinmeyen"

    return best_name, best_sim

# ─────────────────────────────────────────────
#  ANA FONKSİYON
# ─────────────────────────────────────────────
def main() -> None:
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"

    # Model yükle
    face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(320, 320))

    # Kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.critical("❌ Kamera açılamadı. Program sonlandırılıyor.")
        return

    known_users   = load_known_faces()
    known_count   = len(known_users)

    detected_faces: list = []
    last_detect:   float = 0.0
    last_reload:   float = time.time()

    logger.info("🚀 Tanıma modu başlatıldı. Çıkış: [Q]")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠ Kamera çerçevesi alınamadı.")
                continue

            now = time.time()
            h, w = frame.shape[:2]

            # ── Periyodik yüz algılama (throttle) ─────────────────
            if now - last_detect >= DETECT_PERIOD:
                detected_faces = face_app.get(frame)
                last_detect = now

            # ── Periyodik DB yenileme: yeni kayıt eklenmişse güncelle
            if now - last_reload >= RELOAD_PERIOD:
                current_count = get_user_count()
                if current_count != known_count:
                    known_users = load_known_faces()
                    known_count = current_count
                    logger.info("🔄 Kullanıcı listesi güncellendi.")
                last_reload = now

            # ── Her yüz için tanıma ve görselleştirme ─────────────
            for face in detected_faces:
                x1 = max(0, int(face.bbox[0]))
                y1 = max(0, int(face.bbox[1]))
                x2 = min(w, int(face.bbox[2]))
                y2 = min(h, int(face.bbox[3]))

                if x2 <= x1 or y2 <= y1:   # geçersiz kutu
                    continue

                name, sim = recognize(face.normed_embedding, known_users)
                color     = (0, 255, 0) if name != "Bilinmeyen" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                frame = put_text(
                    frame,
                    f"{name} ({sim:.2f})",
                    (x1, max(0, y1 - 28)),
                    size=20,
                    color=color,
                )

            # ── FPS / kullanıcı sayısı bilgi satırı ───────────────
            frame = put_text(
                frame,
                f"Kayıtlı: {known_count} kişi",
                (8, 8),
                size=18,
                color=(200, 200, 200),
            )

            cv2.imshow("Turnike - Tanıma Modu", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("👋 Program sonlandırıldı.")

if __name__ == "__main__":
    main()