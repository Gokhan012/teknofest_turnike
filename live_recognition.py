
import cv2
import sqlite3
import numpy as np
import time
import os
import logging
from datetime import datetime
from insightface.app import FaceAnalysis
from PIL import ImageFont, ImageDraw, Image
from scipy.spatial import distance as dist

# ─────────────────────────────────────────────
#  LOGGING AYARLARI
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("turnike.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────
DB_PATH             = "turnike.db"
THRESHOLD           = 0.45       # Yüz tanıma benzerlik eşiği
SESSION_TIMEOUT     = 5.0        # Bu süre geçince tekrar log atılabilir (saniye)
FACE_ANALYZE_PERIOD = 0.3        # Kaç saniyede bir InsightFace çalıştırılsın

# Liveness
EAR_THRESHOLD      = 0.21
EAR_CONSEC_FRAMES  = 2
TEXTURE_THRESHOLD  = 75.0
BLINK_WINDOW       = 50          # Frame sayısı
BLINK_REQUIRED     = 1

# MediaPipe göz landmark indeksleri
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# ─────────────────────────────────────────────
#  FONT ÖNBELLEĞİ  (BUG FIX: disk I/O azaltır)
# ─────────────────────────────────────────────
_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

def _get_font(size: int):
    if size not in _font_cache:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
        font = None
        for path in candidates:
            if os.path.exists(path):
                font = ImageFont.truetype(path, size)
                break
        _font_cache[size] = font or ImageFont.load_default()
    return _font_cache[size]


def put_turkish_text(img: np.ndarray, text: str, position: tuple,
                     font_size: int = 20, color: tuple = (0, 255, 0)) -> np.ndarray:
    """BGR görüntüye Türkçe karakter destekli metin yazar."""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=_get_font(font_size), fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
#  VERİTABANI
# ─────────────────────────────────────────────
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY,
                name       TEXT NOT NULL,
                student_id TEXT UNIQUE,
                embedding  BLOB NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id        INTEGER PRIMARY KEY,
                user_name TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()


def log_to_db(name: str) -> None:
    try:
        now_str = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO logs (user_name, timestamp) VALUES (?, ?)",
                (name, now_str)
            )
            conn.commit()
        logger.info(f"📢 LOG KAYDEDİLDİ → {name} | {now_str}")
    except sqlite3.Error as e:
        logger.error(f"❌ LOG HATASI: {e}")


def load_known_faces() -> list[dict]:
    """Veritabanındaki tüm kullanıcıları yükler."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT name, embedding FROM users").fetchall()
        return [
            {"name": r[0], "emb": np.frombuffer(r[1], dtype=np.float32)}
            for r in rows
        ]
    except sqlite3.Error as e:
        logger.error(f"❌ Kullanıcı yükleme hatası: {e}")
        return []


# ─────────────────────────────────────────────
#  ANALİZ YARDIMCILARI
# ─────────────────────────────────────────────
def eye_aspect_ratio(landmarks, eye_indices: list[int], w: int, h: int) -> float:
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def compute_texture(gray_roi: np.ndarray) -> float:
    """Laplacian varyansı ile doku skorunu hesaplar (sahte fotoğraf filtresi)."""
    if gray_roi.size == 0:
        return 0.0
    roi = cv2.resize(gray_roi, (64, 64))
    return float(min(cv2.Laplacian(roi, cv2.CV_64F).var(), 150.0))


# ─────────────────────────────────────────────
#  CANLILIK TAKİPÇİSİ
# ─────────────────────────────────────────────
class LivenessTracker:
    """Tek bir yüz için göz kırpma + doku tabanlı canlılık takibi."""

    def __init__(self):
        self.consec_below: int = 0
        self.blink_frames: list[int] = []
        self.frame_idx: int = 0
        self.texture_scores: list[float] = []
        self.is_live: bool = False
        self.reason: str = "Bekleniyor..."

    def update_ear(self, ear: float) -> None:
        self.frame_idx += 1
        if ear < EAR_THRESHOLD:
            self.consec_below += 1
        else:
            if self.consec_below >= EAR_CONSEC_FRAMES:
                self.blink_frames.append(self.frame_idx)
            self.consec_below = 0
        # Eski çerçeveleri temizle
        cutoff = self.frame_idx - BLINK_WINDOW
        self.blink_frames = [f for f in self.blink_frames if f > cutoff]

    def update_texture(self, score: float) -> None:
        self.texture_scores.append(score)
        if len(self.texture_scores) > 15:
            self.texture_scores.pop(0)

    def decide(self) -> bool:
        avg_texture = float(np.mean(self.texture_scores)) if self.texture_scores else 0.0
        texture_ok = avg_texture >= TEXTURE_THRESHOLD
        blink_ok   = len(self.blink_frames) >= BLINK_REQUIRED
        self.is_live = texture_ok and blink_ok

        if not texture_ok:
            self.reason = f"⚠ SAHTE (Doku:{avg_texture:.1f})"
        elif not blink_ok:
            self.reason = f"👁 KIRPIN ({len(self.blink_frames)}/{BLINK_REQUIRED})"
        else:
            self.reason = "✅ CANLI"
        return self.is_live


# ─────────────────────────────────────────────
#  YÜZE GÖRE TRACKER ANAHTARI
# ─────────────────────────────────────────────
def face_key(cx: int, cy: int) -> tuple[int, int]:
    """Yüzü ızgara hücresine eşler; küçük kaymaları tolere eder."""
    return (cx // 50, cy // 50)


# ─────────────────────────────────────────────
#  ANA FONKSİYON
# ─────────────────────────────────────────────
def main() -> None:
    init_db()

    # InsightFace
    face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(160, 160))

    # MediaPipe FaceMesh
    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Kamera  (BUG FIX: açılmazsa erken çık)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.critical("❌ Kamera açılamadı. Program sonlandırılıyor.")
        face_mesh.close()
        return

    known_users = load_known_faces()
    logger.info(f"✅ {len(known_users)} kullanıcı yüklendi.")

    # Durum değişkenleri
    active_sessions: dict[str, float] = {}   # name → son log zamanı
    liveness_map: dict[tuple, LivenessTracker] = {}
    last_face_analysis: float = 0.0
    detected_faces: list = []

    logger.info("🚀 SİSTEM BAŞLATILDI. Çıkmak için 'q' tuşuna basın.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠ Kamera çerçevesi alınamadı.")
                break

            current_time = time.time()
            h, w = frame.shape[:2]

            # ── InsightFace: throttle edilmiş yüz algılama ──────────
            if current_time - last_face_analysis >= FACE_ANALYZE_PERIOD:
                detected_faces = face_app.get(frame)
                last_face_analysis = current_time

            # ── MediaPipe ──────────────────────────────────────────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_results = face_mesh.process(rgb)

            # ── Yeni çerçevede tracker'ları koru (BUG FIX) ──────────
            # Önceki döngüde "new_liveness_map" ile tamamen sıfırlanıyordu;
            # artık var olan tracker'lar güncellenerek korunuyor.
            active_keys: set[tuple] = set()

            for face in detected_faces:
                x1, y1, x2, y2 = (
                    max(0, int(face.bbox[0])),
                    max(0, int(face.bbox[1])),
                    min(w, int(face.bbox[2])),
                    min(h, int(face.bbox[3])),
                )
                if x2 <= x1 or y2 <= y1:        # geçersiz kutu
                    continue

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tk_key = face_key(cx, cy)
                active_keys.add(tk_key)

                tracker = liveness_map.setdefault(tk_key, LivenessTracker())

                # Doku analizi
                gray_roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                tracker.update_texture(compute_texture(gray_roi))

                # EAR analizi
                ear_val = 0.35   # varsayılan (göz görünmüyorsa canlı saysın)
                if mp_results.multi_face_landmarks:
                    for fm_lm in mp_results.multi_face_landmarks:
                        nx = int(fm_lm.landmark[1].x * w)
                        ny = int(fm_lm.landmark[1].y * h)
                        if abs(nx - cx) + abs(ny - cy) < (x2 - x1):
                            left_ear  = eye_aspect_ratio(fm_lm.landmark, LEFT_EYE_IDX,  w, h)
                            right_ear = eye_aspect_ratio(fm_lm.landmark, RIGHT_EYE_IDX, w, h)
                            ear_val   = (left_ear + right_ear) / 2.0
                            tracker.update_ear(ear_val)
                            break

                is_live = tracker.decide()

                # ── Yüz tanıma ────────────────────────────────────
                name, max_sim = "Bilinmeyen", 0.0
                emb = face.normed_embedding
                for user in known_users:
                    sim = float(np.dot(emb, user["emb"]))
                    if sim > max_sim:
                        max_sim = sim
                        name = user["name"] if sim > THRESHOLD else "Bilinmeyen"

                # ── Loglama (BUG FIX: SESSION_TIMEOUT artık kullanılıyor) ──
                if name != "Bilinmeyen" and is_live:
                    box_color = (0, 255, 0)
                    last_log_time = active_sessions.get(name, 0.0)
                    if current_time - last_log_time >= SESSION_TIMEOUT:
                        log_to_db(name)
                        active_sessions[name] = current_time
                elif name != "Bilinmeyen":
                    box_color = (0, 165, 255)
                else:
                    box_color = (0, 0, 255)

                # ── Görselleştirme ────────────────────────────────
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                frame = put_turkish_text(
                    frame, f"Kişi: {name} ({max_sim:.2f})",
                    (x1, max(0, y1 - 35)), 20, box_color
                )
                frame = put_turkish_text(
                    frame, tracker.reason,
                    (x1, max(0, y1 - 60)), 18, box_color
                )
                frame = put_turkish_text(
                    frame, f"EAR: {ear_val:.2f}",
                    (x1, min(h - 15, y2 + 5)), 14, (200, 200, 200)
                )

            # ── Eski tracker'ları temizle ─────────────────────────
            stale_keys = set(liveness_map) - active_keys
            for k in stale_keys:
                del liveness_map[k]

            # ── Oturum temizliği (sadece log için, tracker'dan bağımsız) ──
            expired = [
                u for u, t in active_sessions.items()
                if current_time - t > SESSION_TIMEOUT * 12   # ~1 dakika görünmezse sil
            ]
            for u in expired:
                del active_sessions[u]

            cv2.imshow("Turnike Kontrol Sistemi", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("⛔ Kullanıcı tarafından durduruldu.")
    finally:
        cap.release()
        face_mesh.close()           # BUG FIX: MediaPipe kaynakları serbest bırak
        cv2.destroyAllWindows()
        logger.info("👋 Program sonlandırıldı.")


if __name__ == "__main__":
    main()
