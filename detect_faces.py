import cv2
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
DETECT_PERIOD  = 0.5    # saniye: AI modeli ne sıklıkla çalışsın
CAM_WIDTH      = 640
CAM_HEIGHT     = 480

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
             size: int = 18, color: tuple = (255, 255, 255)) -> np.ndarray:
    """BGR görüntüye Türkçe destekli metin yazar."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(pos, text, font=_get_font(size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ─────────────────────────────────────────────
#  LANDMARK ÇİZİMİ (5 nokta)
# ─────────────────────────────────────────────
LANDMARK_COLORS = [
    (0, 255, 255),   # Sol göz
    (0, 255, 255),   # Sağ göz
    (0, 200, 255),   # Burun
    (255, 150, 0),   # Sol ağız köşesi
    (255, 150, 0),   # Sağ ağız köşesi
]

def draw_landmarks(img: np.ndarray, kps: np.ndarray) -> None:
    """InsightFace 5-nokta landmark'larını çizer."""
    if kps is None:
        return
    for i, (x, y) in enumerate(kps.astype(int)):
        color = LANDMARK_COLORS[i % len(LANDMARK_COLORS)]
        cv2.circle(img, (x, y), 3, color, -1)

# ─────────────────────────────────────────────
#  FPS SAYACI
# ─────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: list[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.time()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])

# ─────────────────────────────────────────────
#  ANA FONKSİYON
# ─────────────────────────────────────────────
def main() -> None:
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"

    # Model yükle
    logger.info("🔄 buffalo_s modeli yükleniyor (SCRFD + ArcFace)...")
    face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(320, 320))
    logger.info("✅ Model hazır.")

    # Kamera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.critical("❌ Kamera açılamadı. Program sonlandırılıyor.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"📷 Kamera çözünürlüğü: {actual_w}×{actual_h}")
    logger.info("🚀 Demo modu başlatıldı. Çıkış: [Q]")

    detected_faces: list = []
    last_detect:   float = 0.0
    fps_counter = FPSCounter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠ Kamera çerçevesi alınamadı.")
                continue

            now = time.time()
            h, w = frame.shape[:2]

            # ── Throttle: sadece DETECT_PERIOD'da bir AI çalıştır ──
            if now - last_detect >= DETECT_PERIOD:
                detected_faces = face_app.get(frame)
                last_detect = now

            # ── Her yüz için görselleştirme ───────────────────────
            for face in detected_faces:
                x1 = max(0, int(face.bbox[0]))
                y1 = max(0, int(face.bbox[1]))
                x2 = min(w, int(face.bbox[2]))
                y2 = min(h, int(face.bbox[3]))

                if x2 <= x1 or y2 <= y1:
                    continue

                det_score = float(face.det_score)
                color = (0, 255, 0) if det_score >= 0.7 else (0, 165, 255)

                # Bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Landmark noktaları
                draw_landmarks(frame, getattr(face, "kps", None))

                # Embedding bilgisi: boyut + L2 normu (kayıt kalitesi göstergesi)
                emb  = face.normed_embedding
                norm = float(np.linalg.norm(emb))
                frame = put_text(
                    frame,
                    f"Det: {det_score:.2f}  Emb: {emb.shape[0]}d  Norm: {norm:.3f}",
                    (x1, max(0, y1 - 28)),
                    size=16,
                    color=color,
                )

            # ── HUD: FPS + yüz sayısı ─────────────────────────────
            fps = fps_counter.tick()
            frame = put_text(frame, f"FPS: {fps:.1f}", (8, 8),  size=18, color=(200, 200, 200))
            frame = put_text(frame, f"Yüz: {len(detected_faces)}", (8, 30), size=18, color=(200, 200, 200))
            frame = put_text(frame, "SCRFD + ArcFace (buffalo_s)", (8, h - 26), size=16, color=(100, 100, 100))

            cv2.imshow("Turnike Demo - SCRFD & ArcFace", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("👋 Program sonlandırıldı.")

if __name__ == "__main__":
    main()