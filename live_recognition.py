import cv2
import sqlite3
import numpy as np
import time
import os
import logging
import json
from collections import deque
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from insightface.app import FaceAnalysis
from PIL import ImageFont, ImageDraw, Image
from scipy.spatial import distance as dist

# ─────────────────────────────────────────────
#  DOSYA YOLLARI  — script yanına sabit
# ─────────────────────────────────────────────
_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(_BASE_DIR, "turnike.db")
LOG_PATH    = os.path.join(_BASE_DIR, "turnike.log")
SEC_PATH    = os.path.join(_BASE_DIR, "security.log")
JSONL_PATH  = os.path.join(_BASE_DIR, "events.jsonl")

# ─────────────────────────────────────────────
#  LOG SİSTEMİ
# ─────────────────────────────────────────────

class EventType(str, Enum):
    SYSTEM_START    = "SYSTEM_START"
    SYSTEM_STOP     = "SYSTEM_STOP"
    BLINK_DETECTED  = "BLINK_DETECTED"
    ACCESS_GRANTED  = "ACCESS_GRANTED"
    ACCESS_DENIED   = "ACCESS_DENIED"
    SPOOF_DETECTED  = "SPOOF_DETECTED"
    ALERT_TRIGGERED = "ALERT_TRIGGERED"
    ALERT_CLEARED   = "ALERT_CLEARED"
    LIVENESS_FAIL   = "LIVENESS_FAIL"

@dataclass
class LogEvent:
    event_type:  str
    timestamp:   str   = ""
    unix_time:   float = 0.0
    user_name:   str   = ""
    similarity:  float = 0.0
    spoof_reason:str   = ""
    alert_level: str   = ""
    face_box:    str   = ""
    session_id:  str   = ""
    extra:       str   = ""

class TurnikeLogger:
    SECURITY_EVENTS = {
        EventType.SPOOF_DETECTED,
        EventType.ALERT_TRIGGERED,
        EventType.ALERT_CLEARED,
        EventType.ACCESS_DENIED,
        EventType.LIVENESS_FAIL,
    }

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_loggers()
        self._setup_db()
        self._jsonl = open(JSONL_PATH, "a", encoding="utf-8")

    def _setup_loggers(self):
        fmt     = "%(asctime)s | %(levelname)-8s | %(message)s"
        datefmt = "%d-%m-%Y %H:%M:%S"

        self.main = logging.getLogger("turnike.main")
        self.main.setLevel(logging.DEBUG)
        if not self.main.handlers:
            self.main.addHandler(self._fh(LOG_PATH, fmt, datefmt))
            self.main.addHandler(self._sh(fmt, datefmt))

        self.sec = logging.getLogger("turnike.security")
        self.sec.setLevel(logging.WARNING)
        if not self.sec.handlers:
            self.sec.addHandler(self._fh(
                SEC_PATH,
                "%(asctime)s | SECURITY | %(message)s",
                datefmt))

    @staticmethod
    def _fh(path, fmt, datefmt):
        h = logging.FileHandler(path, encoding="utf-8")
        h.setFormatter(logging.Formatter(fmt, datefmt))
        return h

    @staticmethod
    def _sh(fmt, datefmt):
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(fmt, datefmt))
        return h

    def _setup_db(self):
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY,
                name       TEXT NOT NULL,
                student_id TEXT UNIQUE,
                embedding  BLOB NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS access_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                event_type   TEXT NOT NULL,
                user_name    TEXT,
                similarity   REAL,
                spoof_reason TEXT,
                alert_level  TEXT,
                face_box     TEXT,
                extra        TEXT,
                timestamp    TEXT NOT NULL,
                unix_time    REAL NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS security_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                event_type   TEXT NOT NULL,
                spoof_reason TEXT,
                alert_level  TEXT,
                face_box     TEXT,
                user_name    TEXT,
                timestamp    TEXT NOT NULL,
                unix_time    REAL NOT NULL)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS logs (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL,
                timestamp TEXT NOT NULL)""")
            conn.commit()

    def log(self, event: LogEvent):
        event.session_id = self.session_id
        event.timestamp  = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        event.unix_time  = time.time()

        msg = self._fmt(event)
        if event.event_type in self.SECURITY_EVENTS:
            self.main.warning(msg)
            self.sec.warning(msg)
        else:
            self.main.info(msg)

        self._jsonl.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self._jsonl.flush()
        self._write_db(event)

    def _fmt(self, e: LogEvent) -> str:
        parts = [f"[{e.event_type}]"]
        if e.user_name:    parts.append(f"kullanici={e.user_name}")
        if e.similarity:   parts.append(f"benzerlik=%{int(e.similarity * 100)}")
        if e.spoof_reason: parts.append(f"sahte={e.spoof_reason}")
        if e.alert_level:  parts.append(f"alarm={e.alert_level}")
        if e.face_box:     parts.append(f"konum=[{e.face_box}]")
        if e.extra:        parts.append(f"not={e.extra}")
        return "  ".join(parts)

    def _write_db(self, e: LogEvent):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute(
                    """INSERT INTO access_log
                       (session_id, event_type, user_name, similarity,
                        spoof_reason, alert_level, face_box, extra,
                        timestamp, unix_time)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (e.session_id, e.event_type, e.user_name, e.similarity,
                     e.spoof_reason, e.alert_level, e.face_box,
                     e.extra, e.timestamp, e.unix_time))

                if e.event_type in self.SECURITY_EVENTS:
                    conn.execute(
                        """INSERT INTO security_log
                           (session_id, event_type, spoof_reason, alert_level,
                            face_box, user_name, timestamp, unix_time)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (e.session_id, e.event_type, e.spoof_reason,
                         e.alert_level, e.face_box, e.user_name,
                         e.timestamp, e.unix_time))

                if e.event_type == EventType.ACCESS_GRANTED and e.user_name:
                    conn.execute(
                        "INSERT INTO logs (user_name, timestamp) VALUES (?,?)",
                        (e.user_name, e.timestamp))

                conn.commit()
        except sqlite3.Error as ex:
            self.main.error(f"DB hatasi: {ex}")

    def close(self):
        self._jsonl.close()

# Global
tlog: TurnikeLogger = None  # type: ignore

# ─────────────────────────────────────────────
#  ACİL DURUM YÖNETİCİSİ
# ─────────────────────────────────────────────

class AlertLevel(str, Enum):
    NONE     = "NONE"
    WARNING  = "WARNING"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

class AlertManager:
    LOCKOUT_SEC = 30.0
    RESET_SEC   = 120.0

    def __init__(self):
        self.level         = AlertLevel.NONE
        self.spoof_count   = 0
        self.last_spoof_t  = 0.0
        self.lockout_until = 0.0

    def spoof_detected(self, reason: str, face_box: str = "") -> AlertLevel:
        now = time.time()
        self.spoof_count += 1
        self.last_spoof_t = now
        n = self.spoof_count
        if   n == 1: self.level = AlertLevel.WARNING
        elif n <= 3: self.level = AlertLevel.HIGH
        else:
            self.level         = AlertLevel.CRITICAL
            self.lockout_until = now + self.LOCKOUT_SEC
        tlog.log(LogEvent(event_type=EventType.ALERT_TRIGGERED,
                          spoof_reason=reason, alert_level=self.level,
                          face_box=face_box, extra=f"toplam={n}"))
        return self.level

    def is_locked(self) -> bool:
        return time.time() < self.lockout_until

    def tick(self, now: float):
        if self.level != AlertLevel.NONE and now - self.last_spoof_t > self.RESET_SEC:
            tlog.log(LogEvent(event_type=EventType.ALERT_CLEARED,
                              alert_level=self.level))
            self.level         = AlertLevel.NONE
            self.spoof_count   = 0
            self.lockout_until = 0.0

# ─────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────
THRESHOLD           = 0.45
SESSION_TIMEOUT     = 5.0
FACE_ANALYZE_PERIOD = 0.25

# Göz kırpma — iyileştirilmiş parametreler
# Göz kırpma parametreleri
CALIBRATION_FRAMES  = 60
EAR_RATIO           = 0.72        # açık EAR'ın bu oranı altı = kapalı
EAR_SMOOTH_N        = 2           # moving-average pencere (gürültü bastırma)
BLINK_WINDOW_SEC    = 6.0
BLINK_REQUIRED      = 1
BLINK_COOLDOWN      = 0.25        # iki kırpma arası min bekleme (s)

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# ─────────────────────────────────────────────
#  FONT
# ─────────────────────────────────────────────
_font_cache: dict = {}

def _font(size: int):
    if size not in _font_cache:
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]:
            if os.path.exists(p):
                _font_cache[size] = ImageFont.truetype(p, size)
                return _font_cache[size]
        _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def put_text(img, text, pos, size=20, color=(255, 255, 255)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text(pos, text, font=_font(size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ─────────────────────────────────────────────
#  VERİTABANI
# ─────────────────────────────────────────────
def load_known_faces() -> list:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute("SELECT name, embedding FROM users").fetchall()
        return [{"name": r[0], "emb": np.frombuffer(r[1], dtype=np.float32)} for r in rows]
    except sqlite3.Error as e:
        logging.error(f"Yukleme hatasi: {e}")
        return []

# ─────────────────────────────────────────────
#  YARDIMCILAR
# ─────────────────────────────────────────────
def ear_val(lm, idx, w, h):
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in idx], dtype=np.float32)
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aA = max(1, (a[2]-a[0]) * (a[3]-a[1]))
    aB = max(1, (b[2]-b[0]) * (b[3]-b[1]))
    return inter / (aA + aB - inter + 1e-6)

def recognize(emb, known):
    best_name, best_sim = "Bilinmeyen", 0.0
    for u in known:
        s = float(np.dot(emb, u["emb"]))
        if s > best_sim:
            best_sim  = s
            best_name = u["name"] if s >= THRESHOLD else "Bilinmeyen"
    return best_name, best_sim

def box_str(x1, y1, x2, y2): return f"{x1},{y1},{x2},{y2}"

# ─────────────────────────────────────────────
#  GÖZ KIRPMA  —  iyileştirilmiş
# ─────────────────────────────────────────────
class BlinkDetector:
    """
    Süre tabanlı göz kırpma tespiti.

    Gerçek insan kırpması: ~150–400 ms
      → 80 ms altı  : titreme / gürültü → sayılmaz
      → 350 ms üstü : kasıtlı kapama / uyuma → sayılmaz

    Her iki göz aynı anda kapanmalı (tek gözün kapanması sayılmaz).
    EAR moving-average ile kare gürültüsü bastırılır.
    """

    BLINK_MS_MIN = 80    # ms — bu süreden kısa kapanma gürültüdür
    BLINK_MS_MAX = 350   # ms — bu süreden uzun kapanma kırpma değildir

    def __init__(self):
        self.cal_buf:  list = []
        self.calibrated     = False
        self.thr            = 0.21

        self._l_buf = deque(maxlen=EAR_SMOOTH_N)
        self._r_buf = deque(maxlen=EAR_SMOOTH_N)

        self.was_closed  = False
        self.close_start = 0.0    # kapanmanın başladığı zaman (time.time())
        self.times: list = []
        self.last_blink  = 0.0
        self._box        = ""

    def set_box(self, b): self._box = b

    def _smooth(self, buf) -> float:
        return float(np.mean(buf)) if buf else 0.3

    def update(self, lm, w, h, now: float):
        l_ear = ear_val(lm, LEFT_EYE_IDX,  w, h)
        r_ear = ear_val(lm, RIGHT_EYE_IDX, w, h)
        self._l_buf.append(l_ear)
        self._r_buf.append(r_ear)

        # ── Kalibrasyon ──────────────────────────────────────────────
        if not self.calibrated:
            avg = (self._smooth(self._l_buf) + self._smooth(self._r_buf)) / 2.0
            if avg > 0.22:
                self.cal_buf.append(avg)
            if len(self.cal_buf) >= CALIBRATION_FRAMES:
                s       = sorted(self.cal_buf)
                trim_lo = len(s) // 4
                trim_hi = max(trim_lo + 1, int(len(s) * 0.90))
                base    = float(np.mean(s[trim_lo:trim_hi]))
                self.thr = base * EAR_RATIO
                self.calibrated = True
            return

        # ── Kırpma tespiti — gerçek süre ────────────────────────────
        l_closed    = self._smooth(self._l_buf) < self.thr
        r_closed    = self._smooth(self._r_buf) < self.thr
        both_closed = l_closed and r_closed

        if both_closed:
            if not self.was_closed:
                self.close_start = now   # kapanma başladı, zamanı kaydet
            self.was_closed = True
        else:
            if self.was_closed:
                # Kapanma bitti — kaç ms sürdü?
                ms = (now - self.close_start) * 1000.0
                since_last = now - self.last_blink
                if self.BLINK_MS_MIN <= ms <= self.BLINK_MS_MAX and since_last >= BLINK_COOLDOWN:
                    self.times.append(now)
                    self.last_blink = now
                    tlog.log(LogEvent(
                        event_type = EventType.BLINK_DETECTED,
                        face_box   = self._box,
                        extra      = f"toplam={len(self.times)} sure={ms:.0f}ms"))
            self.was_closed = False

        self.times = [t for t in self.times if t > now - BLINK_WINDOW_SEC]

    def is_live(self) -> bool:
        return self.calibrated and len(self.times) >= BLINK_REQUIRED

    def status_str(self) -> str:
        if not self.calibrated:
            pct = int(len(self.cal_buf) / CALIBRATION_FRAMES * 100)
            return f"Kalibrasyon %{pct}"
        return "Canli" if self.is_live() else "Goz kirpin"

# ─────────────────────────────────────────────
#  CANLILIK TAKİPÇİSİ
# ─────────────────────────────────────────────
class LivenessTracker:
    def __init__(self):
        self.bl      = BlinkDetector()
        self.is_live = False
        self.status  = "Bekleniyor"

    def set_box(self, b): self.bl.set_box(b)

    def update(self, lm, w, h, now):
        self.bl.update(lm, w, h, now)

    def decide(self, name, sim, box_s, alert_mgr: AlertManager) -> bool:
        ok           = self.bl.is_live()
        self.is_live = ok
        self.status  = self.bl.status_str()
        return ok

    def reset(self):
        self.bl      = BlinkDetector()
        self.is_live = False
        self.status  = "Bekleniyor"

# ─────────────────────────────────────────────
#  TRACKER HAVUZU
# ─────────────────────────────────────────────
class TrackerPool:
    IOU_MIN = 0.30

    def __init__(self): self._pool: dict = {}

    def match(self, new_boxes):
        matched, used = {}, set()
        for nb in new_boxes:
            bv, bk = 0.0, None
            for ok in self._pool:
                v = iou(nb, ok)
                if v > bv: bv, bk = v, ok
            matched[nb] = self._pool[bk] if bk and bv >= self.IOU_MIN else LivenessTracker()
            if bk: used.add(bk)
        for ok, tr in self._pool.items():
            if ok not in used: tr.reset()
        self._pool = matched
        return matched

# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────

# Renk paleti (BGR)
C_GREEN = ( 80, 220, 120)
C_AMBER = ( 40, 170, 255)
C_RED   = ( 60,  60, 210)
C_WHITE = (240, 240, 240)
C_DIM   = (130, 130, 130)

ALERT_BG = {
    AlertLevel.WARNING:  ( 20, 160, 220),
    AlertLevel.HIGH:     ( 20,  60, 200),
    AlertLevel.CRITICAL: ( 10,  10, 160),
}
ALERT_TEXT = {
    AlertLevel.WARNING:  "UYARI  —  Sahte yuz denemesi",
    AlertLevel.HIGH:     "YUKSEK ALARM  —  Tekrar sahte deneme",
    AlertLevel.CRITICAL: "KRITIK  —  SISTEM KILITLI",
}

def _rect_alpha(frame, x1, y1, x2, y2, color, alpha=0.55):
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

def _brackets(frame, x1, y1, x2, y2, color, gap=4, thick=2):
    length = max(14, (x2 - x1) // 5)
    for cx, cy, sx, sy in [
        (x1 - gap, y1 - gap,  1,  1),
        (x2 + gap, y1 - gap, -1,  1),
        (x1 - gap, y2 + gap,  1, -1),
        (x2 + gap, y2 + gap, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + sx * length, cy),           color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx,               cy + sy * length), color, thick, cv2.LINE_AA)

def draw_face(frame, x1, y1, x2, y2, name, sim, status, is_live):
    img_h = frame.shape[0]

    bc = C_GREEN if (name != "Bilinmeyen" and is_live) else \
         C_AMBER if name != "Bilinmeyen" else C_RED
    sc = C_GREEN if is_live else (C_RED if "Sahte" in status else C_AMBER)

    _brackets(frame, x1, y1, x2, y2, bc)

    # Alt bilgi şeridi
    bar_h  = 44
    bx1, bx2 = x1 - 4, x2 + 4
    by1 = min(y2 + 4, img_h - bar_h - 2)
    by2 = by1 + bar_h

    _rect_alpha(frame, bx1, by1, bx2, by2, (12, 12, 12), alpha=0.65)
    cv2.line(frame, (bx1, by1), (bx2, by1), bc, 1, cv2.LINE_AA)

    lbl = f"{name}  %{int(sim * 100)}" if name != "Bilinmeyen" else "Bilinmeyen"
    frame = put_text(frame, lbl,    (bx1 + 6, by1 + 3),  size=17, color=bc)
    frame = put_text(frame, status, (bx1 + 6, by1 + 24), size=13, color=sc)
    return frame

def draw_hud(frame, face_count: int, alert_mgr: AlertManager):
    h, w    = frame.shape[:2]
    now_str = datetime.now().strftime("%H:%M:%S")

    # Sol üst panel
    _rect_alpha(frame, 0, 0, 220, 36, (10, 10, 10), alpha=0.55)
    frame = put_text(frame, now_str,              (10, 8), size=16, color=C_WHITE)
    frame = put_text(frame, f"Yuz: {face_count}", (125, 8), size=16, color=C_DIM)

    # Alarm şeridi
    if alert_mgr.level != AlertLevel.NONE:
        blink_on = True
        if alert_mgr.level == AlertLevel.CRITICAL:
            blink_on = int(time.time() * 2) % 2 == 0
        bg = ALERT_BG.get(alert_mgr.level, (30, 30, 180))
        if blink_on:
            _rect_alpha(frame, 0, 0, w, 40, bg, alpha=0.85)
        frame = put_text(frame, ALERT_TEXT.get(alert_mgr.level, "ALARM"),
                         (12, 10), size=18, color=C_WHITE)
        if alert_mgr.is_locked():
            rem = max(0, alert_mgr.lockout_until - time.time())
            frame = put_text(frame, f"{rem:.0f}s", (w - 55, 10),
                             size=17, color=C_WHITE)
    return frame

# ─────────────────────────────────────────────
#  ANA FONKSİYON
# ─────────────────────────────────────────────
def main():
    global tlog
    tlog = TurnikeLogger()
    tlog.log(LogEvent(event_type=EventType.SYSTEM_START, extra="v9.0"))

    face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(160, 160))

    import mediapipe as mp
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=5, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        tlog.main.critical("Kamera açılamadı.")
        face_mesh.close(); tlog.close(); return

    known_users  = load_known_faces()
    active_sess: dict = {}
    pool         = TrackerPool()
    alert_mgr    = AlertManager()
    det_faces:   list = []
    last_det     = 0.0

    tlog.main.info(f"{len(known_users)} kullanici yuklendi. DB: {DB_PATH}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            now  = time.time()
            h, w = frame.shape[:2]
            alert_mgr.tick(now)

            if alert_mgr.is_locked():
                frame = draw_hud(frame, 0, alert_mgr)
                cv2.imshow("Turnike", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue

            if now - last_det >= FACE_ANALYZE_PERIOD:
                det_faces = face_app.get(frame)
                last_det  = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpr = face_mesh.process(rgb)

            boxes: list = []
            fmap:  dict = {}
            for face in det_faces:
                x1 = max(0, int(face.bbox[0])); y1 = max(0, int(face.bbox[1]))
                x2 = min(w, int(face.bbox[2])); y2 = min(h, int(face.bbox[3]))
                if x2 > x1 and y2 > y1:
                    b = (x1, y1, x2, y2)
                    boxes.append(b)
                    fmap[b] = face

            trackers = pool.match(boxes)

            for box, tr in trackers.items():
                x1, y1, x2, y2 = box
                face = fmap[box]
                bs   = box_str(x1, y1, x2, y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tr.set_box(bs)

                best_lm, best_d = None, float("inf")
                if mpr.multi_face_landmarks:
                    for lm in mpr.multi_face_landmarks:
                        nx = int(lm.landmark[1].x * w)
                        ny = int(lm.landmark[1].y * h)
                        d  = (nx - cx) ** 2 + (ny - cy) ** 2
                        if d < best_d:
                            best_d, best_lm = d, lm

                if best_lm:
                    tr.update(best_lm.landmark, w, h, now)

                name, sim = recognize(face.normed_embedding, known_users)
                is_live   = tr.decide(name, sim, bs, alert_mgr)

                if is_live:
                    k = name + bs
                    if now - active_sess.get(k, 0.0) >= SESSION_TIMEOUT:
                        active_sess[k] = now
                        ev = (EventType.ACCESS_GRANTED
                              if name != "Bilinmeyen"
                              else EventType.ACCESS_DENIED)
                        tlog.log(LogEvent(
                            event_type   = ev,
                            user_name    = name,
                            similarity   = sim,
                            face_box     = bs,
                            spoof_reason = "" if name != "Bilinmeyen" else "unknown_live",
                        ))

                frame = draw_face(frame, x1, y1, x2, y2,
                                  name, sim, tr.status, is_live)

            frame = draw_hud(frame, len(boxes), alert_mgr)
            active_sess = {k: t for k, t in active_sess.items() if now - t <= 60.0}

            cv2.imshow("Turnike", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        tlog.main.info("Kullanici tarafindan durduruldu.")
    finally:
        tlog.log(LogEvent(event_type=EventType.SYSTEM_STOP))
        cap.release()
        face_mesh.close()
        cv2.destroyAllWindows()
        tlog.close()

if __name__ == "__main__":
    main()