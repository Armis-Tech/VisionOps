"""Nexora Vision AI - Intelligent Traffic Surveillance + ITMS / e-Challan Workflow (v3)
PySide6 desktop dashboard. Uses OpenCV, Ultralytics YOLO and SQLite.

Changes in v3
  * Every page scrolls as a whole (mouse wheel / scrollbar) and tables still scroll inside.
  * The window opens maximised and fits small laptop screens.
  * Refresh buttons (and F5) always give visible feedback and report errors instead of failing silently.
  * More exact counting: a vehicle is counted once, after it is tracked for several frames, with its
    most-voted class. Optional counting line. History rows now carry a run id.
  * History tools: scope filter, per-class totals, delete legacy per-frame rows, clear all (with backup).
  * Choose the YOLO model and image size in Settings; check your violations.pt classes with one click.
"""
from __future__ import annotations
import sys, re, csv, json, sqlite3, time, threading, faulthandler, traceback, html
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime
from poc_support import ensure_column, should_count_vehicle, normalize_plate

import cv2
import numpy as np
from ultralytics import YOLO
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QIcon, QPdfWriter, QPageSize, QTextDocument, QShortcut, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QStackedWidget, QFileDialog, QMessageBox, QLineEdit,
    QComboBox, QTableWidget, QTableWidgetItem, QHeaderView, QFormLayout,
    QDoubleSpinBox, QSpinBox, QProgressBar, QTextEdit, QScrollArea, QSplitter,
    QDialog, QDialogButtonBox, QButtonGroup, QCheckBox, QSizePolicy, QAbstractItemView
)

# --------------------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"                    # default COCO model
VIOLATION_MODEL_PATH = BASE_DIR / "violations.pt"       # optional custom-trained model
VIOLATION_MAP_PATH = BASE_DIR / "violation_classes.json"  # optional class-name -> violation key map
LOGO_PATH = BASE_DIR / "assets" / "nexora_logo.png"
DB_PATH = BASE_DIR / "traffic_history.db"
OUTPUT_DIR = BASE_DIR / "processed_videos"
EVIDENCE_DIR = BASE_DIR / "evidence"
OUTPUT_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR.mkdir(exist_ok=True)

_crash_log = open(BASE_DIR / "crash.log", "a")   # native crashes are written here
faulthandler.enable(_crash_log)


def _excepthook(exc_type, exc, tb):
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        with open(BASE_DIR / "app_errors.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}]\n{text}\n")
    except OSError:
        pass
    print(text, file=sys.stderr)


sys.excepthook = _excepthook

# --------------------------------------------------------------------------- domain data
VEHICLES = {1: "Bicycle", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
PERSON_CLASS = 0

# key: (label, default fine in INR, needs enhanced review). Fines are placeholders: set yours in Settings.
VIOLATIONS = {
    "helmet":   ("Riding without helmet", 1000, False),
    "triple":   ("Triple riding", 1000, False),
    "belt":     ("Driving without seat belt", 1000, False),
    "wrongway": ("Wrong-way / no entry", 500, False),
    "lights":   ("Unauthorized police lights", 1000, True),
    "rooftop":  ("Person sitting on top of vehicle", 1000, True),
}
STATUS = {
    "pending": "Pending review", "reeval": "Re-evaluation", "accepted": "Operator accepted",
    "approved": "Authority approved", "issued": "Challan issued", "paid": "Paid", "rejected": "Rejected",
}
# key: (button label, role allowed ('*' = any), allowed from-states, new state)
ACTIONS = {
    "accept":  ("Accept", "operator", ("pending", "reeval"), "accepted"),
    "reeval":  ("Send for re-evaluation", "operator", ("pending",), "reeval"),
    "approve": ("Approve", "authority", ("accepted",), "approved"),
    "issue":   ("Generate DEMO e-Challan", "authority", ("approved",), "issued"),
    "paid":    ("Mark as paid", "authority", ("issued",), "paid"),
    "reject":  ("Reject", "*", ("pending", "reeval", "accepted"), "rejected"),
}
PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$")

NAME_HINTS = {   # class-name hints for a custom violation model
    "helmet": ("no_helmet", "nohelmet", "without_helmet", "no_hardhat"),
    "belt": ("no_seatbelt", "no_seat_belt", "noseatbelt", "without_seatbelt", "without_seat_belt"),
    "lights": ("police_light", "beacon", "siren"),
    "rooftop": ("person_on_top", "person_on_roof", "rooftop", "on_roof"),
    "triple": ("triple_riding", "triple"),
}


def norm_name(name):
    return str(name).lower().replace("-", "_").replace(" ", "_")


def build_violation_map(names):
    """class index -> violation key (or None). violation_classes.json overrides the name hints."""
    custom = {}
    if VIOLATION_MAP_PATH.exists():
        try:
            custom = {norm_name(k): v for k, v in json.loads(VIOLATION_MAP_PATH.read_text(encoding="utf-8")).items()}
        except (OSError, ValueError):
            custom = {}
    out = {}
    for idx, name in names.items():
        n = norm_name(name)
        key = custom.get(n)
        if key is None:
            key = next((k for k, hints in NAME_HINTS.items() if any(h in n for h in hints)), None)
        out[idx] = key if key in VIOLATIONS else None
    return out


def norm_plate(text):
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------- database
class Database:
    """One shared, lock-protected SQLite connection (safe to use from worker threads)."""

    def __init__(self, path):
        self._lock = threading.RLock()
        self._con = sqlite3.connect(str(path), check_same_thread=False, timeout=15)
        with self._lock:
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
            self._init_schema()

    def _init_schema(self):
        c = self._con
        c.execute("""CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT, event_time TEXT NOT NULL,
            source TEXT, vehicle_type TEXT, confidence REAL, frame_number INTEGER, video_path TEXT)""")
        if "run_id" not in [r[1] for r in c.execute("PRAGMA table_info(detections)").fetchall()]:
            c.execute("ALTER TABLE detections ADD COLUMN run_id INTEGER")   # NULL = legacy per-frame row
        c.execute("""CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, stream_url TEXT,
            location TEXT, created_at TEXT)""")
        c.execute("""CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, event_time TEXT,
            violation_type TEXT, plate TEXT DEFAULT '', camera TEXT, location TEXT DEFAULT '',
            confidence REAL, status TEXT DEFAULT 'pending', evidence_path TEXT DEFAULT '',
            challan_no TEXT, fine_amount INTEGER, challan_date TEXT, note TEXT DEFAULT '')""")
        c.execute("""CREATE TABLE IF NOT EXISTS audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT, violation_id INTEGER, at TEXT, role TEXT,
            action TEXT, note TEXT)""")
        c.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        # Preserve all older rows while adding a link back to source evidence.
        ensure_column(c, "detections", "track_id", "INTEGER")
        for name, definition in (("run_id", "INTEGER"), ("track_id", "INTEGER"),
                                 ("frame_number", "INTEGER"), ("video_path", "TEXT")):
            ensure_column(c, "violations", name, definition)
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_detection_track "
                  "ON detections(run_id,track_id) "
                  "WHERE run_id IS NOT NULL AND track_id IS NOT NULL")
        c.execute("CREATE INDEX IF NOT EXISTS ix_violation_status ON violations(status)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_violation_run ON violations(run_id)")
        c.commit()

    def execute(self, sql, params=()):
        with self._lock:
            cur = self._con.execute(sql, params)
            self._con.commit()
            return cur.lastrowid

    def query(self, sql, params=()):
        with self._lock:
            return self._con.execute(sql, params).fetchall()

    def scalar(self, sql, params=(), default=0):
        rows = self.query(sql, params)
        return rows[0][0] if rows and rows[0][0] is not None else default

    def get_setting(self, key, default=""):
        rows = self.query("SELECT value FROM settings WHERE key=?", (key,))
        return rows[0][0] if rows else default

    def set_setting(self, key, value):
        self.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, str(value)))

    def next_counter(self, key):
        with self._lock:
            value = int(self.get_setting(key, "0") or 0) + 1
            self.set_setting(key, value)
            return value

    def next_challan_no(self, prefix):
        return f"{prefix}{datetime.now():%Y%m%d}{self.next_counter('challan_seq'):05d}"

    def apply_review_action(self, vid, expected_status, new_status, role, label,
                            plate, location, note, *, challan_no=None, fine_amount=None):
        """Commit status, edits and audit in one transaction, or none of them."""
        with self._lock:
            con = self._con
            try:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute("SELECT plate,location,status FROM violations WHERE id=?", (vid,)).fetchone()
                if row is None or row[2] != expected_status:
                    raise ValueError("Violation changed since it was selected. Refresh and retry.")
                if plate != (row[0] or ""):
                    con.execute("INSERT INTO audit(violation_id,at,role,action,note) VALUES(?,?,?,?,?)",
                                (vid, now_str(), role, "Plate corrected", f"{row[0] or '(blank)'} to {plate or '(blank)'}"))
                con.execute("UPDATE violations SET plate=?, location=?, status=? WHERE id=?",
                            (plate, location, new_status, vid))
                if challan_no is not None:
                    con.execute("UPDATE violations SET challan_no=?, fine_amount=?, challan_date=? WHERE id=?",
                                (challan_no, fine_amount, now_str(), vid))
                con.execute("INSERT INTO audit(violation_id,at,role,action,note) VALUES(?,?,?,?,?)",
                            (vid, now_str(), role, label, note))
                con.commit()
            except Exception:
                con.rollback()
                raise

    def backup(self, path):
        with self._lock:
            dst = sqlite3.connect(str(path))
            try:
                self._con.backup(dst)
            finally:
                dst.close()


db = Database(DB_PATH)


def save_detection(source, vehicle_type, confidence, frame_number=None,
                   video_path=None, run_id=None, track_id=None):
    db.execute("INSERT OR IGNORE INTO detections("
               "event_time,source,vehicle_type,confidence,frame_number,video_path,run_id,track_id) "
               "VALUES(?,?,?,?,?,?,?,?)",
               (now_str(), source, vehicle_type, float(confidence),
                frame_number, video_path, run_id, track_id))


def write_jpeg(path, frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(str(path))
    return ok


def read_image(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except OSError:
        return None


def audit(violation_id, role, action, note=""):
    db.execute("INSERT INTO audit(violation_id,at,role,action,note) VALUES(?,?,?,?,?)",
               (violation_id, now_str(), role, action, note))


def save_violation(type_key, frame, source, confidence=None, note="", plate="",
                   location="", actor="system", *, run_id=None, track_id=None,
                   frame_number=None, video_path=None):
    """Record a review-only violation candidate, linking it to its source."""
    if type_key not in VIOLATIONS:
        raise ValueError(f"Unsupported violation class: {type_key}")
    path = ""
    if frame is not None:
        ev = frame.copy()
        cv2.putText(ev, f"{VIOLATIONS[type_key][0]} | {now_str()} | {source}"[:110],
                    (12, max(16, ev.shape[0] - 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        path = str(EVIDENCE_DIR / f"{type_key}_{time.time_ns()}.jpg")
        if not write_jpeg(path, ev):
            raise OSError(f"Could not save evidence image: {path}")
    vid = db.execute(
        "INSERT INTO violations(created_at,event_time,violation_type,plate,camera,location,"
        "confidence,status,evidence_path,note,run_id,track_id,frame_number,video_path) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (now_str(), now_str(), type_key, norm_plate(plate), source, location,
         confidence, "pending", path, note, run_id, track_id, frame_number, video_path))
    audit(vid, actor, "Review candidate saved", note)
    return vid


# --------------------------------------------------------------------------- detection engine
class FrameAnalyzer:
    """Tracking, exact-once counting and violation candidates. Used by both workers."""
    MIN_AGE = 6          # a track must be seen this many frames before it is counted
    MIN_MOVE_PX = 60     # minimum movement for the wrong-way rule

    def __init__(self, source_label, opts, video_path=None):
        self.source = source_label
        self.video_path = video_path
        self.conf = float(opts.get("conf", 0.35))
        self.rules = bool(opts.get("rules", True))
        self.legal = opts.get("legal", "Off")
        self.imgsz = int(opts.get("imgsz", 640))
        self.line_pct = int(opts.get("line", 0))
        self.min_count_move = int(opts.get("min_count_move", 25))
        self.run_id = db.next_counter("run_seq")
        self.model = YOLO(str(opts.get("model_path", MODEL_PATH)))
        self.classes = list(VEHICLES.keys()) + ([PERSON_CLASS] if self.rules else [])
        # The selected primary model must use COCO indices for this POC.
        expected = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        names = self.model.names
        if any(str(names.get(k, "")).lower() != label for k, label in expected.items()):
            raise ValueError("Primary YOLO model must have standard COCO class IDs. "
                             "Use yolo11n.pt for vehicle detection or implement a class map.")
        self.vmodel, self.vmap = None, {}
        if self.rules and VIOLATION_MODEL_PATH.exists():
            self.vmodel = YOLO(str(VIOLATION_MODEL_PATH))
            self.vmap = build_violation_map(self.vmodel.names)
        self.age = defaultdict(int)
        self.votes = defaultdict(lambda: defaultdict(int))
        self.conf_sum = defaultdict(float)
        self.last_cy = {}
        self.counted = set()
        self.history = defaultdict(lambda: deque(maxlen=30))
        self.triple_hits = defaultdict(int)
        self.vhits = defaultdict(int)
        self.flagged = set()

    @property
    def unique(self):
        return len(self.counted)

    def process(self, frame, frame_no):
        """Returns (annotated_frame, vehicles_in_frame, rows_saved)."""
        h, w = frame.shape[:2]
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.conf,
                                  classes=self.classes, imgsz=self.imgsz, verbose=False)[0]
        annotated = result.plot()
        line_y = int(h * self.line_pct / 100) if self.line_pct > 0 else None
        if line_y is not None:
            cv2.line(annotated, (0, line_y), (w, line_y), (0, 200, 255), 2)
            cv2.putText(annotated, "COUNT LINE", (10, max(18, line_y - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2, cv2.LINE_AA)
        in_frame, saved = 0, 0
        pending = []     # (violation key, confidence)
        boxes = result.boxes
        if boxes is not None and len(boxes):
            cls_l = boxes.cls.int().tolist()
            conf_l = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(cls_l)
            persons = [b for c, b in zip(cls_l, xyxy) if c == PERSON_CLASS]
            for cls_id, conf, box, tid in zip(cls_l, conf_l, xyxy, ids):
                if cls_id not in VEHICLES:
                    continue
                in_frame += 1
                if tid is None:
                    continue
                self.age[tid] += 1
                self.votes[tid][cls_id] += 1
                self.conf_sum[tid] += conf
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                self.history[tid].append((cx, cy))
                if tid not in self.counted:
                    if line_y is None:
                        ready = should_count_vehicle(self.age[tid], self.history[tid],
                                                     self.last_cy.get(tid), cy, None,
                                                     self.min_count_move, self.MIN_AGE)
                    else:
                        ready = should_count_vehicle(self.age[tid], self.history[tid],
                                                     self.last_cy.get(tid), cy, line_y,
                                                     self.min_count_move, self.MIN_AGE)
                    if ready:
                        self.counted.add(tid)
                        votes = self.votes[tid]
                        major = max(votes, key=votes.get)
                        save_detection(self.source, VEHICLES[major], self.conf_sum[tid] / self.age[tid],
                                        frame_no, self.video_path, self.run_id, tid)
                        saved += 1
                self.last_cy[tid] = cy
                if not self.rules:
                    continue
                if cls_id == 3 and ("triple", tid) not in self.flagged:
                    x1, y1, x2, y2 = box
                    bw, bh = x2 - x1, y2 - y1
                    riders = sum(1 for p in persons
                                 if x1 - 0.15 * bw <= (p[0] + p[2]) / 2 <= x2 + 0.15 * bw
                                 and y1 - bh <= (p[1] + p[3]) / 2 <= y2)
                    self.triple_hits[tid] = self.triple_hits[tid] + 1 if riders >= 3 else 0
                    if self.triple_hits[tid] >= 4:
                        self.flagged.add(("triple", tid))
                        pending.append(("triple", conf, tid))
                if self.legal != "Off" and ("wrongway", tid) not in self.flagged and len(self.history[tid]) >= 20:
                    hist = self.history[tid]
                    dx, dy = hist[-1][0] - hist[0][0], hist[-1][1] - hist[0][1]
                    wrong = ((self.legal == "Down" and dy < -self.MIN_MOVE_PX and abs(dy) > abs(dx)) or
                             (self.legal == "Up" and dy > self.MIN_MOVE_PX and abs(dy) > abs(dx)) or
                             (self.legal == "Right" and dx < -self.MIN_MOVE_PX and abs(dx) > abs(dy)) or
                             (self.legal == "Left" and dx > self.MIN_MOVE_PX and abs(dx) > abs(dy)))
                    if wrong:
                        self.flagged.add(("wrongway", tid))
                        pending.append(("wrongway", conf, tid))
        # Optional custom model (helmet / seat belt / police lights / person on top), tracked so each case is flagged once.
        if self.vmodel is not None:
            vr = self.vmodel.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.conf,
                                   imgsz=self.imgsz, verbose=False)[0]
            vb = vr.boxes
            if vb is not None and len(vb):
                v_ids = vb.id.int().tolist() if vb.id is not None else [None] * len(vb)
                for ci, cf, bx, vt in zip(vb.cls.int().tolist(), vb.conf.tolist(), vb.xyxy.tolist(), v_ids):
                    key = self.vmap.get(ci)
                    if not key:
                        continue
                    p1, p2 = (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3]))
                    cv2.rectangle(annotated, p1, p2, (0, 0, 255), 3)
                    cv2.putText(annotated, VIOLATIONS[key][0], (p1[0], max(18, p1[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    if vt is None:
                        continue
                    k = (key, vt)
                    self.vhits[k] += 1
                    if self.vhits[k] >= 3 and k not in self.flagged:
                        self.flagged.add(k)
                        # vt belongs to the *violation* model tracker, not the
                        # vehicle tracker. Do not falsely link the two IDs.
                        pending.append((key, cf, None))
        for key, conf, track_id in pending:
            save_violation(key, annotated, self.source, confidence=round(conf * 100, 1),
                           note="Unverified AI candidate; operator review required.",
                           run_id=self.run_id, track_id=track_id,
                           frame_number=frame_no, video_path=self.video_path)
            saved += 1
        return annotated, in_frame, saved


class VideoPreview(QLabel):
    """Keeps the latest frame and redraws it at the current label size."""

    def __init__(self, text=""):
        super().__init__(text)
        self._frame = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(False)
        self.setMinimumSize(200, 140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def current_frame(self):
        return self._frame

    def set_frame(self, frame):
        self._frame = frame.copy() if frame is not None else None
        self._render_frame()

    def clear_frame(self, message=""):
        self._frame = None
        self.clear()
        self.setText(message)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_frame()

    def _render_frame(self):
        if self._frame is None or self.width() < 10 or self.height() < 10:
            return
        rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(image).scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


# --------------------------------------------------------------------------- workers
class VideoWorker(QThread):
    frame_ready = Signal(object)
    progress = Signal(int)
    data_changed = Signal()
    finished_info = Signal(str, int, bool, bool)  # path, tracks, stopped, succeeded
    error = Signal(str)

    def __init__(self, video_path, opts):
        super().__init__()
        self.video_path = str(video_path)
        self.opts = opts
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        cap, writer = None, None
        out_path, unique, success, frame_no = "", 0, False, 0
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise RuntimeError("Could not open selected video. Check path and codec.")
            ok, first_frame = cap.read()
            if not ok or first_frame is None:
                raise RuntimeError("Video opened but has no readable frames.")
            total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            fps = fps if 1 <= fps <= 240 else 25.0
            height, width = first_frame.shape[:2]
            src = Path(self.video_path)
            out_path = str(OUTPUT_DIR / f"{src.stem}_processed_{time.time_ns()}.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Cannot create processed MP4. Check the output path and codec.")
            analyzer = FrameAnalyzer("Offline: " + src.name, self.opts, self.video_path)
            last_emit = 0.0
            frame = first_frame
            while not self._stop_requested and frame is not None:
                frame_no += 1
                annotated, in_frame, saved = analyzer.process(frame, frame_no)
                unique = analyzer.unique
                cv2.putText(annotated, f"Frame {frame_no} | in frame: {in_frame} | counted: {unique}",
                            (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 120, 210), 2, cv2.LINE_AA)
                writer.write(annotated)
                if saved:
                    self.data_changed.emit()
                now = time.monotonic()
                if now - last_emit >= 0.04:
                    last_emit = now
                    self.frame_ready.emit(annotated)
                    self.progress.emit(min(99, int(frame_no * 100 / total)))
                ok, next_frame = cap.read()
                frame = next_frame if ok else None
            success = frame_no > 0 and not self._stop_requested
        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            self.finished_info.emit(out_path, unique, self._stop_requested, success)


class LiveCameraWorker(QThread):
    frame_ready = Signal(object, int, int)   # annotated frame, vehicles in frame, counted
    data_changed = Signal()
    error = Signal(str)
    stopped = Signal()

    def __init__(self, source, opts):
        super().__init__()
        self.source = source
        self.opts = opts
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        cap = None
        try:
            is_rtsp = isinstance(self.source, str) and self.source.lower().startswith("rtsp://")
            if is_rtsp:
                # Open/read timeout parameters must be provided when opening.
                # Some OpenCV builds ignore them; live validation is still required.
                try:
                    cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
                    ])
                except (TypeError, cv2.error):
                    cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(self.source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.error.emit("Could not open this camera source. Check the webcam index or RTSP URL.")
                return
            analyzer = FrameAnalyzer("Live: " + str(self.source), self.opts)
            frame_no, last_emit, fails = 0, 0.0, 0
            while not self._stop_requested:
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails >= 10:
                        if not self._stop_requested:
                            self.error.emit("The camera stream stopped responding. Check the camera/network, then start again.")
                        break
                    time.sleep(0.1)
                    continue
                fails = 0
                frame_no += 1
                annotated, in_frame, saved = analyzer.process(frame, frame_no)
                cv2.putText(annotated, f"In frame: {in_frame} | Counted: {analyzer.unique}",
                            (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 120, 210), 2, cv2.LINE_AA)
                if saved:
                    self.data_changed.emit()
                now = time.monotonic()
                if now - last_emit >= 0.04:
                    last_emit = now
                    self.frame_ready.emit(annotated, in_frame, analyzer.unique)
        except Exception as exc:
            if not self._stop_requested:
                self.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            self.stopped.emit()


class CameraProbeWorker(QThread):
    """Check that a saved camera opens and yields an actual frame, off the UI thread."""
    checked = Signal(bool, str)

    def __init__(self, source):
        super().__init__()
        self.source = source

    def run(self):
        cap = None
        try:
            rtsp = isinstance(self.source, str) and self.source.lower().startswith("rtsp://")
            if rtsp:
                try:
                    cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, [
                        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000,
                    ])
                except (TypeError, cv2.error):
                    cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.checked.emit(False, "Camera could not be opened. Check source and credentials.")
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                self.checked.emit(False, "Camera opened but did not return a readable frame.")
                return
            self.checked.emit(True, f"Camera test passed: {frame.shape[1]}x{frame.shape[0]} frame received.")
        except Exception as exc:
            self.checked.emit(False, f"Camera test failed: {type(exc).__name__}: {exc}")
        finally:
            if cap is not None:
                cap.release()


# --------------------------------------------------------------------------- UI helpers
class Page(QWidget):
    """Every page scrolls as a whole. Tables and previews keep a minimum height, so on a small
    screen the page scrolls, and tables also scroll inside themselves."""

    def __init__(self, title, subtitle=""):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        area = QScrollArea()
        self.scroll_area = area
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        host = QWidget()
        area.setWidget(host)
        outer.addWidget(area)
        self.body = QVBoxLayout(host)
        self.body.setContentsMargins(26, 22, 26, 24)
        self.body.setSpacing(14)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        self.body.addWidget(heading)
        if subtitle:
            sub = QLabel(subtitle)
            sub.setObjectName("subtitle")
            sub.setWordWrap(True)
            self.body.addWidget(sub)


def card(title, value, note=""):
    box = QFrame()
    box.setObjectName("metricCard")
    lay = QVBoxLayout(box)
    t = QLabel(title); t.setObjectName("metricTitle")
    v = QLabel(str(value)); v.setObjectName("metricValue")
    v.setMinimumHeight(42)
    v.setMinimumWidth(90)
    v.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
    v.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    lay.addWidget(t); lay.addWidget(v)
    if note:
        n = QLabel(note); n.setObjectName("subtitle"); lay.addWidget(n)
    return box, v


def button(text, kind="secondaryButton", slot=None):
    b = QPushButton(text)
    b.setObjectName(kind)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if slot:
        b.clicked.connect(lambda checked=False: slot())
    return b


def setup_table(table, min_height=360):
    table.setAlternatingRowColors(True)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.verticalHeader().setVisible(False)
    table.setWordWrap(False)
    table.setMinimumHeight(min_height)
    table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)


def fill_table(table, rows):
    """Replace table contents but keep the user's scroll position."""
    vbar = table.verticalScrollBar()
    pos = vbar.value()
    table.setUpdatesEnabled(False)
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.setItem(r, c, QTableWidgetItem("" if value is None else str(value)))
    table.setUpdatesEnabled(True)
    vbar.setValue(pos)


class ManualViolationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add violation manually")
        self.setMinimumWidth(420)
        form = QFormLayout(self)
        self.type_combo = QComboBox()
        for key, (label, _, _) in VIOLATIONS.items():
            self.type_combo.addItem(label, key)
        self.plate = QLineEdit(); self.plate.setPlaceholderText("KA01AB1234")
        self.location = QLineEdit(); self.location.setPlaceholderText("MG Road junction")
        self.image_path = QLineEdit(); self.image_path.setPlaceholderText("Optional evidence image")
        row = QHBoxLayout(); row.addWidget(self.image_path, 1); row.addWidget(button("Browse…", "secondaryButton", self._browse))
        form.addRow("Violation", self.type_combo)
        form.addRow("Vehicle number", self.plate)
        form.addRow("Location", self.location)
        form.addRow("Evidence image", row)
        box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept); box.rejected.connect(self.reject)
        form.addRow(box)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select evidence image", str(BASE_DIR),
                                              "Images (*.jpg *.jpeg *.png *.bmp);;All files (*)")
        if path:
            self.image_path.setText(path)


# --------------------------------------------------------------------------- main window
NAV = ["Home Dashboard", "Live Camera", "Camera Management", "Offline Analysis", "Vehicle Analytics",
       "Detection History", "Violation Review", "Reports", "Settings"]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nexora Vision AI | Traffic Intelligence")
        if LOGO_PATH.exists():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1440, screen.width() - 40), min(900, screen.height() - 60))
        self.setMinimumSize(900, 520)
        self.live_worker = None
        self.worker = None
        self.camera_probe_worker = None
        self.live_source = 0
        self.live_count = 0
        self.current_video = ""
        self.current_violation_id = None
        self._dirty = False
        self._ticks = 0
        self.build_ui()
        self.apply_style()
        self.load_settings()
        self.refresh_all(quiet=True)
        QShortcut(QKeySequence("F5"), self).activated.connect(
            lambda: self.safe_refresh(self.refresh_current_page, "This page"))
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(1000)
        self.refresh_timer.timeout.connect(self._tick)
        self.refresh_timer.start()

    @property
    def role(self):
        return self.role_combo.currentData()

    # ---- layout --------------------------------------------------------------
    def build_ui(self):
        root = QWidget()
        rl = QHBoxLayout(root); rl.setContentsMargins(0, 0, 0, 0); rl.setSpacing(0)
        sidebar = QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(245)
        sl = QVBoxLayout(sidebar); sl.setContentsMargins(18, 18, 18, 14); sl.setSpacing(6)
        if LOGO_PATH.exists():
            brand = QLabel(); brand.setAlignment(Qt.AlignmentFlag.AlignCenter); brand.setFixedHeight(120)
            brand.setPixmap(QPixmap(str(LOGO_PATH)).scaled(205, 115, Qt.AspectRatioMode.KeepAspectRatio,
                                                           Qt.TransformationMode.SmoothTransformation))
            sl.addWidget(brand)
        else:
            brand = QLabel("NEXORA\nVISION AI"); brand.setObjectName("brand"); sl.addWidget(brand)
        tag = QLabel("INTELLIGENT TRAFFIC SURVEILLANCE"); tag.setObjectName("brandSub"); sl.addWidget(tag)

        nav_host = QWidget(); nav_host.setObjectName("navHost")
        nav_l = QVBoxLayout(nav_host); nav_l.setContentsMargins(0, 8, 0, 0); nav_l.setSpacing(6)
        nav_area = QScrollArea(); nav_area.setWidgetResizable(True); nav_area.setFrameShape(QFrame.Shape.NoFrame)
        nav_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        nav_area.viewport().setObjectName("navViewport")
        nav_area.setWidget(nav_host)
        sl.addWidget(nav_area, 1)

        self.stack = QStackedWidget()
        self.pages, self.nav_buttons = {}, {}
        group = QButtonGroup(self); group.setExclusive(True)
        builders = {
            "Home Dashboard": self.build_home, "Live Camera": self.build_live,
            "Camera Management": self.build_cameras, "Offline Analysis": self.build_offline,
            "Vehicle Analytics": self.build_analytics, "Detection History": self.build_history,
            "Violation Review": self.build_review, "Reports": self.build_reports, "Settings": self.build_settings,
        }
        for name in NAV:
            btn = QPushButton(name); btn.setObjectName("navButton"); btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, n=name: self.navigate(n))
            group.addButton(btn); nav_l.addWidget(btn); self.nav_buttons[name] = btn
            page = builders[name](); self.pages[name] = page; self.stack.addWidget(page)
        nav_l.addStretch()
        self.nav_buttons["Home Dashboard"].setChecked(True)
        foot = QLabel("●  SYSTEM READY\nYOLO • OpenCV • SQLite"); foot.setObjectName("footerLabel"); sl.addWidget(foot)

        content = QWidget()
        cl = QVBoxLayout(content); cl.setContentsMargins(0, 0, 0, 0); cl.setSpacing(0)
        top = QFrame(); top.setObjectName("topbar")
        tl = QHBoxLayout(top); tl.setContentsMargins(25, 12, 25, 12)
        self.top_title = QLabel("Home Dashboard"); self.top_title.setObjectName("topTitle")
        self.role_combo = QComboBox()
        self.role_combo.addItem("Role: Operator", "operator")
        self.role_combo.addItem("Role: Approving authority", "authority")
        self.role_combo.currentIndexChanged.connect(lambda _=0: self.update_review_buttons())
        self.status = QLabel("●  Ready"); self.status.setObjectName("statusPill")
        tl.addWidget(self.top_title); tl.addStretch(); tl.addWidget(self.role_combo); tl.addWidget(self.status)
        cl.addWidget(top); cl.addWidget(self.stack, 1)
        rl.addWidget(sidebar); rl.addWidget(content, 1)
        self.setCentralWidget(root)

    # ---- pages ---------------------------------------------------------------
    def build_home(self):
        p = Page("Traffic Intelligence Dashboard", "Live overview of traffic detection and violation processing. Press F5 to refresh any page.")
        self.metric_labels = {}
        grid = QGridLayout(); grid.setSpacing(14)
        cards = [("total", "Approx. vehicle tracks"), ("cars", "Cars"), ("motorcycles", "Motorcycles"),
                 ("other", "Buses / Trucks / Bikes"), ("awaiting", "Violations awaiting action"),
                 ("approved", "Approved, challan pending"), ("issued", "Challans issued"), ("fines", "Fines raised")]
        for i, (key, title) in enumerate(cards):
            box, val = card(title, "0")
            self.metric_labels[key] = val
            grid.addWidget(box, i // 4, i % 4)
        p.body.addLayout(grid)
        row = QHBoxLayout()
        self.scope_combo = QComboBox(); self.scope_combo.addItems(["All completed runs", "Today", "Latest run"])
        self.scope_combo.currentIndexChanged.connect(lambda _=0: self.refresh_dashboard())
        self.dash_updated = QLabel("Not refreshed yet"); self.dash_updated.setObjectName("infoText")
        row.addWidget(QLabel("Vehicle counts for:")); row.addWidget(self.scope_combo)
        row.addWidget(button("Refresh dashboard", "primaryButton", lambda: self.safe_refresh(self.refresh_all, "Dashboard")))
        row.addWidget(self.dash_updated); row.addStretch()
        p.body.addLayout(row)
        info = QFrame(); info.setObjectName("panel"); il = QVBoxLayout(info)
        h = QLabel("Quick actions"); h.setObjectName("sectionTitle"); il.addWidget(h)
        for label, target in [("Open Live Camera", "Live Camera"), ("Analyze a Traffic Video", "Offline Analysis"),
                              ("Review Violations", "Violation Review"), ("View Detection History", "Detection History")]:
            il.addWidget(button(label, "primaryButton", lambda t=target: self.navigate(t)))
        sp = QFrame(); sp.setObjectName("panel"); spl = QVBoxLayout(sp)
        h2 = QLabel("System status"); h2.setObjectName("sectionTitle"); spl.addWidget(h2)
        self.model_status = QLabel(""); self.vmodel_status = QLabel("")
        spl.addWidget(self.model_status); spl.addWidget(self.vmodel_status); spl.addWidget(QLabel(f"Database: {DB_PATH.name}"))
        note = QLabel("Counts are approximate tracked moving objects, not independently verified passing traffic. "
                      "Older per-frame rows are excluded from dashboard totals. "
                      "All AI violations are unverified candidates until human review.")
        note.setObjectName("infoText"); note.setWordWrap(True); spl.addWidget(note)
        r2 = QHBoxLayout(); r2.addWidget(info, 1); r2.addWidget(sp, 1)
        p.body.addLayout(r2); p.body.addStretch()
        return p

    def build_live(self):
        p = Page("Live Camera Monitoring", "Connect a webcam (0, 1, …) or an IP camera RTSP URL. Press Stop and wait a moment before restarting.")
        ctl = QHBoxLayout()
        self.live_input = QLineEdit("0"); self.live_input.setPlaceholderText("0 for webcam or RTSP URL")
        self.start_live_btn = button("Start camera", "primaryButton", self.start_live_camera)
        self.stop_live_btn = button("Stop camera", "secondaryButton", self.stop_live_camera)
        self.stop_live_btn.setEnabled(False)
        ctl.addWidget(QLabel("Camera source:")); ctl.addWidget(self.live_input, 1)
        ctl.addWidget(self.start_live_btn); ctl.addWidget(self.stop_live_btn)
        p.body.addLayout(ctl)
        self.live_view = VideoPreview("Camera preview will appear here"); self.live_view.setObjectName("videoView")
        self.live_view.setMinimumHeight(420)
        p.body.addWidget(self.live_view, 1)
        bottom = QHBoxLayout()
        self.live_summary = QLabel("Vehicles counted this session: 0"); self.live_summary.setObjectName("infoText")
        self.capture_type = QComboBox()
        for key, (label, _, _) in VIOLATIONS.items():
            self.capture_type.addItem(label, key)
        bottom.addWidget(self.live_summary, 1)
        bottom.addWidget(QLabel("Flag current frame as:")); bottom.addWidget(self.capture_type)
        bottom.addWidget(button("Capture violation", "secondaryButton", self.capture_live_violation))
        p.body.addLayout(bottom)
        return p

    def build_cameras(self):
        p = Page("Camera Management", "Save camera connection details for later use.")
        box = QFrame(); box.setObjectName("panel"); form = QFormLayout(box)
        self.cam_name = QLineEdit(); self.cam_url = QLineEdit(); self.cam_location = QLineEdit()
        self.cam_url.setPlaceholderText("rtsp://username:password@camera-ip:554/stream")
        form.addRow("Camera name", self.cam_name); form.addRow("RTSP / stream URL", self.cam_url)
        form.addRow("Location", self.cam_location)
        btns = QHBoxLayout()
        btns.addWidget(button("Save camera", "primaryButton", self.save_camera))
        btns.addWidget(button("Use selected in Live Camera", "secondaryButton", self.use_selected_camera))
        btns.addWidget(button("Test selected camera", "secondaryButton", self.test_selected_camera))
        btns.addWidget(button("Delete selected", "secondaryButton", self.delete_selected_camera)); btns.addStretch()
        form.addRow(btns)
        p.body.addWidget(box)
        self.camera_table = QTableWidget(0, 4)
        self.camera_table.setHorizontalHeaderLabels(["ID", "Name", "Location", "Stream URL"])
        setup_table(self.camera_table, 300)
        p.body.addWidget(self.camera_table, 1)
        return p

    def build_offline(self):
        p = Page("Offline Traffic Video Analysis", "Choose a recorded road video. Vehicles and violation candidates are saved to the database.")
        ctl = QHBoxLayout()
        self.video_path_label = QLabel("No video selected"); self.video_path_label.setObjectName("infoText")
        self.analyze_btn = button("Start analysis", "primaryButton", self.start_analysis)
        self.stop_analysis_btn = button("Stop", "secondaryButton", self.stop_analysis)
        self.stop_analysis_btn.setEnabled(False)
        ctl.addWidget(button("Choose video", "secondaryButton", self.choose_video))
        ctl.addWidget(self.video_path_label, 1); ctl.addWidget(self.analyze_btn); ctl.addWidget(self.stop_analysis_btn)
        p.body.addLayout(ctl)
        self.offline_view = VideoPreview("Processed video preview will appear here"); self.offline_view.setObjectName("videoView")
        self.offline_view.setMinimumHeight(420)
        p.body.addWidget(self.offline_view, 1)
        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0); p.body.addWidget(self.progress_bar)
        self.analysis_status = QLabel("Ready to analyze"); self.analysis_status.setObjectName("infoText")
        self.analysis_status.setWordWrap(True)
        p.body.addWidget(self.analysis_status)
        return p

    def build_analytics(self):
        p = Page("Vehicle Analytics", "Summary calculated from the rows stored in the local database.")
        bar = QHBoxLayout()
        bar.addWidget(button("Refresh analytics", "primaryButton", lambda: self.safe_refresh(self.refresh_analytics, "Analytics")))
        self.analytics_updated = QLabel(""); self.analytics_updated.setObjectName("infoText")
        bar.addWidget(self.analytics_updated); bar.addStretch()
        p.body.addLayout(bar)
        self.analytics_text = QTextEdit(); self.analytics_text.setReadOnly(True); self.analytics_text.setObjectName("textPanel")
        self.analytics_text.setMinimumHeight(460)
        p.body.addWidget(self.analytics_text, 1)
        return p

    def build_history(self):
        p = Page("Detection History", "Each counted vehicle is stored once, with the run that counted it.")
        bar = QHBoxLayout()
        self.history_filter = QComboBox(); self.history_filter.addItems(["All vehicles", "Bicycle", "Car", "Motorcycle", "Bus", "Truck"])
        self.history_scope = QComboBox(); self.history_scope.addItems(["All rows", "Latest run only", "Legacy rows (old per-frame data)"])
        self.history_search = QLineEdit(); self.history_search.setPlaceholderText("Filter by source…")
        for w in (self.history_filter, self.history_scope):
            w.currentIndexChanged.connect(lambda _=0: self.refresh_history())
        self.history_search.textChanged.connect(lambda _=None: self.refresh_history())
        bar.addWidget(QLabel("Vehicle:")); bar.addWidget(self.history_filter)
        bar.addWidget(QLabel("Show:")); bar.addWidget(self.history_scope)
        bar.addWidget(self.history_search, 1)
        p.body.addLayout(bar)
        bar2 = QHBoxLayout()
        bar2.addWidget(button("Refresh", "primaryButton", lambda: self.safe_refresh(self.refresh_history, "Detection history")))
        bar2.addWidget(button("Export CSV", "secondaryButton", self.export_detections_csv))
        bar2.addStretch()
        bar2.addWidget(button("Delete legacy rows", "secondaryButton", self.delete_legacy_rows))
        bar2.addWidget(button("Clear all history", "dangerButton", self.clear_history))
        p.body.addLayout(bar2)
        self.history_info = QLabel(""); self.history_info.setObjectName("infoText"); self.history_info.setWordWrap(True)
        p.body.addWidget(self.history_info)
        self.history_table = QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels(["ID", "Date / time", "Source", "Run", "Track", "Vehicle", "Confidence", "Frame"])
        setup_table(self.history_table, 440)
        p.body.addWidget(self.history_table, 1)
        return p

    def build_review(self):
        p = Page("Violation Review & e-Challan",
                 "Operator review → authority approval → e-Challan → payment. Every step is written to the audit trail.")
        bar = QHBoxLayout()
        self.rv_filter = QComboBox(); self.rv_filter.addItem("All cases", "all")
        for key, label in STATUS.items():
            self.rv_filter.addItem(label, key)
        self.rv_filter.currentIndexChanged.connect(lambda _=0: self.refresh_violations())
        bar.addWidget(QLabel("Show:")); bar.addWidget(self.rv_filter); bar.addStretch()
        bar.addWidget(button("Add violation manually", "secondaryButton", self.add_manual_violation))
        bar.addWidget(button("Refresh", "primaryButton", lambda: self.safe_refresh(self.refresh_violations, "Violation list")))
        p.body.addLayout(bar)
        # Keep the two work areas independent: the violation list and the e-Challan/details
        # panel each get their own scroll container, rather than scrolling the whole page.
        p.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        p.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setMinimumHeight(520)
        split.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.rv_table = QTableWidget(0, 5)
        self.rv_table.setHorizontalHeaderLabels(["ID", "Date / time", "Violation", "Plate", "Status"])
        setup_table(self.rv_table, 320)
        self.rv_table.setMinimumHeight(420)
        self.rv_table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.rv_table.itemSelectionChanged.connect(self.on_violation_selected)
        list_scroll = QScrollArea()
        list_scroll.setWidgetResizable(True)
        list_scroll.setFrameShape(QFrame.Shape.NoFrame)
        list_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        list_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        list_scroll.setWidget(self.rv_table)
        split.addWidget(list_scroll)

        detail = QFrame(); detail.setObjectName("panel"); dl = QVBoxLayout(detail)
        self.rv_image = VideoPreview("Select a case to see its evidence"); self.rv_image.setObjectName("videoView")
        self.rv_image.setMinimumHeight(200)
        self.rv_info = QLabel(""); self.rv_info.setWordWrap(True); self.rv_info.setTextFormat(Qt.TextFormat.RichText)
        form = QFormLayout()
        self.rv_plate = QLineEdit(); self.rv_plate.setPlaceholderText("KA01AB1234 (correct if the plate is wrong)")
        self.rv_location = QLineEdit(); self.rv_note = QLineEdit()
        self.rv_note.setPlaceholderText("Required for reject, re-evaluation and enhanced-review types")
        form.addRow("Vehicle number", self.rv_plate); form.addRow("Location", self.rv_location); form.addRow("Note", self.rv_note)
        self.rv_buttons = {}
        brow = QGridLayout()
        for i, (key, (label, _, _, _)) in enumerate(ACTIONS.items()):
            kind = "dangerButton" if key == "reject" else ("secondaryButton" if key == "reeval" else "successButton")
            b = button(label, kind, lambda k=key: self.do_action(k))
            self.rv_buttons[key] = b
            brow.addWidget(b, i // 3, i % 3)
        self.rv_pdf_btn = button("Export challan PDF", "primaryButton", self.export_challan_pdf)
        self.rv_audit = QTextEdit(); self.rv_audit.setReadOnly(True); self.rv_audit.setMinimumHeight(110)
        al = QLabel("Audit trail"); al.setObjectName("sectionTitle")
        dl.addWidget(self.rv_image, 1); dl.addWidget(self.rv_info); dl.addLayout(form); dl.addLayout(brow)
        dl.addWidget(self.rv_pdf_btn); dl.addWidget(al); dl.addWidget(self.rv_audit)
        detail_scroll = QScrollArea()
        detail_scroll.setWidgetResizable(True)
        detail_scroll.setFrameShape(QFrame.Shape.NoFrame)
        detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        detail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        detail_scroll.setWidget(detail)
        split.addWidget(detail_scroll)
        split.setStretchFactor(0, 3); split.setStretchFactor(1, 2)
        split.setSizes([620, 500])
        p.body.addWidget(split, 1)
        self.update_review_buttons()
        return p

    def build_reports(self):
        p = Page("Reports & Export", "Generate CSV reports from your stored data.")
        box = QFrame(); box.setObjectName("panel"); lay = QVBoxLayout(box)
        h = QLabel("Reports"); h.setObjectName("sectionTitle"); lay.addWidget(h)
        lay.addWidget(QLabel("Detections: date/time, source, run, vehicle class, confidence, frame, processed video path."))
        lay.addWidget(button("Export all detections to CSV", "primaryButton", self.export_detections_csv))
        lay.addWidget(QLabel("Violations: type, plate, location, status, challan number, fine and dates."))
        lay.addWidget(button("Export all violations to CSV", "primaryButton", self.export_violations_csv))
        p.body.addWidget(box); p.body.addStretch()
        return p

    def build_settings(self):
        p = Page("Settings", "Settings are saved in the database and used the next time a camera or video starts.")
        box = QFrame(); box.setObjectName("panel"); form = QFormLayout(box)
        self.model_combo = QComboBox()
        files = sorted(f.name for f in BASE_DIR.glob("*.pt") if f.name.lower() != VIOLATION_MODEL_PATH.name)
        self.model_combo.addItems(files or [MODEL_PATH.name])
        self.conf_spin = QDoubleSpinBox(); self.conf_spin.setRange(0.10, 0.90); self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setDecimals(2); self.conf_spin.setValue(0.35)
        self.imgsz_spin = QSpinBox(); self.imgsz_spin.setRange(320, 1280); self.imgsz_spin.setSingleStep(32); self.imgsz_spin.setValue(640)
        self.line_spin = QSpinBox(); self.line_spin.setRange(0, 95); self.line_spin.setSuffix(" %"); self.line_spin.setSpecialValueText("Off")
        self.movement_spin = QSpinBox(); self.movement_spin.setRange(0, 300); self.movement_spin.setValue(25); self.movement_spin.setSuffix(" px")
        self.rules_check = QCheckBox("Detect triple riding and wrong-way candidates (also tracks people)")
        self.legal_combo = QComboBox(); self.legal_combo.addItems(["Off", "Down", "Up", "Left", "Right"])
        self.prefix_edit = QLineEdit("KA"); self.prefix_edit.setMaxLength(4)
        form.addRow("YOLO model (.pt in project folder)", self.model_combo)
        form.addRow("Detection confidence threshold", self.conf_spin)
        form.addRow("Image size (larger = better for small/far vehicles, slower)", self.imgsz_spin)
        form.addRow("Counting line, % from top of frame", self.line_spin)
        form.addRow("Minimum movement before counting (pixels)", self.movement_spin)
        form.addRow("Experimental violation candidates", self.rules_check)
        form.addRow("Legal travel direction in the image", self.legal_combo)
        form.addRow("Challan number prefix", self.prefix_edit)
        self.fine_spins = {}
        for key, (label, default, _) in VIOLATIONS.items():
            sp = QSpinBox(); sp.setRange(0, 1000000); sp.setValue(default); sp.setPrefix("₹ ")
            self.fine_spins[key] = sp
            form.addRow(f"Fine: {label}", sp)
        row = QHBoxLayout()
        row.addWidget(button("Save settings", "primaryButton", self.save_settings))
        row.addWidget(button("Check violations.pt classes", "secondaryButton", self.check_violation_model)); row.addStretch()
        form.addRow(row)
        p.body.addWidget(box)
        vm = "found" if VIOLATION_MODEL_PATH.exists() else "not found (optional)"
        for text in (f"Custom violation model: {VIOLATION_MODEL_PATH} - {vm}", f"Database: {DB_PATH}",
                     f"Evidence images: {EVIDENCE_DIR}", f"Processed videos: {OUTPUT_DIR}"):
            lab = QLabel(text); lab.setWordWrap(True); lab.setObjectName("infoText"); p.body.addWidget(lab)
        note = QLabel("This is a LOCAL DEMO. Switching the role dropdown is not authentication; do not issue real challans. "
                      "Helmet, seat-belt, police-light and person-on-top detection need a custom model saved as violations.pt. "
                      "The standard yolo11n.pt cannot see helmets. See train_violations.py. Fines are starting values only: "
                      "set them from your state's current notification.")
        note.setObjectName("infoText"); note.setWordWrap(True); p.body.addWidget(note); p.body.addStretch()
        return p

    # ---- style ---------------------------------------------------------------
    def apply_style(self):
        self.setStyleSheet("""
        QWidget { background: #f4f8ff; color: #20334d; font-family: 'Segoe UI'; font-size: 14px; }
        QLabel { background: transparent; }
        #sidebar, #navHost, #navViewport { background: #ffffff; }
        #sidebar { border-right: 1px solid #dce7f5; }
        #brand { color: #1769aa; font-size: 25px; font-weight: 800; letter-spacing: 1px; }
        #brandSub { color: #7890ab; font-size: 9px; font-weight: 700; }
        #navButton { text-align: left; padding: 11px 13px; color: #47627f; background: transparent; border: none; border-radius: 8px; font-weight: 600; }
        #navButton:hover { background: #eaf4ff; color: #1769aa; }
        #navButton:checked { background: #dcecff; color: #1769aa; }
        #footerLabel { color: #438d75; font-size: 12px; padding: 8px 3px; background: #ffffff; }
        #topbar { background: #ffffff; border-bottom: 1px solid #dce7f5; }
        #topTitle { font-size: 18px; font-weight: 700; color: #1d3552; }
        #statusPill { color: #198754; background: #e8f8ef; padding: 8px 12px; border-radius: 14px; font-weight: 600; }
        #pageTitle { font-size: 26px; font-weight: 750; color: #173b63; }
        #subtitle { color: #7186a0; font-size: 13px; }
        #metricCard, #panel { background: #ffffff; border: 1px solid #dce7f5; border-radius: 13px; padding: 12px; }
        #metricTitle { color: #7186a0; font-size: 13px; }
        #metricValue { color: #1674bd; font-size: 28px; font-weight: 800; }
        #sectionTitle { font-size: 17px; font-weight: 700; color: #244765; }
        #infoText { color: #5e7895; }
        #videoView { background: #eaf3fc; color: #6d87a2; border: 2px dashed #bfd5eb; border-radius: 13px; }
        QPushButton#primaryButton { color: white; background: #2385d1; border: none; padding: 9px 16px; border-radius: 8px; font-weight: 700; }
        QPushButton#primaryButton:hover { background: #176fb6; }
        QPushButton#secondaryButton { color: #23699e; background: #e5f2ff; border: 1px solid #c7e1f8; padding: 8px 14px; border-radius: 8px; font-weight: 600; }
        QPushButton#successButton { color: white; background: #198754; border: none; padding: 9px 12px; border-radius: 8px; font-weight: 700; }
        QPushButton#dangerButton { color: white; background: #c0392b; border: none; padding: 9px 12px; border-radius: 8px; font-weight: 700; }
        QPushButton#primaryButton:disabled, QPushButton#secondaryButton:disabled,
        QPushButton#successButton:disabled, QPushButton#dangerButton:disabled { color: #9aa9ba; background: #e6ebf2; border: 1px solid #d6dee8; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QTableWidget { background: #ffffff; border: 1px solid #d2e0ef; border-radius: 7px; padding: 6px; selection-background-color: #cfe8ff; selection-color: #14304d; }
        QHeaderView::section { background: #e9f3fd; color: #355774; border: none; padding: 8px; font-weight: 700; }
        QProgressBar { background: #e3edf8; border: none; border-radius: 5px; text-align: center; height: 16px; color: #23496c; }
        QProgressBar::chunk { background: #3997df; border-radius: 5px; }
        QSplitter::handle { background: #dce7f5; width: 5px; }
        QScrollBar:vertical { background: #eef3fa; width: 13px; margin: 0; }
        QScrollBar::handle:vertical { background: #b3c9e0; border-radius: 6px; min-height: 36px; }
        QScrollBar:horizontal { background: #eef3fa; height: 13px; margin: 0; }
        QScrollBar::handle:horizontal { background: #b3c9e0; border-radius: 6px; min-width: 36px; }
        QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
        #textPanel { padding: 14px; }
        """)

    # ---- feedback / refresh --------------------------------------------------
    def notify(self, text, ms=4000):
        self.status.setText("●  " + text)
        QTimer.singleShot(ms, self._restore_status)

    def _restore_status(self):
        busy = any(w is not None and w.isRunning() for w in (self.live_worker, self.worker))
        self.status.setText("●  Processing" if busy else "●  Ready")

    def safe_refresh(self, fn, label="Data"):
        try:
            fn()
            self.refresh_dashboard()
            self.notify(f"{label} refreshed at {datetime.now():%H:%M:%S}")
        except Exception as exc:
            _excepthook(type(exc), exc, exc.__traceback__)
            QMessageBox.warning(self, "Refresh failed", f"{label} could not be refreshed:\n{exc}")

    def navigate(self, name):
        self.top_title.setText(name)
        self.nav_buttons[name].setChecked(True)
        self.stack.setCurrentWidget(self.pages[name])
        self.refresh_current_page()

    def current_page_name(self):
        cur = self.stack.currentWidget()
        return next((n for n, pg in self.pages.items() if pg is cur), "")

    def refresh_current_page(self):
        name = self.current_page_name()
        self.refresh_dashboard()
        if name == "Detection History": self.refresh_history()
        elif name == "Vehicle Analytics": self.refresh_analytics()
        elif name == "Camera Management": self.refresh_cameras()
        elif name == "Violation Review": self.refresh_violations()

    def mark_dirty(self):
        self._dirty = True

    def _tick(self):
        """Refresh data labels periodically without rebuilding layouts or changing page size."""
        self._ticks += 1
        try:
            if self._dirty:
                self._dirty = False
                self.refresh_current_page()
            # Violation status can change from the review/e-Challan actions, not only from
            # camera workers. Keep the Home Dashboard's Awaiting Actions counter current.
            # Updating labels only avoids the old dashboard zoom/flicker problem.
            if self._ticks % 3 == 0:
                self.refresh_dashboard()
                if self.current_page_name() == "Violation Review":
                    self.refresh_violations()
        except Exception as exc:
            _excepthook(type(exc), exc, exc.__traceback__)

    def refresh_all(self, quiet=False):
        self.refresh_dashboard(); self.refresh_history(); self.refresh_cameras()
        self.refresh_analytics(); self.refresh_violations()
        if not quiet:
            self.notify(f"Data refreshed {datetime.now():%H:%M:%S}")

    def scope_where(self):
        s = self.scope_combo.currentText()
        if s == "Today":
            return " WHERE run_id IS NOT NULL AND event_time >= ?", [datetime.now().strftime("%Y-%m-%d 00:00:00")]
        if s == "Latest run":
            return " WHERE run_id = (SELECT MAX(run_id) FROM detections)", []
        # Legacy per-frame rows have NULL run_id and inflate vehicle counts.
        return " WHERE run_id IS NOT NULL", []

    def refresh_dashboard(self):
        where, params = self.scope_where()
        counts = dict(db.query(f"SELECT vehicle_type, COUNT(*) FROM detections{where} GROUP BY vehicle_type", params))
        m = self.metric_labels
        m["total"].setText(str(sum(counts.values())))
        m["cars"].setText(str(counts.get("Car", 0)))
        m["motorcycles"].setText(str(counts.get("Motorcycle", 0)))
        m["other"].setText(str(sum(counts.get(x, 0) for x in ("Bicycle", "Bus", "Truck"))))
        m["awaiting"].setText(str(db.scalar("SELECT COUNT(*) FROM violations WHERE status IN ('pending','reeval','accepted')")))
        m["approved"].setText(str(db.scalar("SELECT COUNT(*) FROM violations WHERE status='approved'")))
        m["issued"].setText(str(db.scalar("SELECT COUNT(*) FROM violations WHERE challan_no IS NOT NULL")))
        fines = int(db.scalar("SELECT SUM(fine_amount) FROM violations WHERE challan_no IS NOT NULL"))
        m["fines"].setText(f"₹{fines:,}")
        self.dash_updated.setText(f"Last updated {datetime.now():%H:%M:%S}")
        sel = self.model_combo.currentText()
        self.model_status.setText(f"Model: {sel} " + ("(found)" if (BASE_DIR / sel).exists() else "(MISSING)"))
        self.vmodel_status.setText("Violation model: " + ("violations.pt loaded when a run starts" if VIOLATION_MODEL_PATH.exists()
                                                           else "not installed (helmet / seat belt detection off)"))

    def refresh_cameras(self):
        fill_table(self.camera_table, db.query("SELECT id,name,location,stream_url FROM cameras ORDER BY id DESC"))

    def history_where(self):
        where, params = [], []
        if self.history_filter.currentText() != "All vehicles":
            where.append("vehicle_type=?"); params.append(self.history_filter.currentText())
        text = self.history_search.text().strip()
        if text:
            where.append("source LIKE ?"); params.append(f"%{text}%")
        scope = self.history_scope.currentIndex()
        if scope == 1:
            where.append("run_id = (SELECT MAX(run_id) FROM detections)")
        elif scope == 2:
            where.append("run_id IS NULL")
        return ((" WHERE " + " AND ".join(where)) if where else ""), params

    def refresh_history(self, *_):
        clause, params = self.history_where()
        total = db.scalar(f"SELECT COUNT(*) FROM detections{clause}", params)
        rows = db.query(f"SELECT id,event_time,source,run_id,track_id,vehicle_type,confidence,frame_number FROM detections{clause} "
                        f"ORDER BY id DESC LIMIT 5000", params)
        shown = [(r[0], r[1], r[2], r[3] if r[3] is not None else "legacy",
                  r[4] if r[4] is not None else "-", r[5],
                  f"{float(r[6]):.1%}" if r[6] is not None else "-",
                  r[7] if r[7] is not None else "-") for r in rows]
        fill_table(self.history_table, shown)
        per = db.query(f"SELECT vehicle_type,COUNT(*) FROM detections{clause} GROUP BY vehicle_type ORDER BY 2 DESC", params)
        legacy = db.scalar("SELECT COUNT(*) FROM detections WHERE run_id IS NULL")
        summary = "  ·  ".join(f"{t}: {n}" for t, n in per) or "no rows"
        warn = f"  |  {legacy} legacy per-frame rows from the old version inflate totals; use 'Delete legacy rows'." if legacy else ""
        self.history_info.setText(f"Showing {len(shown)} of {total} rows  |  {summary}  |  refreshed {datetime.now():%H:%M:%S}{warn}")

    def _confirm(self, title, text):
        return QMessageBox.question(self, title, text) == QMessageBox.StandardButton.Yes

    def _backup_db(self):
        path = BASE_DIR / f"traffic_history_backup_{datetime.now():%Y%m%d_%H%M%S}.db"
        db.backup(path)
        return path

    def delete_legacy_rows(self):
        n = db.scalar("SELECT COUNT(*) FROM detections WHERE run_id IS NULL")
        if not n:
            QMessageBox.information(self, "Nothing to delete", "There are no legacy rows.")
            return
        if self._confirm("Delete legacy rows", f"Delete {n} old per-frame rows? A backup copy is saved first."):
            backup = self._backup_db()
            db.execute("DELETE FROM detections WHERE run_id IS NULL")
            self.refresh_all(quiet=True)
            self.notify(f"Deleted {n} legacy rows. Backup: {backup.name}", 8000)

    def clear_history(self):
        n = db.scalar("SELECT COUNT(*) FROM detections")
        if self._confirm("Clear all history", f"Delete all {n} detection rows? A backup copy is saved first."):
            backup = self._backup_db()
            db.execute("DELETE FROM detections")
            self.refresh_all(quiet=True)
            self.notify(f"History cleared. Backup: {backup.name}", 8000)

    def refresh_analytics(self, *_):
        total = db.scalar("SELECT COUNT(*) FROM detections WHERE run_id IS NOT NULL")
        rows = db.query("SELECT vehicle_type,COUNT(*),AVG(confidence) FROM detections "
                        "WHERE run_id IS NOT NULL GROUP BY vehicle_type ORDER BY 2 DESC")
        sources = db.query("SELECT source,COUNT(*) FROM detections WHERE run_id IS NOT NULL "
                           "GROUP BY source ORDER BY 2 DESC LIMIT 10")
        vtypes = db.query("SELECT violation_type,COUNT(*) FROM violations GROUP BY violation_type ORDER BY 2 DESC")
        vstat = db.query("SELECT status,COUNT(*) FROM violations GROUP BY status ORDER BY 2 DESC")
        last = db.scalar("SELECT MAX(run_id) FROM detections", default=None)
        lines = []
        if last is not None:
            src = db.scalar("SELECT source FROM detections WHERE run_id=? LIMIT 1", (last,), "")
            lrows = db.query("SELECT vehicle_type,COUNT(*) FROM detections WHERE run_id=? GROUP BY vehicle_type ORDER BY 2 DESC", (last,))
            lines += [f"LATEST RUN #{last}: {src}", f"Vehicles counted: {sum(n for _, n in lrows)}"]
            lines += [f"• {t}: {n}" for t, n in lrows] + [""]
        lines += [f"ALL COUNTED TRACKS (excluding legacy): {total}", "", "BY VEHICLE CLASS"]
        lines += [f"• {t}: {n} | average confidence {a:.1%}" if a is not None else f"• {t}: {n}" for t, n, a in rows] \
            or ["• No detections saved yet. Start a camera or analyze a video."]
        legacy = db.scalar("SELECT COUNT(*) FROM detections WHERE run_id IS NULL")
        if legacy:
            lines += ["", f"NOTE: {legacy} older per-frame rows were excluded from these totals. "
                          "They remain in Detection History for review or backup/deletion."]
        lines += ["", "TOP SOURCES"] + ([f"• {s}: {n}" for s, n in sources] or ["• No sources recorded yet."])
        lines += ["", "VIOLATIONS BY TYPE"] + ([f"• {VIOLATIONS.get(t, (t,))[0]}: {n}" for t, n in vtypes] or ["• None recorded yet."])
        lines += ["", "VIOLATIONS BY STATUS"] + ([f"• {STATUS.get(s, s)}: {n}" for s, n in vstat] or ["• None recorded yet."])
        issued = db.scalar("SELECT COUNT(*) FROM violations WHERE challan_no IS NOT NULL")
        rejected = db.scalar("SELECT COUNT(*) FROM violations WHERE status='rejected'")
        decided = issued + rejected
        lines += ["", f"Operator rejection share (NOT model false-positive rate): {rejected / decided:.0%}" if decided
                  else "Operator rejection share: no reviewed/issued cases yet",
                  "", "Number-plate OCR (ANPR) is not included. Plates are entered or corrected during review."]
        vbar = self.analytics_text.verticalScrollBar(); pos = vbar.value()
        self.analytics_text.setPlainText("\n".join(lines))
        vbar.setValue(pos)
        self.analytics_updated.setText(f"Last refreshed {datetime.now():%H:%M:%S}")

    # ---- settings ------------------------------------------------------------
    def load_settings(self):
        idx = self.model_combo.findText(db.get_setting("model", MODEL_PATH.name))
        self.model_combo.setCurrentIndex(max(0, idx))
        self.conf_spin.setValue(float(db.get_setting("conf", "0.35")))
        self.imgsz_spin.setValue(int(float(db.get_setting("imgsz", "640"))))
        self.line_spin.setValue(int(float(db.get_setting("line", "0"))))
        self.movement_spin.setValue(int(float(db.get_setting("min_count_move", "25"))))
        self.rules_check.setChecked(db.get_setting("rules", "1") == "1")
        self.legal_combo.setCurrentIndex(max(0, self.legal_combo.findText(db.get_setting("legal", "Off"))))
        self.prefix_edit.setText(db.get_setting("prefix", "KA"))
        for key, sp in self.fine_spins.items():
            sp.setValue(int(float(db.get_setting(f"fine_{key}", VIOLATIONS[key][1]))))

    def save_settings(self):
        db.set_setting("model", self.model_combo.currentText())
        db.set_setting("conf", self.conf_spin.value())
        db.set_setting("imgsz", self.imgsz_spin.value())
        db.set_setting("line", self.line_spin.value())
        db.set_setting("min_count_move", self.movement_spin.value())
        db.set_setting("rules", "1" if self.rules_check.isChecked() else "0")
        db.set_setting("legal", self.legal_combo.currentText())
        db.set_setting("prefix", self.prefix_edit.text().strip().upper() or "KA")
        for key, sp in self.fine_spins.items():
            db.set_setting(f"fine_{key}", sp.value())
        self.refresh_dashboard()
        self.notify("Settings saved")

    def detection_options(self):
        return {"conf": self.conf_spin.value(), "rules": self.rules_check.isChecked(),
                "legal": self.legal_combo.currentText(), "imgsz": self.imgsz_spin.value(),
                 "line": self.line_spin.value(), "min_count_move": self.movement_spin.value(),
                 "model_path": BASE_DIR / self.model_combo.currentText()}

    def check_violation_model(self):
        if not VIOLATION_MODEL_PATH.exists():
            QMessageBox.information(self, "violations.pt not found",
                                    f"Train a model with train_violations.py, or save your model as:\n{VIOLATION_MODEL_PATH}")
            return
        try:
            names = YOLO(str(VIOLATION_MODEL_PATH)).names
            vmap = build_violation_map(names)
        except Exception as exc:
            QMessageBox.warning(self, "Could not load violations.pt", str(exc))
            return
        lines = [f"{i}: {n}  →  " + (VIOLATIONS[vmap[i]][0] if vmap.get(i) else "ignored (not a violation class)")
                 for i, n in names.items()]
        QMessageBox.information(self, "violations.pt classes",
                                "\n".join(lines) + '\n\nTo map other class names, create violation_classes.json, '
                                                   'for example: {"head": "helmet", "NoSeatBelt": "belt"}')

    # ---- live camera ---------------------------------------------------------
    def start_live_camera(self):
        old = self.live_worker
        if old is not None and old.isRunning():
            old.wait(1500)
            if old.isRunning():
                QMessageBox.information(self, "Camera busy", "The camera is still stopping. Wait a moment and try again.")
                return
        opts = self.detection_options()
        if not opts["model_path"].exists():
            QMessageBox.critical(self, "Model missing", f"Model file not found:\n{opts['model_path']}")
            return
        text = self.live_input.text().strip() or "0"
        self.live_source = int(text) if text.isdigit() else text
        self.live_count = 0
        self.live_summary.setText("Starting camera and loading the model…")
        self.live_view.clear_frame("Connecting to camera…")
        self.live_worker = LiveCameraWorker(self.live_source, opts)
        self.live_worker.frame_ready.connect(self.receive_live_frame)
        self.live_worker.data_changed.connect(self.mark_dirty)
        self.live_worker.error.connect(self.live_camera_error)
        self.live_worker.stopped.connect(self.live_camera_stopped)
        self.start_live_btn.setEnabled(False)
        self.stop_live_btn.setEnabled(True)
        self.live_worker.start()
        self.status.setText("●  Camera starting")

    def receive_live_frame(self, annotated, vehicles_in_frame, counted):
        self.live_count = counted
        self.live_view.set_frame(annotated)
        self.live_summary.setText(f"Vehicles counted this session: {counted}  |  In current frame: {vehicles_in_frame}")
        self.status.setText("●  Camera running")

    def live_camera_error(self, message):
        self.status.setText("●  Camera issue")
        QMessageBox.warning(self, "Live camera", message)

    def live_camera_stopped(self):
        """The thread object is kept alive on purpose; it is only replaced after it has finished."""
        self.start_live_btn.setEnabled(True)
        self.stop_live_btn.setEnabled(False)
        self.live_view.clear_frame("Camera stopped")
        self.live_summary.setText(f"Camera stopped. Vehicles counted this session: {self.live_count}")
        self.status.setText("●  Ready")
        self.mark_dirty()

    def stop_live_camera(self):
        w = self.live_worker
        if w is not None and w.isRunning():
            w.stop()
            self.stop_live_btn.setEnabled(False)
            self.status.setText("●  Stopping camera…")
            self.live_summary.setText("Stopping… the dashboard stays open.")

    def capture_live_violation(self):
        frame = self.live_view.current_frame()
        if frame is None:
            QMessageBox.information(self, "No frame", "Start the camera first, then capture a violation.")
            return
        save_violation(self.capture_type.currentData(), frame, f"Live (manual): {self.live_source}",
                       note="Flagged manually by operator.", actor=self.role)
        self.mark_dirty()
        self.notify("Violation captured. See Violation Review")

    # ---- offline analysis ----------------------------------------------------
    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select traffic video", str(BASE_DIR),
                                              "Video files (*.mp4 *.avi *.mov *.mkv *.mpeg *.mpg);;All files (*)")
        if path:
            self.current_video = path
            self.video_path_label.setText(path)

    def start_analysis(self):
        if not self.current_video:
            QMessageBox.information(self, "Choose a video", "Select a traffic video first.")
            return
        opts = self.detection_options()
        if not opts["model_path"].exists():
            QMessageBox.critical(self, "Model missing", f"Model file not found:\n{opts['model_path']}")
            return
        old = self.worker
        if old is not None and old.isRunning():
            QMessageBox.information(self, "Analysis running", "An analysis is already running or still stopping.")
            return
        self.worker = VideoWorker(self.current_video, opts)
        self.worker.frame_ready.connect(self.offline_view.set_frame)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.data_changed.connect(self.mark_dirty)
        self.worker.error.connect(self.show_analysis_error)
        self.worker.finished_info.connect(self.analysis_finished)
        self.analyze_btn.setEnabled(False)
        self.stop_analysis_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        extra = "" if VIOLATION_MODEL_PATH.exists() else " (violations.pt not installed: helmet / seat-belt detection is off)"
        self.analysis_status.setText("Loading model and analyzing video…" + extra)
        self.status.setText("●  Analyzing video")
        self.worker.start()

    def show_analysis_error(self, message):
        self.analysis_status.setText("Analysis error. See the message for details.")
        QMessageBox.critical(self, "Analysis error", message)

    def stop_analysis(self):
        w = self.worker
        if w is not None and w.isRunning():
            w.stop()
            self.stop_analysis_btn.setEnabled(False)
            self.analysis_status.setText("Stopping after the current frame. The dashboard stays open.")
        else:
            self.analysis_status.setText("No analysis is currently running.")

    def analysis_finished(self, out_path, count, stopped, succeeded):
        self.analyze_btn.setEnabled(True)
        self.stop_analysis_btn.setEnabled(False)
        self.status.setText("●  Ready")
        if stopped:
            self.analysis_status.setText(f"Stopped by user. Tracks counted: {count}. Partial video: {out_path}")
        elif not succeeded:
            self.analysis_status.setText("Analysis failed. See the error message/log. Do not trust partial counts.")
        else:
            self.analysis_status.setText(f"Finished. Approximate tracked vehicles: {count}. Processed video: {out_path}")
            self.progress_bar.setValue(100)
        self.refresh_all(quiet=True)
        if succeeded and out_path:
            QMessageBox.information(self, "Analysis complete",
                                    f"Video analysis finished.\n\nTracked vehicles: {count}\n\nProcessed video:\n{out_path}")

    # ---- cameras -------------------------------------------------------------
    def save_camera(self):
        name, url, loc = self.cam_name.text().strip(), self.cam_url.text().strip(), self.cam_location.text().strip()
        if not name or not url:
            QMessageBox.warning(self, "Missing details", "Enter at least a camera name and stream URL.")
            return
        try:
            db.execute("INSERT INTO cameras(name,stream_url,location,created_at) VALUES(?,?,?,?)", (name, url, loc, now_str()))
            self.cam_name.clear(); self.cam_url.clear(); self.cam_location.clear()
            self.refresh_cameras()
        except sqlite3.IntegrityError:
            QMessageBox.warning(self, "Duplicate name", "A camera with that name already exists.")

    def _selected_camera(self):
        items = self.camera_table.selectedItems()
        if not items:
            QMessageBox.information(self, "Select a camera", "Select a camera row first.")
            return None
        r = items[0].row()
        return int(self.camera_table.item(r, 0).text()), self.camera_table.item(r, 3).text()

    def use_selected_camera(self):
        sel = self._selected_camera()
        if sel:
            self.live_input.setText(sel[1])
            self.navigate("Live Camera")

    def test_selected_camera(self):
        sel = self._selected_camera()
        if not sel:
            return
        if self.camera_probe_worker is not None and self.camera_probe_worker.isRunning():
            QMessageBox.information(self, "Camera check", "Another camera test is still running.")
            return
        source = int(sel[1]) if sel[1].isdigit() else sel[1]
        self.camera_probe_worker = CameraProbeWorker(source)
        self.camera_probe_worker.checked.connect(self.camera_test_finished)
        self.status.setText("●  Testing camera")
        self.camera_probe_worker.start()

    def camera_test_finished(self, worked, message):
        self.status.setText("●  Ready")
        if worked:
            QMessageBox.information(self, "Camera test", message)
        else:
            QMessageBox.warning(self, "Camera test", message)

    def delete_selected_camera(self):
        sel = self._selected_camera()
        if sel and self._confirm("Delete camera", "Delete the selected camera?"):
            db.execute("DELETE FROM cameras WHERE id=?", (sel[0],))
            self.refresh_cameras()

    # ---- violation review ----------------------------------------------------
    def add_manual_violation(self):
        dlg = ManualViolationDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        plate = norm_plate(dlg.plate.text())
        if plate and not PLATE_RE.fullmatch(plate):
            QMessageBox.warning(self, "Check vehicle number",
                                "Enter a valid vehicle number such as KA01AB1234, or leave it blank for later review.")
            return
        path = dlg.image_path.text().strip()
        frame = read_image(path) if path else None
        if path and frame is None:
            QMessageBox.warning(self, "Image could not be read",
                                "The selected evidence image could not be opened. Choose a valid JPG, PNG or BMP image.")
            return
        try:
            vid = save_violation(dlg.type_combo.currentData(), frame, "Manual entry",
                                 plate=plate, location=dlg.location.text().strip(),
                                 note="Added manually; requires review.", actor=self.role)
            self.current_violation_id = vid
            self.refresh_violations()
            self.refresh_dashboard()
            self.notify("Manual violation saved")
        except Exception as exc:
            QMessageBox.critical(self, "Could not save violation", str(exc))

    def refresh_violations(self, *_):
        flt = self.rv_filter.currentData()
        q = "SELECT id,event_time,violation_type,plate,status FROM violations"
        params = []
        if flt != "all":
            q += " WHERE status=?"; params.append(flt)
        rows = db.query(q + " ORDER BY id DESC LIMIT 3000", params)
        shown = [(r[0], r[1], VIOLATIONS.get(r[2], (r[2],))[0], r[3] or "—", STATUS.get(r[4], r[4])) for r in rows]
        vbar = self.rv_table.verticalScrollBar(); pos = vbar.value()
        self.rv_table.blockSignals(True)
        # Clear stale table items first, then load the latest rows from SQLite.
        # Refresh reloads the view; it does not delete database records.
        self.rv_table.clearContents()
        self.rv_table.setRowCount(0)
        fill_table(self.rv_table, shown)
        target = next((i for i, r in enumerate(shown) if r[0] == self.current_violation_id), None)
        if target is not None:
            self.rv_table.selectRow(target)
        else:
            self.rv_table.clearSelection()
            self.current_violation_id = None
        vbar.setValue(min(pos, self.rv_table.verticalScrollBar().maximum()))
        self.rv_table.blockSignals(False)
        self.show_violation()
        self.refresh_dashboard()

    def on_violation_selected(self):
        items = self.rv_table.selectedItems()
        self.current_violation_id = int(self.rv_table.item(items[0].row(), 0).text()) if items else None
        self.show_violation()

    def _violation_row(self, vid):
        rows = db.query("SELECT id,event_time,violation_type,plate,camera,location,confidence,status,evidence_path,"
                        "challan_no,fine_amount,challan_date,note FROM violations WHERE id=?", (vid,))
        return rows[0] if rows else None

    def show_violation(self):
        row = self._violation_row(self.current_violation_id) if self.current_violation_id else None
        if row is None:
            self.rv_image.clear_frame("Select a case to see its evidence")
            self.rv_info.setText("")
            self.rv_plate.clear(); self.rv_location.clear(); self.rv_audit.clear()
            self.update_review_buttons()
            return
        vid, when, vtype, plate, cam, loc, conf, status, ev, cno, fine, cdate, note = row
        label, default_fine, enhanced = VIOLATIONS.get(vtype, (vtype, 0, False))
        frame = read_image(ev) if ev and Path(ev).exists() else None
        if frame is not None:
            self.rv_image.set_frame(frame)
        else:
            self.rv_image.clear_frame("No evidence image attached")
        info = [f"<b>{html.escape(label)}</b> &nbsp; <i>{html.escape(STATUS.get(status, status))}</i>",
                f"{html.escape(when)} · {html.escape(cam or '')}" + (f" · AI confidence {conf:.0f}%" if conf is not None else "")]
        if enhanced:
            info.append("<b>Enhanced review:</b> a written note is required before accepting this type.")
        fine_now = fine if fine is not None else int(float(db.get_setting(f"fine_{vtype}", default_fine)))
        info.append(f"Challan {html.escape(cno)}: ₹{fine_now:,}" if cno else f"Fine if issued: ₹{fine_now:,}")
        self.rv_info.setText("<br>".join(info))
        if not self.rv_plate.hasFocus():
            self.rv_plate.setText(plate or "")
        if not self.rv_location.hasFocus():
            self.rv_location.setText(loc or "")
        log = db.query("SELECT at,role,action,note FROM audit WHERE violation_id=? ORDER BY id", (vid,))
        self.rv_audit.setPlainText("\n".join(f"{a}  [{r}]  {act}" + (f" - {n}" if n else "") for a, r, act, n in log))
        self.update_review_buttons()

    def update_review_buttons(self):
        if not hasattr(self, "rv_buttons"):
            return
        row = self._violation_row(self.current_violation_id) if self.current_violation_id else None
        status = row[7] if row else None
        for key, b in self.rv_buttons.items():
            _, role, froms, _ = ACTIONS[key]
            b.setEnabled(bool(row) and status in froms and role in ("*", self.role))
        self.rv_pdf_btn.setEnabled(bool(row) and bool(row[9]))

    def do_action(self, key):
        row = self._violation_row(self.current_violation_id) if self.current_violation_id else None
        if row is None:
            return
        label, role, froms, new_status = ACTIONS[key]
        vid, status, vtype = row[0], row[7], row[2]
        if status not in froms or role not in ("*", self.role):
            QMessageBox.information(self, "Not allowed", "This action is not available for your role at this stage.")
            return
        plate = norm_plate(self.rv_plate.text())
        location = self.rv_location.text().strip()
        note = self.rv_note.text().strip()
        if key in ("reject", "reeval") and not note:
            QMessageBox.warning(self, "Note required", "Add a note explaining the decision.")
            return
        if key == "accept":
            if not PLATE_RE.fullmatch(plate):
                QMessageBox.warning(self, "Check the plate", "Enter a valid vehicle number, for example KA01AB1234, before accepting.")
                return
            if VIOLATIONS.get(vtype, ("", 0, False))[2] and not note:
                QMessageBox.warning(self, "Enhanced review", "Add a note confirming the checks you made for this violation type.")
                return
        fine = None
        cno = None
        if key == "issue":
            if not PLATE_RE.fullmatch(plate):
                QMessageBox.warning(self, "Check plate", "Enter and manually verify a registration number before the demo challan.")
                return
            fine = int(float(db.get_setting(f"fine_{vtype}", VIOLATIONS.get(vtype, ("", 0, False))[1])))
            cno = db.next_challan_no(db.get_setting("prefix", "KA"))
            note = (note + " " if note else "") + f"Demo challan {cno}, ₹{fine:,}"
        try:
            db.apply_review_action(vid, status, new_status, self.role, label, plate,
                                   location, note, challan_no=cno, fine_amount=fine)
        except (sqlite3.Error, ValueError) as exc:
            QMessageBox.warning(self, "Update failed", str(exc))
            self.refresh_violations()
            return
        self.rv_note.clear()
        self.refresh_violations()
        self.refresh_dashboard()
        self.mark_dirty()
        self.notify(f"{label} done — dashboard counters updated")

    def export_challan_pdf(self):
        row = self._violation_row(self.current_violation_id) if self.current_violation_id else None
        if row is None:
            QMessageBox.information(self, "Select a case", "Select a violation from the table first.")
            return
        if row[7] not in ("issued", "paid") or not row[9]:
            QMessageBox.information(self, "Challan not issued",
                                    "Complete Operator Accept → Authority Approve → Generate e-Challan before exporting the PDF.")
            return
        vid, when, vtype, plate, cam, loc, conf, status, ev, cno, fine, cdate, note = row
        path, _ = QFileDialog.getSaveFileName(self, "Save challan PDF", str(BASE_DIR / f"{cno}.pdf"), "PDF files (*.pdf)")
        if not path:
            return
        esc = html.escape
        img = f'<p><img src="{Path(ev).as_uri()}" width="520"></p>' if ev and Path(ev).exists() else ""
        doc = QTextDocument()
        doc.setHtml(f"""<h2>e-Challan {esc(cno)}</h2><p>Issued: {esc(cdate or '')}</p>
        <table border="1" cellspacing="0" cellpadding="6" width="100%">
        <tr><td><b>Violation</b></td><td>{esc(VIOLATIONS.get(vtype, (vtype,))[0])}</td></tr>
        <tr><td><b>Vehicle number</b></td><td>{esc(plate or '')}</td></tr>
        <tr><td><b>Place</b></td><td>{esc(loc or '')}</td></tr>
        <tr><td><b>Date and time of offence</b></td><td>{esc(when)}</td></tr>
        <tr><td><b>Camera</b></td><td>{esc(cam or '')}</td></tr>
        <tr><td><b>Fine</b></td><td>₹{int(fine or 0):,}</td></tr>
        <tr><td><b>Owner details</b></td><td>To be mapped through the VAHAN / SARATHI integration</td></tr></table>
        {img}<p style="font-size:9pt">Computer-generated notice. Verify the fine amount against the applicable state notification.</p>""")
        writer = QPdfWriter(path)
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        (getattr(doc, "print_", None) or getattr(doc, "print"))(writer)
        audit(vid, self.role, "Challan PDF exported", path)
        self.show_violation()
        QMessageBox.information(self, "Saved", f"Challan PDF saved to:\n{path}")

    # ---- exports -------------------------------------------------------------
    def _export_rows(self, title, default_name, headers, rows):
        path, _ = QFileDialog.getSaveFileName(self, title, str(BASE_DIR / default_name), "CSV files (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerows(rows)
            QMessageBox.information(self, "Export complete", f"Saved {len(rows)} rows to:\n{path}")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def export_detections_csv(self):
        rows = db.query("SELECT id,event_time,source,run_id,vehicle_type,confidence,frame_number,video_path FROM detections ORDER BY id DESC")
        self._export_rows("Export detection history", f"traffic_report_{datetime.now():%Y%m%d_%H%M%S}.csv",
                          ["ID", "Date/Time", "Source", "Run", "Vehicle", "Confidence", "Frame number", "Processed video path"], rows)

    def export_violations_csv(self):
        rows = db.query("SELECT id,event_time,violation_type,plate,camera,location,confidence,status,challan_no,fine_amount,"
                        "challan_date,note,evidence_path FROM violations ORDER BY id DESC")
        self._export_rows("Export violations", f"violations_{datetime.now():%Y%m%d_%H%M%S}.csv",
                          ["ID", "Event time", "Violation", "Plate", "Camera", "Location", "AI confidence", "Status",
                           "Challan no", "Fine", "Challan date", "Note", "Evidence path"], rows)

    # ---- shutdown ------------------------------------------------------------
    def closeEvent(self, event):
        live, off = self.live_worker, self.worker
        active_live = live is not None and live.isRunning()
        active_off = off is not None and off.isRunning()
        if active_live or active_off:
            if not self._confirm("Processing is active", "A camera or video analysis is still running. Stop it and close the dashboard?"):
                event.ignore()
                return
            if active_live: live.stop()
            if active_off: off.stop()
            if active_live: live.wait(10000)
            if active_off: off.wait(10000)
            if (active_live and live.isRunning()) or (active_off and off.isRunning()):
                QMessageBox.warning(self, "Please wait", "The model is finishing its current frame. Close again in a few seconds.")
                event.ignore()
                return
        probe = self.camera_probe_worker
        if probe is not None and probe.isRunning():
            if not probe.wait(5000):
                QMessageBox.warning(self, "Camera test is active",
                                    "Camera check has not finished. Close again when it completes.")
                event.ignore()
                return
        self.refresh_timer.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()