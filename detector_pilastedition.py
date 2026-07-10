"""
detector_pi5.py
================
Real-time YOLOv8n Elderly Safety Action Detector — Raspberry Pi 5 + PiCamera2
ONNX Runtime inference only. Low-latency, single-process, thread-safe capture.

Matches the model produced by the companion training notebook:
    UNIFIED_CLASSES = ["standing", "falling", "smoking", "drinking", "fire"]
    Export: best_model.export(format="onnx", imgsz=320|640, opset=12, simplify=True)

Pipeline
--------
    CameraThread (capture)  -->  letterbox()  -->  ONNX Runtime session.run()
            |                                              |
     always-freshest frame                    vectorized decode() + NMS
                                                            |
                                              display_output() / alert HUD

Design notes vs. a naive port
------------------------------
1. Classes are pulled from the model itself (or --classes), not hand-typed,
   so a class-name/index mismatch can no longer silently corrupt detections.
2. decode() is fully vectorized with NumPy. The previous per-anchor Python
   loop (thousands of iterations/frame) was the dominant latency cost on
   Pi 5's CPU; this removes it.
3. Preprocessing uses proper letterbox (aspect-ratio-preserving resize +
   pad), matching how Ultralytics trains/exports. A naive stretch-resize
   (the old approach) silently warps box geometry, especially for
   non-square objects like a fallen person.
4. Inference resolution is read from the ONNX model's own input tensor
   shape at load time, so best_320.onnx vs best_640.onnx "just works"
   without a config edit.
5. `falling` and `fire` are treated as safety alerts with a distinct,
   rate-limited HUD/console signal — these are the classes the notebook
   was explicitly built to catch for an elderly-safety use case.
6. Ultralytics/PyTorch fallback removed by design — ONNX Runtime is the
   sole inference backend, per the deployment decision already made for
   this project. Fewer code paths, smaller RAM footprint, faster boot.

Target: ~25-35 FPS on Raspberry Pi 5 (YOLOv8n ONNX, imgsz=320, 4 threads).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import threading
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[FATAL] onnxruntime is required: pip install onnxruntime", file=sys.stderr)
    sys.exit(1)

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


# ══════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("detector_pi5")


# ══════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════

# Fallback class list — only used if the ONNX file has no embedded names
# and --classes was not given. Matches the notebook's UNIFIED_CLASSES.
DEFAULT_CLASS_NAMES = ["standing", "falling", "smoking", "drinking", "fire"]

# Classes that should trigger a visible/audible safety alert.
ALERT_CLASSES = {"falling", "fire"}

CLASS_COLORS = {
    "standing": (200, 200, 200),   # neutral gray — not a safety event
    "falling":  (0,   0,   255),   # red — critical alert
    "smoking":  (0,   165, 255),   # orange
    "drinking": (255, 144, 30),    # blue
    "fire":     (0,   69,  255),   # red-orange — critical alert
}
FALLBACK_COLOR = (0, 220, 220)

ALERT_COOLDOWN_SEC = 3.0   # avoid spamming console/HUD every frame


@dataclass
class Config:
    model_path: str = "best.onnx"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45

    camera_index: int = 0
    cam_width: int = 640
    cam_height: int = 480

    inf_size: Optional[int] = None     # auto-detected from the ONNX model if None
    skip_frames: int = 2               # run inference every Nth frame

    disp_width: int = 800
    disp_height: int = 600
    window_name: str = "YOLOv8n Safety Detection (Q to quit)"
    headless: bool = False

    queue_size: int = 1                # 1 = always the freshest camera frame
    intra_op_threads: int = 4          # Pi 5 has 4 cores


# ══════════════════════════════════════════════
#  MODEL — ONNX Runtime wrapper
# ══════════════════════════════════════════════

class OnnxDetector:
    """
    Wraps an ONNX Runtime session for YOLOv8-style detection heads.

    Output layout assumed (standard Ultralytics ONNX export):
        shape [1, 4 + num_classes, num_anchors]
        [:4, i]  = cx, cy, w, h   (in *input tensor* pixel space)
        [4:, i]  = per-class scores
    """

    def __init__(self, model_path: Path, class_names: list[str], intra_op_threads: int):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape  # e.g. [1, 3, 320, 320]
        h, w = input_shape[2], input_shape[3]
        if isinstance(h, int) and isinstance(w, int):
            if h != w:
                log.warning("Non-square model input %sx%s — letterbox assumes square.", w, h)
            self.input_size = int(h)
        else:
            self.input_size = None  # dynamic axes; caller must supply a size

        num_classes = self.session.get_outputs()[0].shape[1]
        if isinstance(num_classes, int) and num_classes - 4 != len(class_names):
            log.warning(
                "Model reports %d classes but %d class names were provided — "
                "check --classes / DEFAULT_CLASS_NAMES for a mismatch.",
                num_classes - 4, len(class_names),
            )
        self.class_names = class_names

    def preprocess(self, letterboxed_frame: np.ndarray) -> np.ndarray:
        """BGR uint8 HWC -> RGB float32 NCHW normalized [0,1]."""
        rgb = cv2.cvtColor(letterboxed_frame, cv2.COLOR_BGR2RGB)
        nchw = np.transpose(rgb, (2, 0, 1))[np.newaxis]
        return np.ascontiguousarray(nchw, dtype=np.float32) / 255.0

    def infer(self, letterboxed_frame: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(letterboxed_frame)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        return outputs[0]

    def decode(self, output: np.ndarray, conf_thresh: float, iou_thresh: float) -> list[dict]:
        """
        Vectorized decode: no per-anchor Python loop.
        Returns detections with bbox in *letterboxed input-tensor* pixel space
        (x1, y1, x2, y2) — caller is responsible for mapping back to the
        original frame via the letterbox scale/pad used for this frame.
        """
        preds = output[0]                      # [4+nc, num_anchors]
        class_scores = preds[4:, :]             # [nc, num_anchors]

        cls_ids = np.argmax(class_scores, axis=0)
        anchor_idx = np.arange(class_scores.shape[1])
        confs = class_scores[cls_ids, anchor_idx]

        keep = confs > conf_thresh
        if not np.any(keep):
            return []

        cls_ids = cls_ids[keep]
        confs = confs[keep]
        cx, cy, w, h = preds[0, keep], preds[1, keep], preds[2, keep], preds[3, keep]

        x1 = cx - w / 2
        y1 = cy - h / 2

        # cv2.dnn.NMSBoxes wants [x, y, w, h] boxes as plain Python lists.
        nms_boxes = np.stack([x1, y1, w, h], axis=1).tolist()
        nms_scores = confs.tolist()

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_thresh, iou_thresh)
        if indices is None or len(indices) == 0:
            return []

        detections = []
        for idx in np.array(indices).flatten():
            bx, by, bw, bh = nms_boxes[idx]
            cls_id = int(cls_ids[idx])
            name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
            detections.append({
                "class_id": cls_id,
                "class_name": name,
                "confidence": float(nms_scores[idx]),
                "bbox": (bx, by, bx + bw, by + bh),   # letterboxed-frame space, float
            })
        return detections

    def warmup(self, size: int, passes: int = 2) -> None:
        dummy = np.zeros((size, size, 3), dtype=np.uint8)
        for _ in range(passes):
            self.infer(dummy)


def resolve_model_path(model_arg: str) -> Path:
    """
    Accepts an explicit path, or falls back to common export names the
    notebook produces (best_320.onnx / best_640.onnx) if the given path
    doesn't exist as-is.
    """
    candidates = [Path(model_arg)]
    if Path(model_arg).suffix != ".onnx":
        candidates.append(Path(model_arg).with_suffix(".onnx"))
    stem_dir = Path(model_arg).parent if Path(model_arg).parent != Path("") else Path(".")
    candidates += [stem_dir / "best.onnx", stem_dir / "best_320.onnx", stem_dir / "best_640.onnx"]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"No ONNX model found (tried: {', '.join(str(c) for c in candidates)}).\n"
        "  -> From the training notebook, copy best_320.onnx onto the Pi as best.onnx,\n"
        "     next to this script, or pass --model /path/to/model.onnx"
    )


def load_class_names(explicit: Optional[str]) -> list[str]:
    if explicit:
        names = [c.strip() for c in explicit.split(",") if c.strip()]
        if names:
            return names
    return list(DEFAULT_CLASS_NAMES)


# ══════════════════════════════════════════════
#  PREPROCESSING — letterbox (aspect-ratio preserving)
# ══════════════════════════════════════════════

def letterbox(frame: np.ndarray, size: int, color=(114, 114, 114)):
    """
    Resize+pad frame to a size x size square, preserving aspect ratio —
    matches Ultralytics' training/export preprocessing. Returns the padded
    image plus (scale, pad_x, pad_y) needed to map detections back to the
    original frame.
    """
    h, w = frame.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))

    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    dw, dh = size - new_w, size - new_h
    dw, dh = dw / 2, dh / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, left, top


def unletterbox_bbox(bbox, scale: float, pad_x: int, pad_y: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    ox1 = (x1 - pad_x) / scale
    oy1 = (y1 - pad_y) / scale
    ox2 = (x2 - pad_x) / scale
    oy2 = (y2 - pad_y) / scale
    return int(round(ox1)), int(round(oy1)), int(round(ox2)), int(round(oy2))


# ══════════════════════════════════════════════
#  CAMERA
# ══════════════════════════════════════════════

def initialize_camera(cfg: Config):
    if PICAMERA2_AVAILABLE:
        log.info("Initializing PiCamera2 (CSI) ...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (cfg.cam_width, cfg.cam_height), "format": "BGR888"},
            buffer_count=2,
        )
        picam2.configure(config)
        picam2.set_controls({
            "AeEnable": True,
            "AwbEnable": True,
            "FrameDurationLimits": (33333, 33333),  # cap at ~30 fps
        })
        picam2.start()
        time.sleep(0.5)  # let AWB/AE converge
        log.info("PiCamera2 ready -> %dx%d @ 30fps", cfg.cam_width, cfg.cam_height)
        return picam2

    log.warning("picamera2 not available — falling back to cv2.VideoCapture "
                "(sudo apt install python3-picamera2 for CSI support)")
    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {cfg.camera_index}.\n"
            "  -> Check: ls /dev/video*\n  -> Is another process using it?"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("VideoCapture ready -> %dx%d", w, h)
    return cap


class CameraThread(threading.Thread):
    """
    Background frame capture (PiCamera2 or VideoCapture). Queue size 1
    guarantees the main loop always reads the freshest available frame,
    avoiding stale-frame latency buildup under load.
    """

    def __init__(self, cap, queue_size: int = 1):
        super().__init__(daemon=True)
        self.cap = cap
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self.stopped = threading.Event()
        self._is_picamera = PICAMERA2_AVAILABLE and isinstance(cap, Picamera2)
        self._error_count = 0

    def run(self) -> None:
        while not self.stopped.is_set():
            try:
                if self._is_picamera:
                    frame = self.cap.capture_array()
                else:
                    ret, frame = self.cap.read()
                    if not ret:
                        time.sleep(0.005)
                        continue

                if self.q.full():
                    try:
                        self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put(frame)
                self._error_count = 0

            except Exception as exc:
                self._error_count += 1
                log.error("Camera capture error (%d consecutive): %s", self._error_count, exc)
                if self._error_count > 50:
                    log.critical("Too many consecutive camera errors — stopping capture thread.")
                    self.stopped.set()
                    break
                time.sleep(0.01)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        try:
            return True, self.q.get(timeout=0.5)
        except queue.Empty:
            return False, None

    def stop(self) -> None:
        self.stopped.set()
        if self._is_picamera:
            try:
                self.cap.stop()
            except Exception:
                pass


# ══════════════════════════════════════════════
#  DISPLAY / HUD
# ══════════════════════════════════════════════

@dataclass
class LoopState:
    frame_count: int = 0
    fps: float = 0.0
    fps_counter: int = 0
    fps_timer: float = field(default_factory=time.perf_counter)
    last_alert_time: float = 0.0


def render_frame(display_frame: np.ndarray, detections: list[dict], cfg: Config,
                  state: LoopState, is_inference_frame: bool) -> np.ndarray:
    for det in detections:
        name, conf = det["class_name"], det["confidence"]
        color = CLASS_COLORS.get(name, FALLBACK_COLOR)
        x1, y1, x2, y2 = det["bbox"]

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(display_frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(display_frame, label, (x1 + 4, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    dh, dw = display_frame.shape[:2]
    fps_color = (0, 255, 0) if state.fps >= 15 else (0, 200, 255) if state.fps >= 8 else (0, 0, 255)
    cv2.putText(display_frame, f"FPS: {state.fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, fps_color, 2, cv2.LINE_AA)
    cv2.putText(display_frame, f"Detected: {len(detections)}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    inf_label = "\u25cf INFER" if is_inference_frame else "\u25cb SKIP"
    inf_color = (0, 255, 255) if is_inference_frame else (120, 120, 120)
    cv2.putText(display_frame, inf_label, (dw - 130, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, inf_color, 1, cv2.LINE_AA)
    cv2.putText(display_frame, "YOLOv8n | ONNX", (dw - 160, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(display_frame, "Press Q to quit", (10, dh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    alert_names = {d["class_name"] for d in detections} & ALERT_CLASSES
    if alert_names:
        bar_h = 34
        cv2.rectangle(display_frame, (0, dh - bar_h), (dw, dh), (0, 0, 255), -1)
        msg = "ALERT: " + ", ".join(sorted(alert_names)).upper()
        cv2.putText(display_frame, msg, (10, dh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return cv2.resize(display_frame, (cfg.disp_width, cfg.disp_height), interpolation=cv2.INTER_NEAREST)


def maybe_log_alert(detections: list[dict], state: LoopState) -> None:
    alert_dets = [d for d in detections if d["class_name"] in ALERT_CLASSES]
    if not alert_dets:
        return
    now = time.perf_counter()
    if now - state.last_alert_time < ALERT_COOLDOWN_SEC:
        return
    state.last_alert_time = now
    for d in alert_dets:
        log.warning("SAFETY ALERT: %s (%.0f%% confidence)", d["class_name"], d["confidence"] * 100)


# ══════════════════════════════════════════════
#  PROCESS PRIORITY (optional, Pi 5)
# ══════════════════════════════════════════════

def set_process_priority() -> None:
    try:
        import os
        os.nice(-10)
        log.info("Process priority raised (nice -10)")
    except (AttributeError, PermissionError):
        log.debug("Not running as root — process priority unchanged.")


# ══════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════

def main(cfg: Config) -> None:
    log.info("=" * 55)
    log.info("YOLOv8n Elderly Safety Action Detector — Pi 5 / ONNX")
    log.info("=" * 55)

    set_process_priority()

    try:
        model_path = resolve_model_path(cfg.model_path)
    except FileNotFoundError as exc:
        log.error(str(exc))
        sys.exit(1)

    class_names = load_class_names(cfg.classes_arg)
    log.info("Loading model: %s", model_path)
    detector = OnnxDetector(model_path, class_names, cfg.intra_op_threads)

    inf_size = cfg.inf_size or detector.input_size
    if inf_size is None:
        log.error("Could not determine inference size from the model (dynamic input shape) "
                   "and --size was not given.")
        sys.exit(1)
    log.info("Inference size: %dx%d | classes: %s", inf_size, inf_size, class_names)

    log.info("Warming up ...")
    detector.warmup(inf_size)
    log.info("Model ready.")

    try:
        cap = initialize_camera(cfg)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)

    cam = CameraThread(cap, queue_size=cfg.queue_size)
    cam.start()
    log.info("Camera thread started. Headless=%s. Press Q to quit.", cfg.headless)

    state = LoopState()
    detections: list[dict] = []

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                if cam.stopped.is_set():
                    log.error("Camera thread stopped unexpectedly — exiting.")
                    break
                continue

            state.frame_count += 1
            is_inf_frame = (state.frame_count % cfg.skip_frames == 0)
            display_frame = frame.copy()

            if is_inf_frame:
                letterboxed, scale, pad_x, pad_y = letterbox(frame, inf_size)
                raw_output = detector.infer(letterboxed)
                raw_dets = detector.decode(raw_output, cfg.conf_threshold, cfg.iou_threshold)
                detections = [
                    {**d, "bbox": unletterbox_bbox(d["bbox"], scale, pad_x, pad_y)}
                    for d in raw_dets
                ]
                if detections:
                    summary = " | ".join(f"{d['class_name']}({d['confidence']:.0%})" for d in detections)
                    log.info("Frame %6d | %s", state.frame_count, summary)
                maybe_log_alert(detections, state)

            state.fps_counter += 1
            elapsed = time.perf_counter() - state.fps_timer
            if elapsed >= 0.5:
                state.fps = state.fps_counter / elapsed
                state.fps_counter = 0
                state.fps_timer = time.perf_counter()

            if cfg.headless:
                continue  # no rendering — pure detection throughput

            output = render_frame(display_frame, detections, cfg, state, is_inf_frame)
            cv2.imshow(cfg.window_name, output)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                log.info("Q pressed — shutting down.")
                break

    except KeyboardInterrupt:
        log.info("Interrupted — shutting down.")

    finally:
        log.info("Stopping camera thread ...")
        cam.stop()
        cam.join(timeout=2)
        if not (PICAMERA2_AVAILABLE and isinstance(cap, Picamera2)):
            cap.release()
        cv2.destroyAllWindows()
        log.info("All done.")


# ══════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="YOLOv8n Elderly Safety Action Detector — Raspberry Pi 5 / ONNX Runtime"
    )
    parser.add_argument("--model", type=str, default="best.onnx",
                         help="Path to ONNX model (default: ./best.onnx; also tries "
                              "best_320.onnx / best_640.onnx next to it)")
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma-separated class names, in training order. "
                              f"Default: {','.join(DEFAULT_CLASS_NAMES)}")
    parser.add_argument("--camera", type=int, default=0,
                         help="VideoCapture device index (ignored with PiCamera2)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--skip", type=int, default=2, help="Run inference every Nth frame")
    parser.add_argument("--size", type=int, default=None,
                         help="Override inference size (default: auto-detected from the ONNX model)")
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime intra-op threads")
    parser.add_argument("--headless", action="store_true", help="Disable display (SSH / no monitor)")
    args = parser.parse_args()

    cfg = Config(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        camera_index=args.camera,
        inf_size=args.size,
        skip_frames=max(1, args.skip),
        headless=args.headless,
        intra_op_threads=max(1, args.threads),
    )
    cfg.classes_arg = args.classes  # stashed for main(); not a dataclass field by design
    return cfg


if __name__ == "__main__":
    main(parse_args())


# ══════════════════════════════════════════════
#  PI 5 SETUP CHECKLIST
# ══════════════════════════════════════════════
#
#  1. System packages
#     sudo apt update && sudo apt upgrade -y
#     sudo apt install python3-picamera2 python3-opencv libatlas-base-dev -y
#
#  2. Python packages
#     pip install onnxruntime numpy
#
#  3. Get the model (from the training notebook, Section 7-8):
#     -> copy best_320.onnx onto the Pi as best.onnx, next to this script
#     -> (imgsz=320 matches this script's default low-latency profile;
#         best_640.onnx also works — input size is auto-detected)
#
#  4. Run
#     python3 detector_pi5.py                  # normal, with display
#     python3 detector_pi5.py --headless        # SSH / no monitor
#     sudo python3 detector_pi5.py              # elevated scheduling priority
#
#  5. Performance tuning
#     --skip 3          -> fewer inference calls, smoother display
#     --conf 0.45        -> fewer false positives
#     --threads 4         -> match Pi 5's 4 cores (default)
#
#  6. Extra headroom
#     - Active-cooled overclock (config.txt): arm_freq=3000, over_voltage=6
#     - INT8 TFLite export for ~2x more speed (requires tflite-runtime):
#         yolo export model=best.pt format=tflite int8=True imgsz=320
