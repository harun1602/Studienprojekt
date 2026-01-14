import threading
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

_FRAGMENT = st.fragment if hasattr(st, "fragment") else st.experimental_fragment


@st.cache_resource
def load_yolo(model_path: str):
    return YOLO(model_path)


class StackCheckerCore:
    """
    Deine Logik, aber OHNE cv2.VideoCapture / cv2.imshow.
    Wir bekommen das Frame von streamlit-webrtc und geben annotated + step_ready zurück.
    """
    def __init__(self, model, imgsz=640, conf_thres=0.5):
        self.model = model
        self.IMGSZ = imgsz
        self.CONF_THRES = conf_thres

        self.RAIL_BAND_HALFHEIGHT_NORM = 0.12
        self.MIN_OVERLAP_RATIO = 0.60
        self.EMA_ALPHA = 0.35
        self.MAX_JUMP_PX = 60
        self.HISTORY_LEN = 7
        self.STEP_CONFIRM_FRAMES = 5

        margin = 0.02
        self.module_layouts = {
            "v1": [
                {"id":0,"label":"small gray module","x":(0.368-margin,0.388+margin)},
                {"id":1,"label":"yellow module","x":(0.388-margin,0.441+margin)},
                {"id":2,"label":"Blue Module","x":(0.441-margin,0.479+margin)},
                {"id":3,"label":"big gray module","x":(0.479-margin,0.519+margin)},
            ],
            # ... v2/v3/v4 wie bei dir ...
        }

        self.COLOR_MAP = {
            "yellow module":      (0, 255, 255),
            "Blue Module":        (255, 0, 0),
            "big gray module":    (128, 128, 128),
            "small gray module":  (128, 128, 128),
            "gray orange module": (128, 128, 128),
            "35mm":               (0, 128, 128),
            "black module":       (0, 0, 0),
        }

        self.current_step = 0
        self.step_ok_counter = 0
        self.groin_line_smoothed = None
        self.groin_history = deque(maxlen=self.HISTORY_LEN)
        self.active_variant = "v1"

    def set_variant(self, variant_name: str):
        if variant_name not in self.module_layouts:
            raise ValueError(f"Unknown variant: {variant_name}")
        self.active_variant = variant_name
        self.reset()

    def reset(self):
        self.current_step = 0
        self.step_ok_counter = 0
        self.groin_line_smoothed = None
        self.groin_history.clear()

    def next_step(self):
        self.current_step += 1
        self.step_ok_counter = 0

    def is_done(self):
        return self.current_step >= len(self.module_layouts[self.active_variant])

    @staticmethod
    def draw_transparent_rect(frame, x1, y1, x2, y2, color, alpha=0.35, thickness=2):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame

    @staticmethod
    def overlap_ratio(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        area = max(1, (ax2 - ax1) * (ay2 - ay1))
        return inter / area

    def best_box_for_label(self, r, label):
        best, best_conf = None, -1
        if r.boxes is None:
            return None
        for b in r.boxes:
            if self.model.names[int(b.cls[0])] == label:
                conf = float(b.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best = b
        return best

    def find_main_box(self, r):
        if r.boxes is None:
            return None
        for b in r.boxes:
            if self.model.names[int(b.cls[0])] == "Box":
                return list(map(int, b.xyxy[0]))
        return None

    def compute_groin_line(self, r, main_box):
        ys, xs1, xs2 = [], [], []
        for b in r.boxes:
            if self.model.names[int(b.cls[0])] == "groin":
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                xs1.append(x1); xs2.append(x2)
                ys.append((y1 + y2) // 2)
        if not ys:
            return None
        return (min(xs1), int(np.median(ys)), max(xs2), int(np.median(ys)))

    def update_smoothed_line(self, prev, new):
        if new is None:
            return prev
        if prev is None:
            return new
        if abs(new[1] - prev[1]) > self.MAX_JUMP_PX:
            return prev
        return (
            int((1 - self.EMA_ALPHA) * prev[0] + self.EMA_ALPHA * new[0]),
            int((1 - self.EMA_ALPHA) * prev[1] + self.EMA_ALPHA * new[1]),
            int((1 - self.EMA_ALPHA) * prev[2] + self.EMA_ALPHA * new[2]),
            new[1]
        )

    def check_single_step(self, frame, r, main_box, groin_line, step):
        label = step["label"]
        x1n, x2n = step["x"]

        mx1, my1, mx2, my2 = main_box
        gx1, gy, gx2, _ = groin_line

        line_len = max(1, gx2 - gx1)
        y_band = int((my2 - my1) * self.RAIL_BAND_HALFHEIGHT_NORM)

        zx1 = int(gx1 + x1n * line_len)
        zx2 = int(gx1 + x2n * line_len)
        zy1 = gy - y_band
        zy2 = gy + y_band

        frame = self.draw_transparent_rect(frame, zx1, zy1, zx2, zy2, self.COLOR_MAP.get(label, (255, 255, 0)))
        cv2.putText(frame, f"STEP {step['id']} : {label}", (zx1, zy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        det = self.best_box_for_label(r, label)
        if det is None:
            return frame, False

        mod = tuple(map(int, det.xyxy[0]))
        ratio = self.overlap_ratio(mod, (zx1, zy1, zx2, zy2))
        ok = ratio >= self.MIN_OVERLAP_RATIO

        color = (0, 255, 0) if ok else (0, 0, 255)
        cv2.putText(frame, f"{'OK' if ok else 'WRONG'} {ratio:.2f}",
                    (zx1, zy2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame, ok

    def process_frame(self, frame_bgr):
        """
        frame_bgr: numpy BGR Bild
        returns: annotated_bgr, step_ready(bool)
        """
        step_ready = False

        r = self.model.track(frame_bgr, conf=self.CONF_THRES, imgsz=self.IMGSZ, persist=True, verbose=False)[0]
        annotated = r.plot()  # numpy

        main_box = self.find_main_box(r)
        if main_box:
            new_groin = self.compute_groin_line(r, main_box)
            if new_groin is None:
                mx1, my1, mx2, my2 = main_box
                new_groin = (mx1, (my1 + my2) // 2, mx2, (my1 + my2) // 2)

            self.groin_line_smoothed = self.update_smoothed_line(self.groin_line_smoothed, new_groin)
            if self.groin_line_smoothed:
                self.groin_history.append(self.groin_line_smoothed)
                gy = int(np.median([g[1] for g in self.groin_history]))
                self.groin_line_smoothed = (self.groin_line_smoothed[0], gy, self.groin_line_smoothed[2], gy)
                cv2.line(annotated, (self.groin_line_smoothed[0], gy), (self.groin_line_smoothed[2], gy), (0, 255, 255), 2)

                steps = self.module_layouts[self.active_variant]
                if self.current_step < len(steps):
                    annotated, ok = self.check_single_step(annotated, r, main_box, self.groin_line_smoothed, steps[self.current_step])
                    if ok:
                        self.step_ok_counter += 1
                        if self.step_ok_counter >= self.STEP_CONFIRM_FRAMES:
                            step_ready = True
                    else:
                        self.step_ok_counter = 0

        return annotated, step_ready


class StackVideoProcessor:
    def __init__(self, model_path: str, variant: str, imgsz=640, conf=0.5):
        self.lock = threading.Lock()
        model = load_yolo(model_path)
        self.checker = StackCheckerCore(model, imgsz=imgsz, conf_thres=conf)
        self.checker.set_variant(variant)

        self.step_ready = False
        self.current_step = 0

        # optional: Frame-skip (Performance)
        self._i = 0
        self._last_annotated = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # optional: nur jedes 2. Frame rechnen
        self._i += 1
        if self._i % 2 == 0:
            annotated, ready = self.checker.process_frame(img)
            with self.lock:
                self.step_ready = ready
                self.current_step = self.checker.current_step
            self._last_annotated = annotated
        else:
            annotated = self._last_annotated if self._last_annotated is not None else img

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def show_stackchecker_cam(model_path: str, variant: str = "v1", key: str = "stack_cam"):
    """
    Zeigt Webcam + YOLO Overlay und schreibt den Bool live in:
    st.session_state[f"{key}_step_ready"]
    """
    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=lambda: StackVideoProcessor(model_path, variant),
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=True,
        async_processing=True,
    )

    st.session_state[f"{key}_ctx"] = ctx

    @_FRAGMENT(run_every=0.2)
    def status_under_cam():
        ctx2 = st.session_state.get(f"{key}_ctx")
        ready = False
        step = 0

        if ctx2 and ctx2.video_processor:
            vp = ctx2.video_processor
            with vp.lock:
                ready = bool(vp.step_ready)
                step = int(vp.current_step)

        st.session_state[f"{key}_step_ready"] = ready  # <- DAS ist dein Bool fürs Arbeitsplatz.py

        icon = "✅" if ready else "⬜"
        st.markdown(
            f"<div style='text-align:center; font-size:32px; line-height:1; margin-top:8px'>{icon}</div>",
            unsafe_allow_html=True
        )
        st.caption(f"Aktueller Schritt (YOLO): {step}")

    status_under_cam()
    return ctx

from streamlit_webrtc import VideoHTMLAttributes
def show_stackchecker_cam2(model_path: str, variant: str = "v1", key: str = "stack_cam"):
    # Video unsichtbar machen (Stream läuft trotzdem)
    hidden_video = VideoHTMLAttributes(
        autoPlay=True,
        controls=False,
        muted=True,
        style={
            "width": "1px",
            "height": "1px",
            "opacity": "0",
            "position": "absolute",
            "left": "-10000px",
        },
    )

    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=lambda: StackVideoProcessor(model_path, variant),
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=True,
        async_processing=False,     # stabiler
        video_html_attrs=hidden_video,
    )

    st.session_state[f"{key}_ctx"] = ctx

    @_FRAGMENT(run_every=0.2)
    def status_under_cam():
        ctx2 = st.session_state.get(f"{key}_ctx")
        ready = False
        step = 0

        if ctx2 and ctx2.video_processor:
            vp = ctx2.video_processor
            with vp.lock:
                ready = bool(vp.step_ready)
                step = int(vp.current_step)

        st.session_state[f"{key}_step_ready"] = ready

        icon = "✅" if ready else "⬜"
        st.markdown(
            f"<div style='text-align:center; font-size:32px; line-height:1; margin-top:8px'>{icon}</div>",
            unsafe_allow_html=True
        )
        st.caption(f"Aktueller Schritt (YOLO): {step}")

    status_under_cam()
    return ctx
