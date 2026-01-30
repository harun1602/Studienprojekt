import threading
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import cv2

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Fragment kompatibel: Streamlit hat je nach Version fragment oder experimental_fragment
_FRAGMENT = st.fragment if hasattr(st, "fragment") else st.experimental_fragment


class VideoProcessor:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self._lock = threading.Lock()
        self.face_present = False  # <- DAS ist dein Bool

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

        faces = self.face_cascade.detectMultiScale(
            small, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        detected = len(faces) > 0
        with self._lock:
            self.face_present = detected

        # Boxen (Face tracking)
        for (x, y, w, h) in faces:
            x, y, w, h = x * 2, y * 2, w * 2, h * 2
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def show_webcam_always_on(key: str = "arbeitsplatz_webcam"):
    # Webcam anzeigen
    ctx = webrtc_streamer(
        key=key,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=True,   # startet automatisch
        async_processing=True,
    )

    # ctx in session_state ablegen, damit Fragment sauber darauf zugreifen kann
    st.session_state[f"{key}_ctx"] = ctx

    # Icon UNTERHALB der Webcam (live)
    @_FRAGMENT(run_every=0.2)
    def face_icon_fragment():
        ctx2 = st.session_state.get(f"{key}_ctx")
        face_present = False

        if ctx2 and ctx2.video_processor:
            vp = ctx2.video_processor
            with vp._lock:
                face_present = bool(vp.face_present)    

        # optional global verfügbar machen
        st.session_state["face_present"] = face_present

        icon = "✅" if face_present else "⬜"
        st.markdown(
            f"<div style='text-align:center; font-size:32px; line-height:1; margin-top:8px'>{icon}</div>",
            unsafe_allow_html=True
        )

    face_icon_fragment()
    return ctx
