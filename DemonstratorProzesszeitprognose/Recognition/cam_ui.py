# Recognition/cam_ui.py
import streamlit as st
from pathlib import Path
from Recognition.stack_interface_neu import StackChecker  # anpassen an deinen Pfad

def _keys(key: str):
    return {
        "checker": f"{key}_checker",
        "running": f"{key}_running",
        "variant": f"{key}_variant",
        "step_ready": f"{key}_step_ready",
    }

def start_cam(variant: str, model_path: str, camera_index: int = 1, key: str = "stack"):
    K = _keys(key)

    # schon laufend? -> nur Variante updaten
    if st.session_state.get(K["running"], False) and st.session_state.get(K["checker"]) is not None:
        checker = st.session_state[K["checker"]]
    else:
        checker = StackChecker(model_path, camera_index=camera_index)
        st.session_state[K["checker"]] = checker
        st.session_state[K["running"]] = True

    if st.session_state.get(K["variant"]) != variant:
        checker.set_variant(variant)
        st.session_state[K["variant"]] = variant

def stop_cam(key: str = "stack"):
    K = _keys(key)
    checker = st.session_state.get(K["checker"])
    if checker is not None:
        try:
            checker.release()
        except Exception:
            pass

    for k in K.values():
        st.session_state.pop(k, None)

def next_step(key: str = "stack"):
    K = _keys(key)
    checker = st.session_state.get(K["checker"])
    if checker is not None:
        checker.next_step()

def render_cam(key: str = "stack", run_every: float = 0.5):
    K = _keys(key)

    if not st.session_state.get(K["running"], False):
        st.info("Kamera ist nicht gestartet.")
        return

    # stframe = st.empty()
    ready_box = st.empty()

    @st.fragment(run_every=run_every)
    def _loop():
        checker = st.session_state.get(K["checker"])
        if checker is None:
            return

        try:
            frame, step_ready = checker.check(visualize=False)
        except Exception as e:
            st.session_state[K["step_ready"]] = False
            ready_box.error("❌ READY: False")
            st.error(f"Kamera/YOLO Fehler: {e}")
            return

        st.session_state[K["step_ready"]] = bool(step_ready)

        if step_ready:
            ready_box.success("✅ READY: True")
        else:
            ready_box.info("⏳ READY: False")

        # if frame is None:
        #     st.warning("Kein Kameraframe (Kamera nicht verfügbar?).")
        #     return

        # stframe.image(frame, channels="BGR")

    _loop()
