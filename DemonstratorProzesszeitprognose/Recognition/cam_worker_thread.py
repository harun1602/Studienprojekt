# cam_worker_thread.py
import threading
import cv2
import time

class CamRunner:
    def __init__(self, checker, window_name="STACK CHECK"):
        self.checker = checker
        self.window_name = window_name
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.step_ready = False
        self.current_step = 0

        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if not self.thread.is_alive():
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.stop_event.set()

    def next_step(self):
        with self.lock:
            self.checker.next_step()

    def _run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while not self.stop_event.is_set() and not self.checker.is_done():
            with self.lock:
                frame, ready = self.checker.check()
                self.step_ready = bool(ready)
                self.current_step = int(self.checker.current_step)

            if frame is not None:
                cv2.imshow(self.window_name, frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            time.sleep(0.01)

        try:
            self.checker.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
