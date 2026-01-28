import cv2
import json
from ultralytics import YOLO
from DemonstratorProzesszeitprognose.data.database_functions import save_recognized_modules_status, get_step_status

class StackChecker:
    """
    Step 0: Box erkennen (LIVE, nicht locked)
    ab Step 1: Zonen nur relativ zur (gelockten) Box prüfen
    Lock passiert erst beim next_step() aus Step 0 heraus.
    """

    def __init__(self, model_path, camera_index=0, imgsz=640, conf_thres=0.5):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(camera_index)

        self.IMGSZ = imgsz
        self.CONF = conf_thres

        self.DEFAULT_MIN_OVERLAP = 0.60
        self.STEP_CONFIRM_FRAMES = 5

        # feste Basis: Box-Mitte + Offset pro Item (y_offset_norm/y_offset_px)
        self.BASE_Y_NORM = 0.50
        self.DEFAULT_BAND = 0.12

        self.module_layouts = {
            "v1": [
                {"id": 0, "min_overlap": 0.95, "items": [
                    {"label": "cable ending", "x": (0.62, 0.71),
                     "band": 0.11, "y_offset_norm": 0.5, "y_offset_px": 0},
                ]},
                {"id": 1, "min_overlap": 0.90, "items": [
                    {"label": "groin", "x": (0.07, 0.93),
                     "band": 0.08, "y_offset_norm": 0.03, "y_offset_px": 0},
                    {"label": "screw", "x": (0.145, 0.185),
                     "band": 0.03, "y_offset_norm": 0.038, "y_offset_px": 0},
                    {"label": "screw", "x": (0.814, 0.854),
                     "band": 0.03, "y_offset_norm": 0.038, "y_offset_px": 0}
                ]},
                {"id": 2, "min_overlap": 0.90, "items": [
                    {"label": "small gray module", "x": (0.382, 0.414),
                     "band": 0.16, "y_offset_norm": 0.02, "y_offset_px": 0}
                ]},
                {"id": 3, "min_overlap": 0.90, "items": [
                    {"label": "yellow module", "x": (0.4, 0.452),
                     "band": 0.18, "y_offset_norm": 0.01, "y_offset_px": 0}
                ]},
                {"id": 4, "min_overlap": 0.90, "items": [
                    {"label": "Blue Module", "x": (0.448, 0.485),
                     "band": 0.22, "y_offset_norm": -0.02, "y_offset_px": 0}
                ]},
                {"id": 5, "min_overlap": 0.90, "items": [
                    {"label": "big gray module", "x": (0.48, 0.524),
                     "band": 0.16, "y_offset_norm": 0.03, "y_offset_px": 0}
                ]},
                {"id": 6, "min_overlap": 0.90, "items": [
                    {"label": "yellow module cable", "x": (0.4, 0.452),
                     "band": 0.04, "y_offset_norm": -0.03, "y_offset_px": 0}
                ]},
                {"id": 7, "min_overlap": 0.90, "items": [
                    {"label": "blue module cable", "x": (0.45, 0.487),
                     "band": 0.04, "y_offset_norm": 0.08, "y_offset_px": 0}
                ]}
            ],
            "v2": [
                {"id": 0, "min_overlap": 0.95, "items": [
                    {"label": "cable ending", "x": (0.09, 0.19),
                     "band": 0.12, "y_offset_norm": 0.5, "y_offset_px": 0},
                    {"label": "cable ending", "x": (0.19, 0.29),
                    "band": 0.12, "y_offset_norm": 0.5, "y_offset_px": 0},
                    {"label": "cable ending", "x": (0.61, 0.71),
                    "band": 0.12, "y_offset_norm": 0.5, "y_offset_px": 0},
                ]},
                {"id": 1, "min_overlap": 0.90, "items": [
                    {"label": "groin", "x": (0.07, 0.93),
                     "band": 0.08, "y_offset_norm": 0.03, "y_offset_px": 0},
                    {"label": "screw", "x": (0.145, 0.185),
                     "band": 0.03, "y_offset_norm": 0.038, "y_offset_px": 0},
                    {"label": "screw", "x": (0.814, 0.854),
                     "band": 0.03, "y_offset_norm": 0.038, "y_offset_px": 0}
                ]},
                {"id": 2, "min_overlap": 0.90, "items": [
                    {"label": "35mm", "x": (0.82, 0.96),
                     "band": 0.22, "y_offset_norm": 0., "y_offset_px": 0}
                ]},
                {"id": 3, "min_overlap": 0.90, "items": [
                    {"label": "small gray module", "x": (0.80, 0.84),
                     "band": 0.16, "y_offset_norm": 0.02, "y_offset_px": 0}
                ]},
                {"id": 4, "min_overlap": 0.90, "items": [
                    {"label": "yellow module", "x": (0.76, 0.82),
                     "band": 0.2, "y_offset_norm": 0, "y_offset_px": 0}
                ]},
                {"id": 5, "min_overlap": 0.90, "items": [
                    {"label": "yellow module", "x": (0.7, 0.77),
                     "band": 0.2, "y_offset_norm": 0, "y_offset_px": 0}
                ]},
                {"id": 6, "min_overlap": 0.90, "items": [
                    {"label": "big gray module", "x": (0.66, 0.72),
                     "band": 0.16, "y_offset_norm": 0.02, "y_offset_px": 0}
                ]},
                {"id": 7, "min_overlap": 0.90, "items": [
                    {"label": "big gray module", "x": (0.63, 0.68),
                     "band": 0.16, "y_offset_norm": 0.02, "y_offset_px": 0}
                ]},
                {"id": 8, "min_overlap": 0.90, "items": [
                    {"label": "35mm", "x": (0.5, 0.65),
                     "band": 0.22, "y_offset_norm": 0, "y_offset_px": 0}
                ]},
                {"id": 9, "min_overlap": 0.90, "items": [
                    {"label": "gray orange module", "x": (0.47, 0.52),
                     "band": 0.23, "y_offset_norm": -0.03, "y_offset_px": 0}
                ]},
                {"id": 10, "min_overlap": 0.90, "items": [
                    {"label": "Blue Module", "x": (0.42, 0.48),
                     "band": 0.23, "y_offset_norm": -0.03, "y_offset_px": 0}
                ]},
                {"id": 11, "min_overlap": 0.90, "items": [
                    {"label": "gray orange module", "x": (0.39, 0.44),
                     "band": 0.23, "y_offset_norm": -0.03, "y_offset_px": 0}
                ]},
                {"id": 12, "min_overlap": 0.90, "items": [
                    {"label": "Blue Module", "x": (0.36, 0.4),
                     "band": 0.23, "y_offset_norm": -0.03, "y_offset_px": 0}
                ]},
                {"id": 13, "min_overlap": 0.90, "items": [
                    {"label": "small gray module", "x": (0.335, 0.38),
                     "band": 0.16, "y_offset_norm": 0.03, "y_offset_px": 0}
                ]},
                {"id": 14, "min_overlap": 0.90, "items": [
                    {"label": "blue module cable", "x": (0.36, 0.4),
                     "band": 0.04, "y_offset_norm": -0.22, "y_offset_px": 0}
                ]},
                {"id": 15, "min_overlap": 0.90, "items": [
                    {"label": "blue module cable", "x": (0.43, 0.48),
                     "band": 0.04, "y_offset_norm": -0.22, "y_offset_px": 0}
                ]},
                {"id": 16, "min_overlap": 0.90, "items": [
                    {"label": "grey module cable", "x": (0.47, 0.52),
                     "band": 0.05, "y_offset_norm": 0.07, "y_offset_px": 0}
                ]},
                {"id": 17, "min_overlap": 0.90, "items": [
                    {"label": "blue module cable", "x": (0.43, 0.48),
                     "band": 0.05, "y_offset_norm": 0.07, "y_offset_px": 0}
                ]},
                {"id": 18, "min_overlap": 0.90, "items": [
                    {"label": "yellow module cable", "x": (0.7, 0.77),
                     "band": 0.04, "y_offset_norm": -0.03, "y_offset_px": 0}
                ]},
                {"id": 19, "min_overlap": 0.90, "items": [
                    {"label": "blue module cable", "x": (0.36, 0.4),
                     "band": 0.04, "y_offset_norm": 0.07, "y_offset_px": 0}
                ]},
                {"id": 20, "min_overlap": 0.90, "items": [
                    {"label": "yellow module cable", "x": (0.7, 0.77),
                     "band": 0.04, "y_offset_norm": 0.03, "y_offset_px": 0}
                ]},
                {"id": 21, "min_overlap": 0.90, "items": [
                    {"label": "grey module cable", "x": (0.76, 0.82),
                     "band": 0.05, "y_offset_norm": 0.03, "y_offset_px": 0}
                ]},
            ]
        }

        self.COLOR = {
            "yellow module": (0, 255, 255),
            "Blue Module": (255, 0, 0),
            "big gray module": (128, 128, 128),
            "small gray module": (128, 128, 128),
            "gray orange module": (128, 128, 128),
            "35mm": (0, 128, 128),
            "12.5mm": (0, 128, 128),
            "black module": (0, 0, 0),
        }

        self.active_variant = "v2"
        self.current_step = 0  # 0 = Box-Step
        self.ok_counter = 0

        self.box_live = None
        self.box_locked = None
        self.box_is_locked = False

        self.used_ids_by_label = {}
        self._last_step_matches = []

    # -------------------- Drawing helpers --------------------

    @staticmethod
    def _draw_box(img, xyxy, text=None, color=(255, 255, 255), thickness=2):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if text:
            cv2.putText(img, text, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def _draw_midline(img, box, color=(0, 255, 255), thickness=2):
        x1, y1, x2, y2 = map(int, box)
        y = (y1 + y2) // 2
        cv2.line(img, (x1, y), (x2, y), color, thickness)

    @staticmethod
    def _rect_alpha(img, x1, y1, x2, y2, color, alpha=0.30, thickness=2):
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return img

    @staticmethod
    def _overlap_ratio(det, zone):
        ax1, ay1, ax2, ay2 = det
        bx1, by1, bx2, by2 = zone
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        det_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        return inter / det_area

    # -------------------- YOLO helpers --------------------

    def _find_best_box(self, r, cls_name="Box"):
        best, best_conf = None, -1.0
        if r.boxes is None:
            return None
        for b in r.boxes:
            if self.model.names[int(b.cls[0])] != cls_name:
                continue
            conf = float(b.conf[0])
            if conf > best_conf:
                best_conf = conf
                best = tuple(map(int, b.xyxy[0]))
        return best

    def _iter_label(self, r, label):
        if r.boxes is None:
            return
        for b in r.boxes:
            if self.model.names[int(b.cls[0])] != label:
                continue
            conf = float(b.conf[0])
            if conf < self.CONF:
                continue
            xyxy = tuple(map(int, b.xyxy[0]))

            tid = None
            if hasattr(b, "id") and b.id is not None:
                try:
                    tid = int(b.id[0])
                except Exception:
                    tid = None

            yield xyxy, conf, tid

    # -------------------- Zone & matching --------------------

    def _zone_for_item(self, item, box):
        if box is None:
            return None

        (x1n, x2n) = item["x"]
        band = float(item.get("band", self.DEFAULT_BAND))
        y_off_n = float(item.get("y_offset_norm", 0.0))
        y_off_px = int(item.get("y_offset_px", 0))

        mx1, my1, mx2, my2 = box
        bw = max(1, mx2 - mx1)
        bh = max(1, my2 - my1)

        y_center = int(my1 + (self.BASE_Y_NORM + y_off_n) * bh + y_off_px)
        y_band = int(band * bh)

        zx1 = int(mx1 + x1n * bw)
        zx2 = int(mx1 + x2n * bw)
        zy1 = y_center - y_band
        zy2 = y_center + y_band
        return (zx1, zy1, zx2, zy2)

    def _best_det_for_zone(self, r, label, zone):
        used = self.used_ids_by_label.get(label, set())
        candidates = []
        for xyxy, conf, tid in self._iter_label(r, label):
            ratio = self._overlap_ratio(xyxy, zone)
            candidates.append((ratio, conf, tid, xyxy))

        if not candidates:
            return None

        def pick(prefer_new_ids: bool):
            best = None
            best_score = -1.0
            for ratio, conf, tid, xyxy in candidates:
                if prefer_new_ids and (tid is not None) and (tid in used):
                    continue
                score = ratio + 0.001 * conf
                if score > best_score:
                    best_score = score
                    best = (xyxy, conf, tid, ratio)
            return best

        return pick(True) or pick(False)

    # -------------------- Steps --------------------

    def set_variant(self, name):
        if name not in self.module_layouts:
            raise ValueError(f"Unknown variant: {name}")
        self.active_variant = name
        self.reset()

    def reset(self):
        self.current_step = 0
        self.ok_counter = 0
        self.box_live = None
        self.box_locked = None
        self.box_is_locked = False
        self.used_ids_by_label = {}
        self._last_step_matches = []

    def is_done(self):
        return self.current_step >= (len(self.module_layouts[self.active_variant]) + 1)

    def next_step(self):
        if self.current_step == 0 and self.box_live is not None:
            self.box_locked = self.box_live
            self.box_is_locked = True

        for label, tid in self._last_step_matches:
            if tid is None:
                continue
            self.used_ids_by_label.setdefault(label, set()).add(tid)

        self.current_step += 1
        self.ok_counter = 0
        self._last_step_matches = []

    def check(self, task_id=None, session=None):
        ret, frame = self.cap.read()
        if not ret:
            return None, False

        r = self.model.track(frame, conf=self.CONF, imgsz=self.IMGSZ, persist=True, verbose=False)[0]
        out = frame.copy()

        self.box_live = self._find_best_box(r, "Box")

        # --- Step 0: Box-Step ---
        if self.current_step == 0:
            ok = self.box_live is not None
            if ok:
                self._draw_box(out, self.box_live, text="BOX (LIVE)")
                self._draw_midline(out, self.box_live)
            else:
                cv2.putText(out, "BOX MISSING", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- Step 1..N: Montage-Steps ---
        else:
            step_index = self.current_step - 1
            step = self.module_layouts[self.active_variant][step_index]

            box = self.box_locked if self.box_is_locked else self.box_live
            ok = False

            if box is None:
                cv2.putText(out, "NO BOX (lock first)", (15, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                self._draw_box(out, box, text="BOX (LOCKED)" if self.box_is_locked else "BOX (LIVE)")
                self._draw_midline(out, box)

                step_min_overlap = float(step.get("min_overlap", self.DEFAULT_MIN_OVERLAP))

                ok = True
                self._last_step_matches = []

                for item in step.get("items", []):
                    label = item["label"]
                    zone = self._zone_for_item(item, box)
                    if zone is None:
                        ok = False
                        continue

                    zx1, zy1, zx2, zy2 = zone
                    col = self.COLOR.get(label, (255, 255, 0))
                    out = self._rect_alpha(out, zx1, zy1, zx2, zy2, col)

                    # Name am Zielbereich
                    cv2.putText(out, label, (zx1, max(20, zy1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

                    det = self._best_det_for_zone(r, label, zone)
                    if det is None:
                        ok = False
                        cv2.putText(out, f"MISSING (min {step_min_overlap:.2f})", (zx1, zy2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)
                        continue

                    det_xyxy, det_conf, tid, ratio = det

                    # Item kann step-min überschreiben
                    min_ov = float(item.get("min_overlap", step_min_overlap))
                    is_ok = ratio >= min_ov
                    ok = ok and is_ok

                    status_col = (0, 255, 0) if is_ok else (0, 0, 255)

                    # ✅ erkanntes Modul: CONF anzeigen
                    self._draw_box(out, det_xyxy, text=f"confidence {det_conf:.2f}", color=status_col)

                    # ✅ Zielzone: overlap + min overlap anzeigen
                    cv2.putText(out, f"{'OK' if is_ok else 'WRONG'} ov {ratio:.2f} (min {min_ov:.2f})",
                                (zx1, zy2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_col, 2)

                    self._last_step_matches.append((label, tid))

        # Hold counter -> step_ready
        self.ok_counter = self.ok_counter + 1 if ok else 0
        step_ready = self.ok_counter >= self.STEP_CONFIRM_FRAMES
        
        current_status = get_step_status(r, frame)
        
        if task_id and session:
            save_recognized_modules_status(
                session=session,
                task_id=task_id,
                current_step=self.current_step,
                step_status=current_status
            )

        total_steps = len(self.module_layouts[self.active_variant])
        cv2.putText(out,
                    f"Variant: {self.active_variant} | Step: {self.current_step}/{total_steps} | HoldOK: {self.ok_counter}/{self.STEP_CONFIRM_FRAMES}",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        if step_ready:
            cv2.putText(out, "READY (verify + next in UI / press 'n')",
                        (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        return out, step_ready, current_status

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    checker = StackChecker("newbestbestbest.pt")
    checker.set_variant("v2")

    try:
        while not checker.is_done():
            frame, ready = checker.check()
            if frame is None:
                continue

            cv2.imshow("STACK CHECK", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("n"):
                checker.next_step()
    finally:
        checker.release()
