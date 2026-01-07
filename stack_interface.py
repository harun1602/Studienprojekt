from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


class StackChecker:
    def __init__(self, model_path, camera_index=0, imgsz=640, conf_thres=0.5):
        # ---------------- CONFIG ----------------
        self.MODEL_PATH = model_path
        self.CAMERA_INDEX = camera_index
        self.IMGSZ = imgsz
        self.CONF_THRES = conf_thres

        self.MIN_OVERLAP_RATIO = 0.60
        self.STEP_CONFIRM_FRAMES = 5

        # Box smoothing
        self.EMA_ALPHA = 0.35
        self.MAX_JUMP_PX = 60
        self.HISTORY_LEN = 7

        # Zone defaults (nur Box-Relativ)
        self.BASE_Y_NORM = 0.50     # feste Basis: Box-Mitte
        self.DEFAULT_BAND = 0.12    # half-height relativ zur Boxhöhe
        margin = 0.02               # dein X-margin

        # ---------------- LAYOUTS ----------------
        # NUR box-relativ:
        # Item-Felder:
        #  - label
        #  - x: (x1n, x2n) relativ zur Boxbreite (0..1)
        #  - band: half-height relativ zur Boxhöhe (optional)
        #  - y_offset_norm: Offset relativ zur Boxhöhe (optional, default 0.0)
        #  - y_offset_px: Offset in Pixeln (optional, default 0)
        #  - min_overlap (optional)
        #
        # Step 0 (BOX) wird automatisch davor gesetzt.
        self.module_layouts = {
            "v1": [
                {"id": 0, "items": [{"label": "yellow module", "x": (0.368 - margin, 0.388 + margin)}]},
                {"id": 1, "items": [{"label": "yellow module",     "x": (0.388 - margin, 0.441 + margin)}]},
                {"id": 2, "items": [{"label": "yellow module",       "x": (0.441 - margin, 0.479 + margin), "band": 0.30}]},
                {"id": 3, "items": [{"label": "yellow module",     "x": (0.479 - margin, 0.519 + margin)}]},
            ],
            # v2/v3/v4: hier kannst du deine anderen Varianten genauso lassen/ergänzen
        }

        self.COLOR_MAP = {
            "yellow module":      (0, 255, 255),
            "Blue Module":        (255, 0, 0),
            "big gray module":    (128, 128, 128),
            "small gray module":  (128, 128, 128),
            "gray orange module": (128, 128, 128),
            "35mm":               (0, 128, 128),
            "12.5mm":             (0, 128, 128),
            "black module":       (0, 0, 0),
        }

        # ---------------- INIT ----------------
        self.model = YOLO(self.MODEL_PATH)
        self.cap = cv2.VideoCapture(self.CAMERA_INDEX)

        self.active_variant = "v1"

        # Box: live + locked (locked wird ERST beim Wechsel von Step 0 -> Step 1 gesetzt)
        self.box_live_smoothed = None
        self.box_history = deque(maxlen=self.HISTORY_LEN)

        self.box_locked = None
        self.box_is_locked = False

        # Step state
        self.current_step = 0
        self.step_ok_counter = 0

        # Track-IDs pro Label (optional, hilft bei wiederholten Labels)
        self.used_ids_by_label = {}
        self._last_step_matches = []

        # Step 0 (BOX) einfügen
        self._inject_box_step()

    # ------------------------ Layout helper ------------------------

    def _inject_box_step(self):
        for k, steps in list(self.module_layouts.items()):
            if steps and steps[0].get("type") == "box":
                continue
            self.module_layouts[k] = [{"id": "BOX", "type": "box"}] + steps

    # ------------------------ Drawing helpers ------------------------

    @staticmethod
    def _rect(frame, x1, y1, x2, y2, color, alpha=0.30, thickness=2):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        return frame

    @staticmethod
    def _draw_box(frame, xyxy, color, text=None, thickness=2):
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if text:
            cv2.putText(frame, text, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def _draw_midline(frame, box, color=(0, 255, 255), thickness=2):
        """Horizontale Linie durch die Mitte der Box."""
        x1, y1, x2, y2 = map(int, box)
        y = (y1 + y2) // 2
        cv2.line(frame, (x1, y), (x2, y), color, thickness)

    @staticmethod
    def _overlap_ratio(det_xyxy, zone_xyxy):
        # inter / det_area
        ax1, ay1, ax2, ay2 = det_xyxy
        bx1, by1, bx2, by2 = zone_xyxy
        iw = max(0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        det_area = max(1, (ax2 - ax1) * (ay2 - ay1))
        return inter / det_area

    # ------------------------ YOLO helpers ------------------------

    def find_main_box(self, r):
        """Box-Detection mit höchster Confidence."""
        best_xyxy, best_conf = None, -1.0
        if r.boxes is None:
            return None
        for b in r.boxes:
            if self.model.names[int(b.cls[0])] != "Box":
                continue
            conf = float(b.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_xyxy = list(map(int, b.xyxy[0]))
        return best_xyxy

    def _iter_dets(self, r, label):
        """Yield: (xyxy, conf, track_id_or_None) nur für dieses label."""
        if r.boxes is None:
            return
        for b in r.boxes:
            if self.model.names[int(b.cls[0])] != label:
                continue
            conf = float(b.conf[0])
            if conf < self.CONF_THRES:
                continue
            xyxy = tuple(map(int, b.xyxy[0]))

            tid = None
            if hasattr(b, "id") and b.id is not None:
                try:
                    tid = int(b.id[0])
                except Exception:
                    tid = None

            yield xyxy, conf, tid

    # ------------------------ Box smoothing ------------------------

    @staticmethod
    def _box_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _update_smoothed_box(self, prev, new):
        """EMA smoothing + jump limit."""
        if new is None:
            return prev
        if prev is None:
            return tuple(new)

        pcx, pcy = self._box_center(prev)
        ncx, ncy = self._box_center(new)
        if abs(ncy - pcy) > self.MAX_JUMP_PX or abs(ncx - pcx) > self.MAX_JUMP_PX:
            return prev

        a = self.EMA_ALPHA
        x1 = int((1 - a) * prev[0] + a * new[0])
        y1 = int((1 - a) * prev[1] + a * new[1])
        x2 = int((1 - a) * prev[2] + a * new[2])
        y2 = int((1 - a) * prev[3] + a * new[3])
        return (x1, y1, x2, y2)

    @staticmethod
    def _median_box(boxes):
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        return (int(np.median(xs1)), int(np.median(ys1)), int(np.median(xs2)), int(np.median(ys2)))

    # ------------------------ Zones & matching ------------------------

    def _zone_for_item(self, item, box):
        """
        y ist NICHT mehr vorhanden -> Basis ist immer Box-Mitte (BASE_Y_NORM).
        Du verschiebst nur per y_offset_norm / y_offset_px.
        """
        if box is None:
            return None

        x1n, x2n = item["x"]
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
        """Bestes Match zur Zone (max overlap). Bevorzugt unbenutzte Track-IDs."""
        used = self.used_ids_by_label.get(label, set())
        candidates = []

        for xyxy, conf, tid in self._iter_dets(r, label):
            ratio = self._overlap_ratio(xyxy, zone)
            candidates.append((ratio, conf, tid, xyxy))

        if not candidates:
            return None

        best, best_score = None, -1.0

        def try_pick(prefer_new_ids: bool):
            nonlocal best, best_score
            for ratio, conf, tid, xyxy in candidates:
                if prefer_new_ids and (tid is not None) and (tid in used):
                    continue
                score = ratio + 0.001 * conf
                if score > best_score:
                    best_score = score
                    best = (xyxy, conf, tid, ratio)

        try_pick(True)
        if best is None:
            try_pick(False)

        return best

    # ------------------------ Step checks ------------------------

    def _check_box_step(self, frame, live_box):
        ok = live_box is not None
        if ok:
            self._draw_box(frame, live_box, (255, 255, 255), text="BOX (live)")
            self._draw_midline(frame, live_box)  # ✅ Linie auch in Step 0
        else:
            cv2.putText(frame, "BOX MISSING", (15, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return frame, ok

    def _check_items_step(self, frame, r, box, step):
        all_ok = True
        self._last_step_matches = []

        items = step.get("items", [])
        step_id = step.get("id", "?")

        for idx, item in enumerate(items):
            label = item["label"]
            zone = self._zone_for_item(item, box)
            if zone is None:
                all_ok = False
                continue

            zx1, zy1, zx2, zy2 = zone
            col = self.COLOR_MAP.get(label, (255, 255, 0))

            frame = self._rect(frame, zx1, zy1, zx2, zy2, col)
            cv2.putText(frame, f"STEP {step_id} [{idx+1}/{len(items)}] {label}",
                        (zx1, max(20, zy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            det = self._best_det_for_zone(r, label, zone)
            if det is None:
                all_ok = False
                cv2.putText(frame, "MISSING", (zx1, zy2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue

            det_xyxy, conf, tid, ratio = det
            min_overlap = float(item.get("min_overlap", self.MIN_OVERLAP_RATIO))
            ok = ratio >= min_overlap
            all_ok = all_ok and ok

            status_col = (0, 255, 0) if ok else (0, 0, 255)
            self._draw_box(frame, det_xyxy, status_col, text=f"{label} {ratio:.2f}")

            cv2.putText(frame, f"{'OK' if ok else 'WRONG'} (min {min_overlap:.2f})",
                        (zx1, zy2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_col, 2)

            self._last_step_matches.append((label, tid))

        return frame, all_ok

    # ------------------------ Public API ------------------------

    def set_variant(self, variant_name):
        if variant_name not in self.module_layouts:
            raise ValueError(f"Unknown variant: {variant_name}")
        self.active_variant = variant_name
        self.reset()

    def reset(self):
        self.current_step = 0
        self.step_ok_counter = 0

        self.box_live_smoothed = None
        self.box_history.clear()

        self.box_locked = None
        self.box_is_locked = False

        self.used_ids_by_label = {}
        self._last_step_matches = []

    def next_step(self):
        """
        Wichtig:
        - Box wird NUR gelockt, wenn du Step 0 verlässt (BOX -> erster Item-Step).
        - Vorher bleibt alles LIVE.
        """
        steps = self.module_layouts[self.active_variant]
        if self.current_step < len(steps) and steps[self.current_step].get("type") == "box":
            # ✅ JETZT erst locken (beim Wechsel zu Step 1)
            if self.box_live_smoothed is not None:
                self.box_locked = self.box_live_smoothed
                self.box_is_locked = True

        # commit used IDs (optional)
        for label, tid in self._last_step_matches:
            if tid is None:
                continue
            self.used_ids_by_label.setdefault(label, set()).add(tid)

        self.current_step += 1
        self.step_ok_counter = 0
        self._last_step_matches = []

    def is_done(self):
        return self.current_step >= len(self.module_layouts[self.active_variant])

    def check(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, False

        r = self.model.track(frame, conf=self.CONF_THRES, imgsz=self.IMGSZ, persist=True, verbose=False)[0]
        annotated = frame.copy()

        # Live Box finden + smoothen
        live_box = self.find_main_box(r)
        self.box_live_smoothed = self._update_smoothed_box(self.box_live_smoothed, live_box)

        if self.box_live_smoothed is not None:
            self.box_history.append(self.box_live_smoothed)
            if len(self.box_history) >= 3:
                self.box_live_smoothed = self._median_box(list(self.box_history))

        steps = self.module_layouts[self.active_variant]
        step_ready = False

        if self.current_step < len(steps):
            step = steps[self.current_step]

            if step.get("type") == "box":
                # Step 0: IMMER LIVE anzeigen, NICHT locked benutzen
                annotated, ok = self._check_box_step(annotated, self.box_live_smoothed)
            else:
                # Ab Step 1: bevorzugt locked (sollte existieren, wenn du korrekt weitergeklickt hast)
                box_for_zones = self.box_locked if self.box_is_locked else self.box_live_smoothed

                if box_for_zones is not None:
                    self._draw_box(annotated, box_for_zones, (255, 255, 255),
                                   text="BOX (LOCKED)" if self.box_is_locked else "BOX (LIVE)")
                    self._draw_midline(annotated, box_for_zones)  # ✅ Linie bleibt fix, wenn Box fix ist

                    annotated, ok = self._check_items_step(annotated, r, box_for_zones, step)
                else:
                    ok = False
                    cv2.putText(annotated, "NO BOX", (15, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            self.step_ok_counter = self.step_ok_counter + 1 if ok else 0
            step_ready = self.step_ok_counter >= self.STEP_CONFIRM_FRAMES

        cv2.putText(
            annotated,
            f"Variant: {self.active_variant} | Step: {self.current_step}/{len(steps)-1} | HoldOK: {self.step_ok_counter}/{self.STEP_CONFIRM_FRAMES}",
            (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
        )

        if step_ready:
            cv2.putText(
                annotated,
                "READY (verify + next in UI / press 'n')",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
            )

        return annotated, step_ready

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


# ------------------------ Runner (Test) ------------------------

if __name__ == "__main__":
    checker = StackChecker("newbest.pt")
    checker.set_variant("v1")

    try:
        while not checker.is_done():
            frame, step_ready = checker.check()
            if frame is None:
                continue

            cv2.imshow("STACK CHECK", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("n"):
                checker.next_step()

    finally:
        checker.release()
