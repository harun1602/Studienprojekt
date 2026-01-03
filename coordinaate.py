import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1) CONFIG
# ============================================================

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_DIR, "best.pt")

CAMERA_INDEX = 0
CONF_THRES = 0.5
IMGSZ = 640

ACTIVE_VARIANT = "v1"

RAIL_BAND_HALFHEIGHT_NORM = 0.12
MIN_OVERLAP_RATIO = 0.60

EMA_ALPHA = 0.35
MAX_JUMP_PX = 60
HISTORY_LEN = 7

LOG_INTERVAL = 3.0

STEP_CONFIRM_FRAMES = 5   # wie viele Frames OK nötig sind


# ============================================================
# 2) STACK-BASIERTE LAYOUTS (REIHENFOLGE!)
# ============================================================

margin = 0.02

module_layouts = {
    "v1": [
        {"id":0,"label":"small gray module","x":(0.368-margin,0.388+margin)},
        {"id":1,"label":"yellow module","x":(0.388-margin,0.441+margin)},
        {"id":2,"label":"Blue Module","x":(0.441-margin,0.479+margin)},
        {"id":3,"label":"big gray module","x":(0.479-margin,0.519+margin)},
    ],
    "v2": [
        {"id":0,"label":"35mm","x":(0.872-margin,1.00+margin)},
        {"id":1,"label":"small gray module","x":(0.847-margin,0.879+margin)},
        {"id":2,"label":"yellow module","x":(0.795-margin,0.861+margin)},
        {"id":3,"label":"yellow module","x":(0.736-margin,0.811+margin)},
        {"id":4,"label":"big gray module","x":(0.700-margin,0.743+margin)},
        {"id":5,"label":"big gray module","x":(0.661-margin,0.700+margin)},
        {"id":6,"label":"35mm","x":(0.511-margin,0.661+margin)},
        {"id":7,"label":"gray orange module","x":(0.477-margin,0.511+margin)},
        {"id":8,"label":"Blue Module","x":(0.431-margin,0.477+margin)},
        {"id":9,"label":"gray orange module","x":(0.386-margin,0.431+margin)},
        {"id":10,"label":"Blue Module","x":(0.338-margin,0.383+margin)},
        {"id":11,"label":"small gray module","x":(0.321-margin,0.342+margin)}
    ],
    "v3": [
        {"id":0,"label":"35mm","x":(0.0-margin,0.126+margin)},
        {"id":1,"label":"small gray module","x":(0.126-margin,0.162+margin)},
        {"id":2,"label":"Blue Module","x":(0.146-margin,0.201+margin)},
        {"id":3,"label":"big gray module","x":(0.199-margin,0.243+margin)},
        {"id":4,"label":"yellow module","x":(0.225-margin,0.290+margin)},
        {"id":5,"label":"black module","x":(0.280-margin,0.515+margin)},
        {"id":6,"label":"Blue Module","x":(0.515-margin,0.549+margin)},
        {"id":7,"label":"yellow module","x":(0.549-margin,0.609+margin)},
        {"id":8,"label":"Blue Module","x":(0.609-margin,0.642+margin)},
        {"id":9,"label":"yellow module","x":(0.642-margin,0.704+margin)},
        {"id":10,"label":"gray orange module","x":(0.704-margin,0.748+margin)},
        {"id":11,"label":"gray orange module","x":(0.748-margin,0.781+margin)},
        {"id":12,"label":"gray orange module","x":(0.781-margin,0.828+margin)}
    ],
    "v4": [
        {"id":0,"label":"12.5mm","x":(0.0-margin,0.048+margin)},
        {"id":1,"label":"small gray module","x":(0.033-margin,0.069+margin)},
        {"id":2,"label":"big gray module","x":(0.072-margin,0.122+margin)},
        {"id":3,"label":"yellow module","x":(0.108-margin,0.183+margin)},
        {"id":4,"label":"gray orange module","x":(0.171-margin,0.216+margin)},
        {"id":5,"label":"Blue Module","x":(0.207-margin,0.250+margin)},
        {"id":6,"label":"gray orange module","x":(0.240-margin,0.290+margin)},
        {"id":7,"label":"yellow module","x":(0.275-margin,0.333+margin)},
        {"id":8,"label":"yellow module","x":(0.326-margin,0.390+margin)},
        {"id":9,"label":"big gray module","x":(0.390-margin,0.431+margin)},
        {"id":10,"label":"gray orange module","x":(0.431-margin,0.464+margin)},
        {"id":11,"label":"Blue Module","x":(0.470-margin,0.514+margin)},
        {"id":12,"label":"big gray module","x":(0.512-margin,0.551+margin)},
        {"id":13,"label":"black module","x":(0.551-margin,0.782+margin)},
        {"id":14,"label":"yellow module","x":(0.789-margin,0.852+margin)},
        {"id":15,"label":"Blue Module","x":(0.842-margin,0.886+margin)},
        {"id":16,"label":"Blue Module","x":(0.875-margin,0.935+margin)},
        {"id":17,"label":"big gray module","x":(0.923-margin,0.970+margin)},
        {"id":18,"label":"small gray module","x":(0.959-margin,0.988+margin)}
    ]
}



COLOR_MAP = {
    "yellow module":      (0, 255, 255),
    "Blue Module":        (255, 0, 0),
    "big gray module":    (128, 128, 128),
    "small gray module":  (128, 128, 128),
    "gray orange module": (128, 128, 128),
    "35mm":               (0, 128, 128),
    "black module":       (0, 0, 0),
}


# ============================================================
# 3) INIT
# ============================================================

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)

last_log_time = time.time()

current_step = 0
step_ok_counter = 0

groin_line_smoothed = None
groin_history = deque(maxlen=HISTORY_LEN)


# ============================================================
# 4) HELPER
# ============================================================

def draw_transparent_rect(frame, x1, y1, x2, y2, color, alpha=0.35, thickness=2):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), color, -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
    return frame


def overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2,bx2) - max(ax1,bx1))
    ih = max(0, min(ay2,by2) - max(ay1,by1))
    inter = iw * ih
    area = max(1,(ax2-ax1)*(ay2-ay1))
    return inter / area


def best_box_for_label(r, label):
    best, best_conf = None, -1
    if r.boxes is None:
        return None
    for b in r.boxes:
        if model.names[int(b.cls[0])] == label:
            conf = float(b.conf[0])
            if conf > best_conf:
                best_conf = conf
                best = b
    return best


def find_main_box(r):
    if r.boxes is None:
        return None
    for b in r.boxes:
        if model.names[int(b.cls[0])] == "Box":
            return list(map(int, b.xyxy[0]))
    return None


def compute_groin_line(r, main_box):
    ys, xs1, xs2 = [], [], []
    for b in r.boxes:
        if model.names[int(b.cls[0])] == "groin":
            x1,y1,x2,y2 = map(int,b.xyxy[0])
            xs1.append(x1)
            xs2.append(x2)
            ys.append((y1+y2)//2)
    if not ys:
        return None
    return (min(xs1), int(np.median(ys)), max(xs2), int(np.median(ys)))


def update_smoothed_line(prev, new):
    if new is None:
        return prev
    if prev is None:
        return new
    if abs(new[1] - prev[1]) > MAX_JUMP_PX:
        return prev
    return (
        int((1-EMA_ALPHA)*prev[0] + EMA_ALPHA*new[0]),
        int((1-EMA_ALPHA)*prev[1] + EMA_ALPHA*new[1]),
        int((1-EMA_ALPHA)*prev[2] + EMA_ALPHA*new[2]),
        new[1]
    )


def check_single_step(frame, r, main_box, groin_line, step):
    label = step["label"]
    x1n, x2n = step["x"]

    mx1,my1,mx2,my2 = main_box
    gx1,gy,gx2,_ = groin_line

    line_len = max(1, gx2-gx1)
    y_band = int((my2-my1)*RAIL_BAND_HALFHEIGHT_NORM)

    zx1 = int(gx1 + x1n*line_len)
    zx2 = int(gx1 + x2n*line_len)
    zy1 = gy - y_band
    zy2 = gy + y_band

    frame = draw_transparent_rect(frame, zx1, zy1, zx2, zy2, COLOR_MAP.get(label,(255,255,0)))
    cv2.putText(frame, f"STEP {step['id']} : {label}", (zx1, zy1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    det = best_box_for_label(r, label)
    if det is None:
        return frame, False

    mod = tuple(map(int, det.xyxy[0]))
    ratio = overlap_ratio(mod, (zx1,zy1,zx2,zy2))
    ok = ratio >= MIN_OVERLAP_RATIO

    color = (0,255,0) if ok else (0,0,255)
    cv2.putText(frame, f"{'OK' if ok else 'WRONG'} {ratio:.2f}",
                (zx1, zy2+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame, ok


# ============================================================
# 5) MAIN LOOP
# ============================================================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    r = model.track(frame, conf=CONF_THRES, imgsz=IMGSZ, persist=True, verbose=False)[0]
    annotated = r.plot()

    main_box = find_main_box(r)
    if main_box:
        mx1,my1,mx2,my2 = main_box
        cv2.rectangle(annotated,(mx1,my1),(mx2,my2),(0,0,255),3)

        new_groin = compute_groin_line(r, main_box)
        if new_groin is None:
            new_groin = (mx1,(my1+my2)//2,mx2,(my1+my2)//2)

        groin_line_smoothed = update_smoothed_line(groin_line_smoothed, new_groin)
        if groin_line_smoothed:
            groin_history.append(groin_line_smoothed)
            gy = int(np.median([g[1] for g in groin_history]))
            groin_line_smoothed = (groin_line_smoothed[0], gy, groin_line_smoothed[2], gy)
            cv2.line(annotated,(groin_line_smoothed[0],gy),(groin_line_smoothed[2],gy),(0,255,255),2)

            steps = module_layouts[ACTIVE_VARIANT]
            if current_step < len(steps):
                annotated, ok = check_single_step(
                    annotated, r, main_box, groin_line_smoothed, steps[current_step]
                )
                if ok:
                    step_ok_counter += 1
                    if step_ok_counter >= STEP_CONFIRM_FRAMES:
                        print(f"STEP {current_step} DONE")
                        current_step += 1
                        step_ok_counter = 0
                else:
                    step_ok_counter = 0
            else:
                cv2.putText(annotated,"ALL STEPS DONE",(50,50),
                            cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),3)

    cv2.imshow("STACK CHECK", annotated)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        current_step = 0
        step_ok_counter = 0
        print("STACK RESET")

cap.release()
cv2.destroyAllWindows()
