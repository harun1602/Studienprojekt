import os
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1) CONFIG / PARAMETER
# ============================================================

# Ordner der aktuellen Datei (damit MODEL_PATH relativ gesetzt werden kann)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ⚠️ HIER anpassen: Pfad zu deinem trainierten YOLO-Modell
MODEL_PATH = os.path.join(PROJECT_DIR,  "best.pt")

# Kameraquelle (0 = Standard-Webcam, ggf. 1/2 oder RTSP/Dateipfad)
CAMERA_INDEX = 0

# YOLO Konfidenz-Threshold: Detections darunter werden ignoriert
CONF_THRES = 0.5

# Inferenz-Auflösung für YOLO (Performance/Genauigkeit)
IMGSZ = 640

# Welche Variante (Layout) soll geprüft werden?
ACTIVE_VARIANT = "v1"  # "v1" / "v2" / "v3" / "v4"

# ------------------------------------------------------------
# Zonen-Definition:
# Wir definieren "erlaubte Bereiche" (Bounding-Boxes) für Module
# RELATIV zur Schiene. Schiene wird als Linie erkannt: (x_start, y, x_end, y)
# - X-Zonen: 0..1 entlang der Schienenlänge (links->rechts)
# - Y-Zone: Band um die Schienenhöhe (± Anteil der Box-Höhe)
# ------------------------------------------------------------

# Vertikale Toleranz um die Schiene:
# Beispiel: 0.12 => Zone geht +/- 12% der Box-Höhe um die Schiene
RAIL_BAND_HALFHEIGHT_NORM = 0.12

# Prüfkriterium: Wie viel der Modulfläche muss in der Zone liegen?
# 0.60 = mind. 60% der Modul-Bounding-Box muss innerhalb der erlaubten Zone sein.
MIN_OVERLAP_RATIO = 0.60

# ------------------------------------------------------------
# Groin-Line Stabilisierung:
# Wir möchten NICHT "erste erkannte Groin-Line einfrieren", sondern:
# - pro Frame neu bestimmen (falls vorhanden)
# - sprunghafte Ausreißer verwerfen (MAX_JUMP_PX)
# - glätten mit EMA (EMA_ALPHA)
# - zusätzlich Median über History (HISTORY_LEN) als Robustheits-Backup
# ------------------------------------------------------------

EMA_ALPHA = 0.35      # 0..1. Höher = reagiert schneller, niedriger = glatter
MAX_JUMP_PX = 60      # wenn Y der Linie zu stark springt => verwerfen
HISTORY_LEN = 7       # wie viele letzte Linien in Median-Filter einfließen

# Logging / Konsolen-Ausgabe nur alle LOG_INTERVAL Sekunden
LOG_INTERVAL = 3.0


# ============================================================
# 2) MODULE LAYOUTS (X-Zonen 0..1 entlang der Schiene)
# ============================================================
# Jede Zone ist (x_start_norm, x_end_norm) entlang der Schienenlänge.
# Beispiel:
#   (0.10, 0.22) bedeutet:
#     - Zone startet bei 10% der Schienenlänge
#     - Zone endet bei 22% der Schienenlänge
#
# Vorteil: unabhängig von Zoom/Abstand/Boxgröße -> immer relativ zur Schiene.

margin = 0.02

module_layouts_norm = {
    "v1": {
        "small gray module": tuple(np.clip([0.368 - margin, 0.388 + margin], 0.0, 1.0)),
        "yellow module":     tuple(np.clip([0.388 - margin, 0.441 + margin], 0.0, 1.0)),
        "Blue Module":       tuple(np.clip([0.441 - margin, 0.479 + margin], 0.0, 1.0)),
        "big gray module":   tuple(np.clip([0.479 - margin, 0.519 + margin], 0.0, 1.0)),
    },
    "v2": {
        "35mm":              tuple(np.clip([0.872 - margin, 1.00 + margin], 0.0, 1.0)),
        "small gray module": tuple(np.clip([0.847 - margin, 0.879 + margin], 0.0, 1.0)),
        "yellow module":     tuple(np.clip([0.795 - margin, 0.861 + margin], 0.0, 1.0)),
        "yellow module":     tuple(np.clip([0.736 - margin, 0.811 + margin], 0.0, 1.0)),
        "big gray module":   tuple(np.clip([0.700 - margin, 0.743 + margin], 0.0, 1.0)),
        "big gray module":   tuple(np.clip([0.661 - margin, 0.700 + margin], 0.0, 1.0)),
        "35mm":              tuple(np.clip([0.511 - margin, 0.661 + margin], 0.0, 1.0)),
        "gray orange module":tuple(np.clip([0.477 - margin, 0.511 + margin], 0.0, 1.0)),
        "Blue Module":       tuple(np.clip([0.431 - margin, 0.477 + margin], 0.0, 1.0)),
        "gray orange module":tuple(np.clip([0.386 - margin, 0.431 + margin], 0.0, 1.0)),
        "Blue Module":       tuple(np.clip([0.338 - margin, 0.383 + margin], 0.0, 1.0)),
        "small gray module": tuple(np.clip([0.321 - margin, 0.342 + margin], 0.0, 1.0))
    },
    "v3": {
        "35mm":                 tuple(np.clip([0.000 - margin, 0.126 + margin], 0.0, 1.0)),
        "small gray module":    tuple(np.clip([0.126 - margin, 0.162 + margin], 0.0, 1.0)),
        "Blue Module":          tuple(np.clip([0.146 - margin, 0.201 + margin], 0.0, 1.0)),
        "big gray module":      tuple(np.clip([0.199 - margin, 0.243 + margin], 0.0, 1.0)),
        "yellow module":        tuple(np.clip([0.225 - margin, 0.290 + margin], 0.0, 1.0)),
        "black module":         tuple(np.clip([0.280 - margin, 0.515 + margin], 0.0, 1.0)),
        "Blue Module":          tuple(np.clip([0.515 - margin, 0.549 + margin], 0.0, 1.0)),
        "yellow module":        tuple(np.clip([0.549 - margin, 0.609 + margin], 0.0, 1.0)),
        "Blue Module":          tuple(np.clip([0.609 - margin, 0.642 + margin], 0.0, 1.0)),
        "yellow module":        tuple(np.clip([0.642 - margin, 0.704 + margin], 0.0, 1.0)),
        "gray orange module":   tuple(np.clip([0.704 - margin, 0.748 + margin], 0.0, 1.0)),
        "gray orange module":   tuple(np.clip([0.748 - margin, 0.781 + margin], 0.0, 1.0)),
        "gray orange module":   tuple(np.clip([0.781 - margin, 0.828 + margin], 0.0, 1.0))
    },
    "v4": {
        "12.5mm": tuple(np.clip([0.000 - margin, 0.048 + margin], 0.0, 1.0)),
        "small gray module": tuple(np.clip([0.033 - margin, 0.069 + margin], 0.0, 1.0)),
        "big gray module": tuple(np.clip([0.072 - margin, 0.122 + margin], 0.0, 1.0)),
        "yellow module": tuple(np.clip([0.108 - margin, 0.183 + margin], 0.0, 1.0)),
        "gray orange module": tuple(np.clip([0.171 - margin, 0.216 + margin], 0.0, 1.0)),
        "Blue Module": tuple(np.clip([0.207 - margin, 0.250 + margin], 0.0, 1.0)),
        "gray orange module": tuple(np.clip([0.240 - margin, 0.290 + margin], 0.0, 1.0)),
        "yellow module": tuple(np.clip([0.275 - margin, 0.333 + margin], 0.0, 1.0)),
        "yellow module": tuple(np.clip([0.326 - margin, 0.390 + margin], 0.0, 1.0)),
        "big gray module": tuple(np.clip([0.390 - margin, 0.431 + margin], 0.0, 1.0)),
        "gray orange module": tuple(np.clip([0.431 - margin, 0.464 + margin], 0.0, 1.0)),
        "Blue Module": tuple(np.clip([0.470 - margin, 0.514 + margin], 0.0, 1.0)),
        "big gray module": tuple(np.clip([0.512 - margin, 0.551 + margin], 0.0, 1.0)),
        "black module": tuple(np.clip([0.551 - margin, 0.782 + margin], 0.0, 1.0)),
        "yellow module": tuple(np.clip([0.789 - margin, 0.852 + margin], 0.0, 1.0)),
        "Blue Module": tuple(np.clip([0.842 - margin, 0.886 + margin], 0.0, 1.0)),
        "Blue Module": tuple(np.clip([0.875 - margin, 0.935 + margin], 0.0, 1.0)),
        "big gray module": tuple(np.clip([0.923 - margin, 0.970 + margin], 0.0, 1.0)),
        "small gray module": tuple(np.clip([0.959 - margin, 0.988 + margin], 0.0, 1.0))
    }
}

# Farben pro Modul (OpenCV nutzt BGR)
COLOR_MAP = {
    "yellow module":     (0, 255, 255),
    "Blue Module":       (255, 0, 0),
    "big gray module":   (128, 128, 128),
    "small gray module": (128, 128, 128),
    "gray orange module":(128, 128, 128),
    "35mm":              (0, 128, 128),
    "black module":      (0, 0, 0),
}


# ============================================================
# 3) Model / Camera initialisieren
# ============================================================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_INDEX)

last_log_time = time.time()


# ============================================================
# 4) Zeichnen / Geometrie-Helfer
# ============================================================

def draw_transparent_rect(frame, x1, y1, x2, y2, color, alpha=0.35, thickness=2):
    """
    Zeichnet eine semi-transparente Box (Overlay) plus Rahmen.
    """
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # gefüllt
    out = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)  # Rahmen
    return out


def overlap_ratio(module_xyxy, zone_xyxy):
    """
    Intersection(module, zone) / Area(module)
    -> Wie viel Prozent der Modulfläche liegt in der Zone?
    """
    mx1, my1, mx2, my2 = module_xyxy
    zx1, zy1, zx2, zy2 = zone_xyxy

    iw = max(0, min(mx2, zx2) - max(mx1, zx1))
    ih = max(0, min(my2, zy2) - max(my1, zy1))
    inter = iw * ih

    m_area = max(0, mx2 - mx1) * max(0, my2 - my1)
    return (inter / m_area) if m_area else 0.0


def find_main_box(r):
    """
    Sucht die Detection mit Label "Box".
    Gibt xyxy als Liste zurück oder None.
    """
    if r.boxes is None:
        return None

    for box in r.boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] == "Box":
            return box.xyxy[0].tolist()
    return None


def compute_groin_line(r, main_box, label="groin", y_offset=0):
    """
    Bestimmt die Schienenlinie aus allen Detections mit Label "groin".

    Idee:
    - Nimm alle groin-Boxes
    - Bestimme:
        x_start = min(x1)
        x_end   = max(x2)
        y_middle = Mitte aus min(y1) und max(y2)  (Union-Mitte)
    - Ergebnis ist eine horizontale Linie:
        (x_start, y_middle, x_end, y_middle)

    Falls keine groin erkannt => None.
    """
    if r.boxes is None:
        return None

    mx1, my1, mx2, my2 = map(int, main_box)
    xs1, xs2, ys1, ys2 = [], [], [], []
    found = False

    for box in r.boxes:
        cls_id = int(box.cls[0])
        obj_label = model.names[cls_id]
        if obj_label != label:
            continue

        found = True
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        xs1.append(int(x1))
        xs2.append(int(x2))
        ys1.append(int(y1))
        ys2.append(int(y2))

    if not found:
        return None

    x_start = min(xs1)
    x_end = max(xs2)

    # Union-Mitte der groin-Boxes
    y_middle = (min(ys1) + max(ys2)) // 2
    y_middle = int(y_middle + y_offset)

    # In Box clampen (sicherheitshalber)
    y_middle = max(my1, min(my2, y_middle))

    # sanity
    if x_end <= x_start:
        return None

    return (x_start, y_middle, x_end, y_middle)


def clamp_line_to_box(line, main_box):
    """
    Stellt sicher, dass Linienkoordinaten innerhalb der Box liegen.
    """
    x1, y, x2, _ = line
    mx1, my1, mx2, my2 = map(int, main_box)

    x1 = max(mx1, min(mx2, x1))
    x2 = max(mx1, min(mx2, x2))
    y  = max(my1, min(my2, y))

    return (x1, y, x2, y)


def update_smoothed_line(prev_line, new_line, alpha=0.35, max_jump_px=60):
    """
    EMA-Glättung + Ausreißerfilter:

    - Wenn new_line None => prev_line behalten
    - Wenn prev_line None => new_line übernehmen
    - Wenn new Y zu stark springt => new verwerfen
    - Sonst: EMA zwischen prev und new

    Warum?
    YOLO Detections schwanken, insbesondere wenn Objekt kurz falsch erkannt wird.
    EMA macht die Linie stabiler.
    """
    if new_line is None:
        return prev_line

    if prev_line is None:
        return new_line

    px1, py, px2, _ = prev_line
    nx1, ny, nx2, _ = new_line

    # Sprung in y zu groß => vermutlich Fehl-Detection
    if abs(ny - py) > max_jump_px:
        return prev_line

    # EMA
    sx1 = int((1 - alpha) * px1 + alpha * nx1)
    sy  = int((1 - alpha) * py  + alpha * ny)
    sx2 = int((1 - alpha) * px2 + alpha * nx2)

    if sx2 <= sx1:
        return prev_line

    return (sx1, sy, sx2, sy)


def best_box_for_label(r, label):
    """
    Nimmt die best-konfidente Detection für ein bestimmtes Label.
    (Falls mehrere vorhanden: wähle die mit größter conf)
    """
    if r.boxes is None:
        return None

    best = None
    best_conf = -1.0
    for box in r.boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] != label:
            continue

        conf = float(box.conf[0]) if box.conf is not None else 0.0
        if conf > best_conf:
            best_conf = conf
            best = box

    return best


def place_and_check_modules(frame, r, main_box, groin_line, module_map_norm,
                            rail_band_halfheight_norm=0.12,
                            min_overlap=0.60):
    """
    Zeichnet pro Modul eine erlaubte Zone und prüft, ob das erkannte Modul
    überwiegend in dieser Zone liegt.

    Zone-Berechnung:
    - X: aus normiertem Bereich (0..1) entlang der Schienenlänge
    - Y: Band um groin_y +/- (rail_band_halfheight_norm * Box-Höhe)

    Bewertung:
    - overlap_ratio >= min_overlap => OK, sonst WRONG
    - Wenn Modul nicht gefunden => MISSING
    """
    mx1, my1, mx2, my2 = map(int, main_box)
    x_start, groin_y, x_end, _ = groin_line

    line_len = max(1, x_end - x_start)
    box_h = max(1, my2 - my1)
    y_band = int(rail_band_halfheight_norm * box_h)

    for label, (x1n, x2n) in module_map_norm.items():
        zone_color = COLOR_MAP.get(label, (128, 128, 255))

        # Normierte X-Grenzen -> Pixel auf Schiene
        zone_x1 = int(x_start + x1n * line_len)
        zone_x2 = int(x_start + x2n * line_len)

        # Band um Schienenhöhe
        zone_y1 = max(my1, int(groin_y - y_band))
        zone_y2 = min(my2, int(groin_y + y_band))

        # Zone zeichnen
        frame = draw_transparent_rect(frame, zone_x1, zone_y1, zone_x2, zone_y2, zone_color, alpha=0.35, thickness=2)
        cv2.putText(frame, label, (zone_x1, max(0, zone_y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        zone_xyxy = (zone_x1, zone_y1, zone_x2, zone_y2)

        # Bestes Modul für dieses Label suchen
        det = best_box_for_label(r, label)
        if det is None:
            cv2.putText(frame, "MISSING",
                        (zone_x1, min(frame.shape[0]-5, zone_y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            continue

        # Modulbox auslesen
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        mod_xyxy = (x1, y1, x2, y2)

        # Overlap prüfen
        inside_ratio = overlap_ratio(mod_xyxy, zone_xyxy)
        ok = inside_ratio >= min_overlap

        status_color = (0, 255, 0) if ok else (0, 0, 255)
        status_text = "OK" if ok else "WRONG"

        # Marker am Modul
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 7, status_color, -1)
        cv2.putText(frame, f"{status_text} ({inside_ratio:.2f})",
                    (zone_x1, min(frame.shape[0]-5, zone_y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    return frame


# ============================================================
# 5) MAIN LOOP
# ============================================================

# Hier halten wir eine "smoothed" Version der Schienenlinie
groin_line_smoothed = None

# History für Median-Glättung (robust gegen Ausreißer)
groin_history = deque(maxlen=HISTORY_LEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO tracking -> liefert r.boxes etc.
    results = model.track(
        frame,
        conf=CONF_THRES,
        imgsz=IMGSZ,
        persist=True,
        verbose=False
    )
    r = results[0]

    # YOLO Standard-Plot (zeigt Detections + Labels)
    annotated = r.plot()

    # Box suchen (Hauptkoordinatensystem)
    main_box = find_main_box(r)
    current_time = time.time()

    if main_box is not None:
        mx1, my1, mx2, my2 = map(int, main_box)
        w = mx2 - mx1
        h = my2 - my1

        # ------------------------------------------
        # A) Groin-Line pro Frame neu bestimmen
        # ------------------------------------------
        new_groin = compute_groin_line(r, main_box, label="groin", y_offset=0)

        # Fallback:
        # Wenn groin nicht erkannt wird, nehmen wir Box-Mitte als "Schiene",
        # damit das System nicht komplett aussetzt.
        if new_groin is None:
            fallback_y = int(my1 + 0.5 * h)
            new_groin = (mx1, fallback_y, mx2, fallback_y)

        # Sicherstellen, dass Linie in der Box bleibt
        new_groin = clamp_line_to_box(new_groin, main_box)

        # ------------------------------------------
        # B) Glätten + Ausreißer verwerfen
        # ------------------------------------------
        groin_line_smoothed = update_smoothed_line(
            groin_line_smoothed,
            new_groin,
            alpha=EMA_ALPHA,
            max_jump_px=MAX_JUMP_PX
        )

        # ------------------------------------------
        # C) Extra Robustheit: Median aus den letzten Linien
        # ------------------------------------------
        if groin_line_smoothed is not None:
            groin_history.append(groin_line_smoothed)

            # Median pro Komponente (x_start, y, x_end)
            xs1 = [g[0] for g in groin_history]
            ys  = [g[1] for g in groin_history]
            xs2 = [g[2] for g in groin_history]

            med_x1 = int(np.median(xs1))
            med_y  = int(np.median(ys))
            med_x2 = int(np.median(xs2))

            groin_line_smoothed = (med_x1, med_y, med_x2, med_y)

        # ------------------------------------------
        # D) Hauptbox + Achsen zeichnen (nur Anzeige)
        # ------------------------------------------
        cx = (mx1 + mx2) // 2
        cy = (my1 + my2) // 2

        cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (0, 0, 255), 3)
        cv2.line(annotated, (cx, my1), (cx, my2), (255, 0, 0), 2)
        cv2.line(annotated, (mx1, cy), (mx2, cy), (255, 0, 0), 2)
        cv2.circle(annotated, (cx, cy), 5, (255, 0, 0), -1)

        cv2.putText(annotated, "Main Coordinate System",
                    (mx1, max(0, my1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ------------------------------------------
        # E) Schiene + Modulzonen einzeichnen und prüfen
        # ------------------------------------------
        if groin_line_smoothed is not None:
            x_start, gy, x_end, _ = groin_line_smoothed

            # Schienenlinie zeichnen
            cv2.line(annotated, (x_start, gy), (x_end, gy), (0, 255, 255), 2)

            # Module anhand der aktiven Variante prüfen
            module_positions = module_layouts_norm.get(ACTIVE_VARIANT, {})

            annotated = place_and_check_modules(
                annotated,
                r,
                main_box,
                groin_line_smoothed,
                module_positions,
                rail_band_halfheight_norm=RAIL_BAND_HALFHEIGHT_NORM,
                min_overlap=MIN_OVERLAP_RATIO
            )

    # ------------------------------------------
    # F) Logging (alle LOG_INTERVAL Sekunden)
    # ------------------------------------------
    if current_time - last_log_time >= LOG_INTERVAL:
        print("\n--- SYSTEM CHECK ---")
        print(f"Box: {'OK' if main_box is not None else 'MISSING'}")
        print(f"Variant: {ACTIVE_VARIANT}")
        if groin_line_smoothed is not None:
            print(f"Groin-Line: y={groin_line_smoothed[1]} x=[{groin_line_smoothed[0]}..{groin_line_smoothed[2]}]")
        else:
            print("Groin-Line: NONE")
        last_log_time = current_time

    # ------------------------------------------
    # G) Anzeigen + Hotkeys
    # ------------------------------------------
    cv2.imshow("YOLO Live Tracking + Groin Modular Zone (relative)", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("saved_frame.png", frame)
        print("Bild gespeichert: saved_frame.png")

cap.release()
cv2.destroyAllWindows()
