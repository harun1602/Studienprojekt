import time
import cv2
from ultralytics import YOLO
import numpy as np

# =========================
# Model
# =========================
model = YOLO("C:/Users/timan/Downloads/Studienprojekt-master/Studienprojekt-master/best.pt")
cap = cv2.VideoCapture(0)

last_log_time = time.time()
LOG_INTERVAL = 3  # seconds

# =========================
# Groin detection
# =========================
def detect_groin(r, main_box, label="groin", threshold_px=15, color=(0, 215, 255), y_offset=0):
    """
    Detect and visualize the groin in the main coordinate system.
    Draws a horizontal line across the groin boxes at their vertical middle (union of all detected groin boxes).

    Args:
        r: YOLO results object
        main_box: Box coordinates (x1,y1,x2,y2) of main box
        label: The label name of groin
        threshold_px: allowed vertical deviation for zone visualization
        color: BGR color for visualization
        y_offset: pixel offset to move the groin line vertically (optional)

    Returns:
        annotated_frame: frame with groin visualization
        groin_line_coords: (x_start, y, x_end, y) of the horizontal line, or None if not detected
    """
    global annotated_frame

    mx1, my1, mx2, my2 = map(int, main_box.tolist())

    groin_found = False
    all_x1, all_x2 = [], []
    all_y1, all_y2 = [], []

    if r.boxes is not None:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            obj_label = model.names[cls_id]

            if obj_label == label:
                groin_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                all_x1.append(x1)
                all_x2.append(x2)
                all_y1.append(y1)
                all_y2.append(y2)

                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                print(f"ID: {track_id} | Label: {obj_label} | Conf: {conf:.2f}")

    groin_line_coords = None
    if groin_found:
        # Horizontal line spans the union of all groin boxes
        x_start = min(all_x1)
        x_end = max(all_x2)

        # Vertical middle of all groin boxes + optional y_offset
        y_middle = (min(all_y1) + max(all_y2)) // 2 

        # Draw horizontal line
        cv2.line(annotated_frame, (x_start, y_middle), (x_end, y_middle), (0, 255, 255), 2)
        groin_line_coords = (x_start, y_middle, x_end, y_middle)
    else:
        print(f" {label.upper()} not detected")

    # Draw horizontal zone
    zone_y1 = max(y_middle - threshold_px, my1) if groin_found else my1
    zone_y2 = min(y_middle + threshold_px, my2) if groin_found else my2
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (mx1, zone_y1), (mx2, zone_y2), color, -1)
    alpha = 0.2
    annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)

    return annotated_frame, groin_line_coords

# =========================
# Place and check modules (relative X)
# =========================
# =========================
# Place and check modules (relative X) with colorized zones
# =========================
def place_modules_on_line(r, main_box, groin_line_coords, module_map):
    """
    Place modules along the groin line with allowed zones drawn as rectangles.
    module_map = {
        "label": (rel_x_start, rel_x_end, y_threshold)
    }
    The rectangle color is based on the module's "color name".
    """
    global annotated_frame
    mx1, my1, mx2, my2 = map(int, main_box.tolist())
    x_line_start, groin_y, x_line_end, _ = groin_line_coords

    # Define colors per module (BGR)
    color_map = {
        "yellow module": (0, 255, 255),      # Yellow
        "blue module": (255, 0, 0),          # Blue
        "gray module": (128, 128, 128),      # Gray
        "lightgray module": (128, 128, 128),     # alternate naming
    }

    for label, (rel_x_start, rel_x_end, y_thresh) in module_map.items():
        zone_color = color_map.get(label, (128, 128, 255))  # fallback color

        # Convert relative X to absolute on groin line
        zone_x1 = x_line_start + rel_x_start
        zone_x2 = x_line_start + rel_x_end
        zone_y1 = max(groin_y - y_thresh, my1)
        zone_y2 = min(groin_y + y_thresh, my2)

        # Draw allowed zone rectangle
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (zone_x1, zone_y1), (zone_x2, zone_y2), zone_color, -1)
        alpha = 0.7
        annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)

        # Check if module is present and mark it
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                module_label = model.names[cls_id]
                if module_label != label:
                    continue

                x1, _, x2, _ = map(int, box.xyxy[0])
                module_center_x = (x1 + x2) // 2

                if zone_x1 <= module_center_x <= zone_x2:
                    status_color = (0, 255, 0)  # Green for correct
                    status_text = "✅ CORRECTLY PLACED"
                else:
                    status_color = (0, 0, 255)  # Red for incorrect
                    status_text = "❌ NOT IN PLACE"

                # Draw marker
                cv2.circle(annotated_frame, (module_center_x, groin_y), 7, status_color, -1)
                cv2.putText(
                    annotated_frame, f"{label}: {status_text}",
                    (module_center_x + 5, groin_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                print(f"Module '{label}' at {module_center_x}, zone [{zone_x1}, {zone_x2}] -> {status_text}")

    return annotated_frame


# =========================
# Module target positions (relative X)
# =========================
module_positions = {
    "lightgray  module": (20, 30, 60),
    "yellow module": (30, 55, 60),       
    "blue module": (55, 80, 60),
    "gray  module": (80, 105, 60)
}

# =========================
# Main loop
# =========================
first_valid_groin_line = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, conf=0.5, imgsz=640, persist=True, verbose=False)
    r = results[0]
    annotated_frame = r.plot()

    main_box = None
    if r.boxes is not None:
        for box in r.boxes:
            if model.names[int(box.cls[0])] == "Box":
                main_box = box.xyxy[0]
                break

    current_time = time.time()

    # Logging every 3 seconds
    if current_time - last_log_time >= LOG_INTERVAL:
        print("\n--- SYSTEM CHECK ---")
        if main_box is not None and first_valid_groin_line is None:
            annotated_frame, groin_line_coords = detect_groin(
                r, main_box, label="groin", threshold_px=15, color=(0, 215, 255), y_offset=20
            )
            if groin_line_coords is not None:
                first_valid_groin_line = groin_line_coords
        last_log_time = current_time

    # Visualization: main box and axes
    if main_box is not None:
        mx1, my1, mx2, my2 = map(int, main_box.tolist())
        cx = (mx1 + mx2) // 2
        cy = (my1 + my2) // 2

        cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), (0, 0, 255), 3)
        cv2.line(annotated_frame, (cx, my1), (cx, my2), (255, 0, 0), 2)
        cv2.line(annotated_frame, (mx1, cy), (mx2, cy), (255, 0, 0), 2)
        cv2.circle(annotated_frame, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(
            annotated_frame,
            "Main Coordinate System",
            (mx1, my1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        # Draw first valid groin line and module zones
        if first_valid_groin_line is not None:
            x_start, groin_y, x_end, _ = first_valid_groin_line
            cv2.line(annotated_frame, (x_start, groin_y), (x_end, groin_y), (0, 255, 255), 2)
            annotated_frame = place_modules_on_line(r, main_box, first_valid_groin_line, module_positions)

    cv2.imshow("YOLO Live Tracking + Groin Modular Zone", annotated_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("saved_frame.png", frame)
        print("Bild gespeichert: saved_frame.png")

cap.release()
cv2.destroyAllWindows()
