import cv2

ref = cv2.imread("snapshots/Variante_1_2/schritt_2.png")  # Referenzbild mit Box
clone = ref.copy()
roi_points = []  # hier speichern wir die zwei Eckpunkte

def select_roi(event, x, y, flags, param):
    global roi_points, ref

    if event == cv2.EVENT_LBUTTONDOWN:
        # 1. Ecke
        roi_points = [(x, y)]
        print(f"Startpunkt: ({x}, {y})")

    elif event == cv2.EVENT_LBUTTONUP:
        # 2. Ecke
        roi_points.append((x, y))
        print(f"Endpunkt:   ({x}, {y})")

        # Rechteck einzeichnen
        cv2.rectangle(ref, roi_points[0], roi_points[1], (0, 255, 0), 2)
        cv2.imshow("image", ref)

        # ROI-Koordinaten sortieren
        x1 = min(roi_points[0][0], roi_points[1][0])
        y1 = min(roi_points[0][1], roi_points[1][1])
        x2 = max(roi_points[0][0], roi_points[1][0])
        y2 = max(roi_points[0][1], roi_points[1][1])

        print(f"ROI = (x1={x1}, y1={y1}, x2={x2}, y2={y2})")

cv2.namedWindow("image")
cv2.setMouseCallback("image", select_roi)

while True:
    cv2.imshow("image", ref)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        # Bild zurücksetzen
        ref = clone.copy()
    elif key == 27:  # ESC zum Beenden
        break

cv2.destroyAllWindows()