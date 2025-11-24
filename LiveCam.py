import cv2
import os
from datetime import datetime

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden.")
        return

    # Optional: Auflösung einstellen
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Snapshot-Ordner
    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    # Video-Ordner
    video_dir = "videos"
    os.makedirs(video_dir, exist_ok=True)

    print("Tasten:")
    print("  's' = Snapshot")
    print("  'r' = Aufnahme starten/stoppen")
    print("  'q' = Beenden")

    recording = False
    video_writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kein Bild empfangen (Stream beendet?)")
            break

        # Wenn Aufnahme läuft, Frame ins Video schreiben
        if recording and video_writer is not None:
            video_writer.write(frame)

            # Kleiner REC-Indikator im Bild
            cv2.putText(
                frame,
                "REC",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # Rot
                2,
                cv2.LINE_AA
            )

        cv2.imshow("Live-Video (q=Beenden, s=Snapshot, r=Rec)", frame)

        key = cv2.waitKey(1) & 0xFF

        # Beenden
        if key == ord('q'):
            break

        # Snapshot
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(snapshot_dir, f"snapshot_{timestamp}.png")
            cv2.imwrite(filename, frame)
            print(f"Snapshot gespeichert: {filename}")

        # Aufnahme Start/Stop (Toggle)
        if key == ord('r'):
            # Aufnahme starten
            if not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(video_dir, f"video_{timestamp}.mp4")

                # Auflösung aus Kamera holen
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Codec & Writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 30.0  # gewünschte FPS

                video_writer = cv2.VideoWriter(
                    video_path,
                    fourcc,
                    fps,
                    (frame_width, frame_height)
                )

                if not video_writer.isOpened():
                    print("VideoWriter konnte nicht geöffnet werden, Aufnahme nicht gestartet.")
                    video_writer = None
                else:
                    recording = True
                    print(f"Aufnahme gestartet: {video_path}")

            # Aufnahme stoppen
            else:
                recording = False
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                print("Aufnahme gestoppt.")

    # Aufräumen
    if video_writer is not None:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
