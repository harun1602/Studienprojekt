# stack_runner.py
import argparse
import time
import cv2
from queue import Empty
from multiprocessing.managers import BaseManager
from multiprocessing import Queue

from stack_interface import StackChecker


# -------- IPC Objects --------
# Eine Queue für die Befehle 
CMD_Q = Queue()
# Satates 
STATUS = {
    "ready": False,
    "cam_state": "starting", 
    "cam_error": "",
    "variant": "",
    "step_data": None,
    "bauteile": None
}

class IPCManager(BaseManager):
    pass

IPCManager.register("get_cmd_q", callable=lambda: CMD_Q)
IPCManager.register("get_status", callable=lambda: STATUS)

def _serialize_step_data(d: dict) -> dict:
    def conv(x):
        if isinstance(x, tuple):
            return list(x)
        if isinstance(x, list):
            return [conv(v) for v in x]
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        return x
    return conv(d)

def get_Bauteile(items: list[dict]) -> list[tuple[str, bool]]:

    result: list[tuple[str, bool]] = []

    for it in (items or []):
        label = it.get("label")
        if not label:
            continue  

        erkannt = bool(it.get("detected")) and bool(it.get("ok"))
        result.append((label, erkannt))

    return result
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_yolo_small.pt")
    parser.add_argument("--variant", default="v2")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--port", type=int, default=50055)
    parser.add_argument("--auth", default="stackkey")
    args = parser.parse_args()

    # Manager-Server starten (liefert CMD_Q + STATUS)
    mgr = IPCManager(address=("127.0.0.1", args.port), authkey=args.auth.encode("utf-8"))
    server = mgr.get_server()

    # Server in eigenem Thread laufen lassen
    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    checker = None
    try:
        checker = StackChecker(args.model, camera_index=args.camera)
        checker.set_variant(args.variant)
        try:
            if not checker.cap.isOpened():
                STATUS["cam_state"] = "no_device"
                STATUS["cam_error"] = f"Kamera Index {args.camera} konnte nicht geöffnet werden."
            else:
                STATUS["cam_state"] = "starting"
                STATUS["cam_error"] = ""
        except Exception as e:
            STATUS["cam_state"] = "error"
            STATUS["cam_error"] = str(e)
        

        STATUS["variant"] = args.variant
        STATUS["step_data"] = _serialize_step_data(checker.collect_step_data())
        STATUS["bauteile"] = []
        STATUS["error"] = ""
        
        stop_loop = False
        while True:
            # Commands der Queue abarbeiten 
            try:
                while True:
                    cmd = CMD_Q.get_nowait()
                    if cmd == "next":
                        checker.next_step()  
                    elif cmd == "reset":
                        checker.reset()
                    elif cmd.startswith("set_variant:"):
                        v = cmd.split(":", 1)[1].strip()
                        checker.set_variant(v)
                        STATUS["variant"] = v
                    elif cmd == "stop":
                        stop_loop = True
                        break
            except Empty:
                pass
            if stop_loop:
                break

            # Frame Check 
            frame, ready = checker.check() 
            STATUS["step_data"] = _serialize_step_data(checker.collect_step_data())  
            if frame is None:
                # nach z.B. 3 Sekunden ohne Frame => error
                if "no_frame_since" not in STATUS:
                    STATUS["no_frame_since"] = time.time()
                if time.time() - STATUS["no_frame_since"] > 3.0:
                    STATUS["cam_state"] = "error"
                    STATUS["cam_error"] = f"Keine Frames von Kamera Index {args.camera} (Backend/Blockiert?)."
                else:
                    STATUS["cam_state"] = "starting"
                time.sleep(0.05)
                continue
            else:
                STATUS.pop("no_frame_since", None)
                STATUS["cam_state"] = "ok"
                STATUS["cam_error"] = ""
                if not ready:
                    STATUS["bauteile"] = get_Bauteile(checker.collect_step_data()["items"]) 

            STATUS["ready"] = bool(ready)
            STATUS["step"] = int(checker.current_step)
            STATUS["done"] = bool(checker.is_done())
            STATUS["last_update"] = time.time()

            cv2.imshow("STACK CHECK", frame)
            cv2.waitKey(1)

            if STATUS["done"]:
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        STATUS["error"] = str(e)
    finally:
        STATUS["cam_state"] = "stopped"
        try:
            if checker is not None:
                checker.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
