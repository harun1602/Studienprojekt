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
}

class IPCManager(BaseManager):
    pass


IPCManager.register("get_cmd_q", callable=lambda: CMD_Q)
IPCManager.register("get_status", callable=lambda: STATUS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best.pt")
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
        STATUS["error"] = ""
        
        stop_loop = False
        while True:
            # --- Commands der Queue abarbeiten ---
            try:
                while True:
                    cmd = CMD_Q.get_nowait()
                    if cmd == "next":
                        checker.next_step()
                    # if checker.is_done():
                    #         STATUS["done"] = True
                    #         raise KeyboardInterrupt
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

            # --- Frame Check ---
            frame, ready = checker.check()
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

            # variant, = checker.collect_step_data()

            # STATUS["variant"] = str(variant)
            # variant       = str(state["variant"])
                   # Liste von dicts

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
