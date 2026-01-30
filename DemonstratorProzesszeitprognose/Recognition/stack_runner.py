# stack_runner.py
import argparse
import time
import cv2
from queue import Empty
from multiprocessing.managers import BaseManager
from multiprocessing import Queue

from stack_interface import StackChecker


# -------- IPC Objects (werden per Manager geteilt) --------
CMD_Q = Queue()   # Commands: "next", "stop", "reset", "set_variant:v2", ...
STATUS = {
    "running": False,
    "ready": False,
    "step": 0,
    "total_steps": 0,
    "variant": "",
    "done": False,
    "last_update": 0.0,
    "error": ""
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

        STATUS["running"] = True
        STATUS["variant"] = args.variant
        STATUS["total_steps"] = len(checker.module_layouts[checker.active_variant])
        STATUS["error"] = ""

        while True:
            # --- Commands abarbeiten ---
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
                        STATUS["total_steps"] = len(checker.module_layouts[checker.active_variant])
                    elif cmd == "stop":
                        raise KeyboardInterrupt
            except Empty:
                pass

            # --- Frame Check ---
            frame, ready = checker.check()
            if frame is None:
                continue

            STATUS["ready"] = bool(ready)
            STATUS["step"] = int(checker.current_step)
            STATUS["done"] = bool(checker.is_done())
            STATUS["last_update"] = time.time()

            cv2.imshow("STACK CHECK", frame)
            k = cv2.waitKey(1) & 0xFF

            # Optional: Tastatur bleibt zusätzlich möglich
            if k == ord("q"):
                break
            if k == ord("n"):
                checker.next_step()

            if STATUS["done"]:
                break

    except KeyboardInterrupt:
        pass
    except Exception as e:
        STATUS["error"] = str(e)
    finally:
        STATUS["running"] = False
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
