# detector_mock.py
import random

class DetectorMock:
    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)
        self.ready: bool = False

    def change(self) -> None:
        """Ändert den Bool (Mock für neue Kamera-Auswertung)."""
        self.ready = self._rng.choice([True, False])
