"""
Renders bounding boxes, track ID labels, and the live HUD
(unique counter, FPS, latency) onto each frame.
"""
import cv2
import numpy as np
from typing import List
from .detector_tracker import TrackedObject
from .config import Config


class Annotator:
    def __init__(self, cfg: Config):
        self.box_color = tuple(cfg.get("annotation.box_color"))
        self.box_thickness = cfg.get("annotation.box_thickness")
        self.font_scale = cfg.get("annotation.font_scale")
        self.font_thickness = cfg.get("annotation.font_thickness")
        self.hud_color = tuple(cfg.get("annotation.hud_color"))
        self.show_fps = cfg.get("annotation.show_fps")
        self.show_unique = cfg.get("annotation.show_unique_counter")

    def draw_tracks(self, frame: np.ndarray, tracks: List[TrackedObject]) -> np.ndarray:
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.box_color, self.box_thickness)
            label = f"Person {t.track_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                           self.font_scale, self.font_thickness)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), self.box_color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (0, 0, 0),
                        self.font_thickness, cv2.LINE_AA)
        return frame

    def draw_hud(self, frame: np.ndarray, unique_count: int, fps: float, latency_ms: float) -> np.ndarray:
        y = 50
        if self.show_unique:
            cv2.putText(frame, f"Total Unique People: {unique_count}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, self.hud_color, 3, cv2.LINE_AA)
            y += 50
        if self.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f} ms", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, self.hud_color, 3, cv2.LINE_AA)
        return frame