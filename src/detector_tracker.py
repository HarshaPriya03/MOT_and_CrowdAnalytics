from dataclasses import dataclass
from pathlib import Path
from typing import List
from boxmot.reid.core.reid import ReID
import numpy as np
import torch
from ultralytics import YOLO


try:
    from boxmot import BotSort
except ImportError:
    try:
        from boxmot.trackers.bbox.botsort.botsort import BotSort
    except ImportError:
        try:
            from boxmot.trackers.botsort.botsort import BotSort
        except ImportError:
            from boxmot import BoTSORT as BotSort

from .config import Config
from .device import resolve_device


@dataclass
class TrackedObject:
    track_id: int
    bbox: np.ndarray   
    conf: float
    cls: int


class PersonTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = resolve_device(cfg.get("model.device", "auto"))

        weights = cfg.require("model.weights")
        self.model = YOLO(weights)
        self.model.to(self.device)

        self.conf = cfg.get("model.conf_threshold", 0.25)
        self.iou = cfg.get("model.iou_threshold", 0.4)
        self.classes = cfg.get("model.classes", [0])
        self.imgsz = cfg.get("model.image_size", 1280)

        # Optional verbose logging (raw detection counts vs. confirmed
        # tracks per frame) — useful for isolating whether missing people
        # are a detection issue (low conf/imgsz) or a tracker-filtering
        # issue (min_hits/new_track_thresh/track_high_thresh). Off by
        # default; toggle via tracker.debug in config.yaml.
        self.debug = cfg.get("tracker.debug", False)
        if self.debug:
            print(f"[STARTUP CHECK] self.debug = {self.debug}")
        self._debug_frame_idx = 0

        reid_weights = cfg.get("tracker.reid_weights", "osnet_x0_25_msmt17.pt")

        print(f"[TRACKER] BotSort (boxmot) + OSNet ReID: {reid_weights} on {self.device}")
        print("[TRACKER] ReID enabled: yes — identity will persist through occlusion")

        reid_runtime = ReID(
            weights=Path(reid_weights),
            device=torch.device(self.device),
            half=False,
        )

        self.tracker = BotSort(
            reid_model=reid_runtime.model,
            with_reid=True,
            track_high_thresh=cfg.require("tracker.track_high_thresh"),
            track_low_thresh=cfg.require("tracker.track_low_thresh"),
            new_track_thresh=cfg.require("tracker.new_track_thresh"),
            track_buffer=cfg.get("tracker.track_buffer", 60),
            match_thresh=cfg.require("tracker.match_thresh"),
            proximity_thresh=cfg.require("tracker.proximity_thresh"),
            appearance_thresh=cfg.require("tracker.appearance_thresh"),
            min_hits=cfg.require("tracker.min_hits"),
            frame_rate=int(round(cfg.get("tracker.frame_rate", 30))),
        )

    def track_frame(self, frame: np.ndarray) -> List[TrackedObject]:
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            dets = np.empty((0, 6))
        else:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = r.boxes.cls.cpu().numpy().reshape(-1, 1)
            dets = np.hstack([xyxy, conf, cls])  #

        if self.debug:
            det_confs = dets[:, 4].round(2).tolist() if len(dets) else []
            print(f"[DEBUG] frame {self._debug_frame_idx}: "
                  f"raw YOLO detections = {len(dets)}  confs = {det_confs}")

        tracks_arr = self.tracker.update(dets, frame)

        if self.debug:
            track_ids = tracks_arr[:, 4].astype(int).tolist() if len(tracks_arr) else []
            print(f"[DEBUG] frame {self._debug_frame_idx}: "
                  f"confirmed tracks returned = {len(tracks_arr)}  ids = {track_ids}")
            self._debug_frame_idx += 1

        out: List[TrackedObject] = []
        for row in tracks_arr:
            x1, y1, x2, y2, tid, conf, cls = row[:7]
            out.append(TrackedObject(
                track_id=int(tid),
                bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                conf=float(conf),
                cls=int(cls),
            ))
        return out