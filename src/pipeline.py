"""
Orchestrates the full flow:
Video Input -> Frame Reader -> Detector/Tracker -> Counter -> Annotator
-> Video Writer + Analytics Logger.
"""
import time
import sys
import traceback
import cv2
from pathlib import Path

from .config import Config
from .detector_tracker import PersonTracker
from .counter import UniqueVisitorCounter
from .annotator import Annotator
from .logger import AnalyticsLogger


class MOTPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tracker = None
        self.counter = UniqueVisitorCounter()
        self.annotator = Annotator(cfg)
        self.logger = AnalyticsLogger(
            log_path=cfg.get("io.log_path", "outputs/analytics_log.json"),
            log_format=cfg.get("io.log_format", "json"),
        )

    def run(self, input_path: str, output_path: str, max_frames: int = None):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Could not open input video: {input_path}")

        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        self.cfg.override("tracker.frame_rate", fps_in)
        occlusion_tolerance_sec = self.cfg.get("tracker.occlusion_tolerance_sec", 5.0)
        track_buffer_frames = max(1, round(occlusion_tolerance_sec * fps_in))
        self.cfg.override("tracker.track_buffer", track_buffer_frames)
        self.tracker = PersonTracker(self.cfg)
        print(f"[INFO] Occlusion tolerance: {occlusion_tolerance_sec}s "
              f"-> track_buffer={track_buffer_frames} frames at {fps_in:.1f}fps")

        print(f"\n[INFO] Video: {width}x{height} @ {fps_in:.1f}fps | Total frames: {total_frames}")
        print(f"[INFO] Device: {self.tracker.device} | Model: {self.cfg.get('model.weights')}")
        print(f"[INFO] Output: {output_path}\n")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        codec = self.cfg.get("io.output_video_codec", "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

        frame_idx = 0
        t_start = time.time()
        log_every = self.cfg.get("performance.log_every_n_frames", 1)

        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                t0 = time.time()

                try:
                    tracks = self.tracker.track_frame(frame)
                except Exception as e:
                    print(f"\n[ERROR] Tracking crashed on frame {frame_idx}")
                    print(f"[ERROR] {type(e).__name__}: {e}")
                    traceback.print_exc()
                    break

                latency_ms = (time.time() - t0) * 1000.0

                ids = [t.track_id for t in tracks]
                total_unique = self.counter.update(ids)

                elapsed = max(time.time() - t_start, 1e-6)
                live_fps = frame_idx / elapsed if frame_idx > 0 else 0.0

                frame = self.annotator.draw_tracks(frame, tracks)
                frame = self.annotator.draw_hud(frame, total_unique, live_fps, latency_ms)

                writer.write(frame)

                if frame_idx % log_every == 0:
                    timestamp_sec = frame_idx / fps_in
                    self.logger.log_frame(frame_idx, timestamp_sec, tracks)

                if frame_idx % 10 == 0:
                    pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                    bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
                    sys.stdout.write(
                        f"\r[{bar}] {pct:.1f}%  Frame {frame_idx}/{total_frames}"
                        f"  FPS: {live_fps:.1f}  People seen: {total_unique}"
                    )
                    sys.stdout.flush()

                frame_idx += 1

        finally:
            print()
            cap.release()
            writer.release()
            total_time = time.time() - t_start
            summary = {
                "input": input_path,
                "output": output_path,
                "total_frames": frame_idx,
                "total_unique_people": self.counter.total,
                "avg_fps": round(frame_idx / total_time, 2) if total_time > 0 else 0,
                "device": self.tracker.device,
                "model_weights": self.cfg.get("model.weights"),
                "tracker": self.cfg.get("tracker.type"),
            }
            self.logger.close(summary=summary)
            return summary