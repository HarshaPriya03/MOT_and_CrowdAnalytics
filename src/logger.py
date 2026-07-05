"""
Writes frame-by-frame track records.
It stream to disk incrementally, one frame at a time, so memory
usage stays flat regardless of video length.

"""
import csv
import json
from pathlib import Path
from typing import List
from .detector_tracker import TrackedObject


class AnalyticsLogger:
    def __init__(self, log_path: str, log_format: str = "json"):
        self.log_format = log_format.lower()
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._first_record_written = False

        if self.log_format == "csv":
            self._csv_file = open(self.path.with_suffix(".csv"), "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["frame_idx", "timestamp_sec", "track_id", "x1", "y1", "x2", "y2", "conf"])
        else:
            self._json_file = open(self.path.with_suffix(".json"), "w")
            self._json_file.write('{\n  "records": [\n')

    def log_frame(self, frame_idx: int, timestamp_sec: float, tracks: List[TrackedObject]):
        for t in tracks:
            x1, y1, x2, y2 = t.bbox.tolist()
            if self.log_format == "csv":
                self._csv_writer.writerow([frame_idx, round(timestamp_sec, 3), t.track_id,
                                            round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                                            round(t.conf, 3)])
            else:
                record = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "track_id": t.track_id,
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "conf": round(t.conf, 3),
                }
                if self._first_record_written:
                    self._json_file.write(",\n")
                self._json_file.write("    " + json.dumps(record))
                self._first_record_written = True

    def close(self, summary: dict = None):
        if self.log_format == "csv":
            self._csv_file.close()
        else:
            self._json_file.write("\n  ],\n  \"summary\": ")
            self._json_file.write(json.dumps(summary or {}, indent=4))
            self._json_file.write("\n}\n")
            self._json_file.close()