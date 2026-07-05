"""
runs against analytics_log.json after each test 
to check tracker quality (e.g. detecting excessive ID switching).
"""

import json
import sys
from collections import defaultdict

data = json.load(open("../outputs/analytics_log.json"))["records"]

per_frame = defaultdict(set)
for r in data:
    per_frame[r["frame_idx"]].add(r["track_id"])

peak_concurrent = max(len(ids) for ids in per_frame.values())
total_unique = len(set(r["track_id"] for r in data))
churn_ratio = total_unique / peak_concurrent

label = sys.argv[1] if len(sys.argv) > 1 else "run"

print(f"Peak concurrent people in any single frame: {peak_concurrent}")
print(f"Total unique IDs over whole clip: {total_unique}")
print(f"Churn ratio (total/peak): {churn_ratio:.2f}")
print(f"\n[{label}] peak={peak_concurrent} unique={total_unique} churn={churn_ratio:.2f}")