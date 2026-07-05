"""
CLI entrypoint.

Usage:
    python main.py --input video.mp4 --output result.mp4
    python main.py --input video.mp4 --output result.mp4 --config config.yaml \
        --conf 0.5 --model yolov8n.pt --device cuda
"""
import argparse
from src.config import Config
from src.pipeline import MOTPipeline


def parse_args():
    p = argparse.ArgumentParser(description="Real-time MOT & Crowd Analytics Pipeline")
    p.add_argument("--input", required=True, help="Path to input video file")
    p.add_argument("--output", default="outputs/result.mp4", help="Path to annotated output video")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")

    # Optional CLI overrides -- take precedence over config.yaml if provided
    p.add_argument("--model", default=None, help="Override model weights, e.g. yolov8n.pt")
    p.add_argument("--device", default=None, help="cuda | mps | cpu | auto")
    p.add_argument("--conf", type=float, default=None, help="Override detection confidence threshold")
    p.add_argument("--iou", type=float, default=None, help="Override IoU threshold")
    p.add_argument("--tracker", default=None, help="botsort.yaml | bytetrack.yaml")
    p.add_argument("--log-format", default=None, choices=["json", "csv"])

    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.load(args.config)

    if args.model:
        cfg.override("model.weights", args.model)
    if args.device:
        cfg.override("model.device", args.device)
    if args.conf is not None:
        cfg.override("model.conf_threshold", args.conf)
    if args.iou is not None:
        cfg.override("model.iou_threshold", args.iou)
    if args.tracker:
        cfg.override("tracker.type", args.tracker)
    if args.log_format:
        cfg.override("io.log_format", args.log_format)

    pipeline = MOTPipeline(cfg)
    summary = pipeline.run(args.input, args.output)

    print("\n=== Run Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()