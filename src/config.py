import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    raw: dict

    @classmethod
    def load(cls, path: str) -> "Config":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(p, "r") as f:
            data = yaml.safe_load(f)
        return cls(raw=data)

    def get(self, dotted_key: str, default=None):
        node = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    def require(self, dotted_key: str):
        """Like get(), but raises if the key is missing — use for values
        that must never silently fall back to None (e.g. model weights)."""
        sentinel = object()
        val = self.get(dotted_key, sentinel)
        if val is sentinel:
            raise KeyError(f"Required config key missing: '{dotted_key}'")
        return val
    def override(self, dotted_key: str, value):
        """Allow CLI args to override config values at runtime."""
        parts = dotted_key.split(".")
        node = self.raw
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value