from typing import Iterable

class UniqueVisitorCounter:
    def __init__(self):
        self._seen_ids = set()

    def update(self, ids: Iterable[int]) -> int:
        self._seen_ids.update(ids)
        return self.total

    @property
    def total(self) -> int:
        return len(self._seen_ids)