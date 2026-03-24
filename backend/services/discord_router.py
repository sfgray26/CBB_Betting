"""
DiscordRouter: rate-limited routing layer (ADR-005).
Delegates HTTP delivery to discord_notifier.send_to_channel().
Rate limit: 1 msg/channel/60s. Batch: flush at 5 items or 300s age.
"""
from __future__ import annotations
import dataclasses
import logging
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

_RATE_LIMIT_SECONDS = 60
_BATCH_FLUSH_COUNT = 5
_BATCH_FLUSH_AGE = 300


@dataclasses.dataclass
class IntelPackage:
    channel: str
    message: str
    priority: int = 5          # 1=high, 10=low
    embed: Optional[dict] = None
    mention_admin: bool = False


class DiscordRouter:
    def __init__(self):
        self._last_sent: dict[str, float] = defaultdict(float)
        self._batch_queue: list[IntelPackage] = []
        self._batch_queue_ts: float = 0.0

    def route(self, pkg: IntelPackage) -> bool:
        if pkg.priority <= 3:
            return self._deliver(pkg)
        if time.monotonic() - self._last_sent[pkg.channel] >= _RATE_LIMIT_SECONDS:
            return self._deliver(pkg)
        self._enqueue(pkg)
        return False

    def flush_batch(self) -> int:
        if not self._batch_queue:
            return 0
        self._batch_queue.sort(key=lambda p: p.priority)
        sent = 0
        channels: dict[str, list[IntelPackage]] = defaultdict(list)
        for pkg in self._batch_queue:
            channels[pkg.channel].append(pkg)
        self._batch_queue.clear()
        self._batch_queue_ts = 0.0
        from backend.services.discord_notifier import send_to_channel
        for channel, pkgs in channels.items():
            try:
                send_to_channel(channel, "\n---\n".join(p.message for p in pkgs))
                sent += len(pkgs)
            except Exception as e:
                logger.error("DiscordRouter flush failed for %s: %s", channel, e)
        return sent

    def should_flush(self):
        if not self._batch_queue:
            return False
        if len(self._batch_queue) >= _BATCH_FLUSH_COUNT:
            return True
        return time.monotonic() - self._batch_queue_ts >= _BATCH_FLUSH_AGE

    def _deliver(self, pkg: IntelPackage) -> bool:
        from backend.services.discord_notifier import send_to_channel
        try:
            send_to_channel(pkg.channel, pkg.message, embed=pkg.embed, mention_admin=pkg.mention_admin)
            self._last_sent[pkg.channel] = time.monotonic()
            return True
        except Exception as e:
            logger.error("DiscordRouter delivery failed: %s", e)
            return False

    def _enqueue(self, pkg: IntelPackage):
        if not self._batch_queue:
            self._batch_queue_ts = time.monotonic()
        self._batch_queue.append(pkg)
