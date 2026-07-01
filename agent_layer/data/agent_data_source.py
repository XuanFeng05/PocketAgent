from __future__ import annotations

from pathlib import Path
from typing import Iterable

from agent_layer.data.feature_parts import FeaturePartsDataset
from agent_layer.data.single_symbol_reader import (
    CacheBackedAgentTimelineLoader,
    is_agent_cache_dir,
)
from agent_layer.data.timeline import AgentTimelineLoader


def open_agent_timeline_loader(
    store_path: str | Path,
    *,
    universe: Iterable[str] | None = None,
    frequencies: Iterable[str] | None = None,
    validate_store: bool = True,
    cache_size: int = 8192,
    stream_chunk_size: int = 64,
    use_market_cache: bool = True,
    use_decision_cache: bool = True,
    market_cache_workers: int | None = None,
    market_cache_progress=None,
    decision_cache_progress=None,
):
    """Open either an Agent tensor cache or a Feature Parts dataset.

    Feature Parts remain the canonical build artifact, but Agent training can
    now point ``store_path`` at ``runtime_layer/agent_cache/<dataset>`` to use
    the cache-backed hot path without changing the environment/trainer API.
    """
    path = Path(store_path)
    if is_agent_cache_dir(path):
        return CacheBackedAgentTimelineLoader(
            path,
            universe=universe,
            frequencies=frequencies,
            validate_store=validate_store,
            cache_size=cache_size,
            stream_chunk_size=stream_chunk_size,
            use_market_cache=use_market_cache,
            use_decision_cache=use_decision_cache,
            market_cache_workers=market_cache_workers,
            market_cache_progress=market_cache_progress,
            decision_cache_progress=decision_cache_progress,
        )
    return AgentTimelineLoader(
        path,
        universe=universe,
        frequencies=frequencies,
        validate_store=validate_store,
        cache_size=cache_size,
        stream_chunk_size=stream_chunk_size,
        use_market_cache=use_market_cache,
        use_decision_cache=use_decision_cache,
        market_cache_workers=market_cache_workers,
        market_cache_progress=market_cache_progress,
        decision_cache_progress=decision_cache_progress,
    )


def agent_store_kind(store_path: str | Path) -> str:
    path = Path(store_path)
    if is_agent_cache_dir(path):
        return "agent_cache"
    if FeaturePartsDataset.maybe(path) is not None:
        return "feature_parts"
    return "unknown"
