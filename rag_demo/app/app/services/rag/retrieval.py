from __future__ import annotations

"""Dispatcher for RAG retrieval strategies."""

from typing import Any, Dict, List, Optional

from .strategies import BaselineStrategy, ChromaOnlyStrategy, RetrievalStrategy

_STRATEGIES: Dict[str, RetrievalStrategy] = {
    "baseline": BaselineStrategy(),
    "chroma_only": ChromaOnlyStrategy(),
    "multiq": ChromaOnlyStrategy(),
}


def retrieve_docs(
    self,
    q: str,
    *,
    k: int = 6,
    where: Optional[Dict[str, Any]] = None,
    candidate_k: Optional[int] = None,
    use_mmr: bool = True,
    lam: float = 0.5,
    strategy: str = "baseline",
) -> List[Dict[str, Any]]:
    strat = _STRATEGIES.get(strategy)
    if not strat:
        raise ValueError(f"unknown strategy: {strategy}")
    return strat.retrieve(
        self,
        q,
        k=k,
        where=where,
        candidate_k=candidate_k,
        use_mmr=use_mmr,
        lam=lam,
    )


__all__ = ["retrieve_docs"]
