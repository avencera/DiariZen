"""Recording-first mixture sampling and committed coverage acknowledgement."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .contracts import (
    CYCLE_EXAMPLES,
    DYNAMIC_STREAM,
    EFFECTIVE_BATCH,
    FINITE_STREAMS,
    STREAM_QUOTAS,
    UPDATES_PER_CYCLE,
    CoverageState,
    RecordingRow,
)
from .errors import ContractError


def _stable_shuffle(values: Sequence[str], seed: int, salt: str) -> list[str]:
    """Return a deterministic shuffle of parent IDs."""

    keyed = []
    for value in values:
        digest = hashlib.sha256(f"{seed}:{salt}:{value}".encode("utf-8")).hexdigest()
        keyed.append((digest, value))
    keyed.sort()
    return [value for _, value in keyed]


@dataclass
class StreamCursor:
    """Cursor over one finite stream's parent-first then chunk-shuffle schedule."""

    corpus: str
    parents: tuple[str, ...]
    unseen_parents: list[str]
    chunk_queue: list[str]
    chunk_index: int = 0
    parent_cycles: int = 0

    def state_dict(self) -> dict[str, object]:
        """Return a JSON-serializable cursor."""

        return {
            "corpus": self.corpus,
            "parents": list(self.parents),
            "unseen_parents": list(self.unseen_parents),
            "chunk_queue": list(self.chunk_queue),
            "chunk_index": self.chunk_index,
            "parent_cycles": self.parent_cycles,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> StreamCursor:
        """Restore a cursor."""

        return cls(
            corpus=str(payload["corpus"]),
            parents=tuple(payload["parents"]),
            unseen_parents=list(payload["unseen_parents"]),
            chunk_queue=list(payload["chunk_queue"]),
            chunk_index=int(payload["chunk_index"]),
            parent_cycles=int(payload["parent_cycles"]),
        )

    def propose_parent(self, chunks_by_parent: Mapping[str, Sequence[str]], rng: random.Random) -> str:
        """Return the next parent ID without acknowledging coverage."""

        if self.unseen_parents:
            parent = self.unseen_parents[0]
            return parent
        if not self.chunk_queue:
            self._refill_chunks(chunks_by_parent, rng)
        if not self.chunk_queue:
            raise ContractError(f"stream {self.corpus} has no eligible chunks")
        chunk_id = self.chunk_queue[self.chunk_index]
        return chunk_id.rsplit(":", 1)[0]

    def consume_parent(self, parent_id: str, chunks_by_parent: Mapping[str, Sequence[str]], rng: random.Random) -> str:
        """Advance the cursor after a proposed example is built."""

        if self.unseen_parents and self.unseen_parents[0] == parent_id:
            self.unseen_parents.pop(0)
            chunks = list(chunks_by_parent.get(parent_id, ()))
            if not chunks:
                raise ContractError(f"parent {parent_id} has no valid chunks")
            return rng.choice(chunks)
        if not self.chunk_queue:
            self._refill_chunks(chunks_by_parent, rng)
        chunk_id = self.chunk_queue[self.chunk_index]
        self.chunk_index += 1
        if self.chunk_index >= len(self.chunk_queue):
            self.chunk_queue = []
            self.chunk_index = 0
        return chunk_id

    def _refill_chunks(self, chunks_by_parent: Mapping[str, Sequence[str]], rng: random.Random) -> None:
        chunks: list[str] = []
        for parent in self.parents:
            chunks.extend(chunks_by_parent.get(parent, ()))
        rng.shuffle(chunks)
        self.chunk_queue = chunks
        self.chunk_index = 0
        self.parent_cycles += 1
        self.unseen_parents = []


@dataclass
class MixtureSampler:
    """Quota-driven sampler. Prefetch cannot commit coverage."""

    seed: int
    cursors: dict[str, StreamCursor]
    coverage: CoverageState
    rng_state: tuple | None = None
    proposed: int = 0
    committed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.rng_state is not None:
            self._rng.setstate(self.rng_state)

    @classmethod
    def from_rows(cls, rows: Iterable[RecordingRow], seed: int) -> MixtureSampler:
        """Build cursors from accepted training rows only."""

        parents_by_corpus: dict[str, set[str]] = {name: set() for name in FINITE_STREAMS}
        for row in rows:
            if not row.can_sample():
                continue
            if row.corpus not in parents_by_corpus:
                raise ContractError("unknown training corpus", {"corpus": row.corpus})
            parents_by_corpus[row.corpus].add(row.parent_id)
        missing = [name for name, parents in parents_by_corpus.items() if not parents]
        if missing:
            raise ContractError("missing required training corpus", {"corpora": missing})
        cursors = {}
        denominators = {}
        for corpus, parents in parents_by_corpus.items():
            ordered = tuple(_stable_shuffle(sorted(parents), seed, f"parent:{corpus}"))
            cursors[corpus] = StreamCursor(corpus, ordered, list(ordered), [])
            denominators[corpus] = len(ordered)
        coverage = CoverageState(seen={corpus: set() for corpus in FINITE_STREAMS}, denominators=denominators)
        return cls(seed=seed, cursors=cursors, coverage=coverage)

    def state_dict(self) -> dict[str, object]:
        """Return committed sampler and coverage state."""

        version, internal, gauss = self._rng.getstate()
        return {
            "seed": self.seed,
            "cursors": {name: cursor.state_dict() for name, cursor in self.cursors.items()},
            "coverage": self.coverage.state_dict(),
            "rng_state": [version, list(internal), gauss],
            "proposed": self.proposed,
            "committed": self.committed,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, object]) -> MixtureSampler:
        """Restore committed sampler state."""

        cursors = {name: StreamCursor.from_state_dict(value) for name, value in dict(payload["cursors"]).items()}
        coverage = CoverageState.from_state_dict(dict(payload["coverage"]))
        raw_rng = list(payload["rng_state"])
        rng_state = (int(raw_rng[0]), tuple(int(value) for value in raw_rng[1]), raw_rng[2])
        sampler = cls(
            seed=int(payload["seed"]),
            cursors=cursors,
            coverage=coverage,
            rng_state=rng_state,
            proposed=int(payload["proposed"]),
            committed=int(payload["committed"]),
        )
        return sampler

    def choose_stream(self) -> str:
        """Select a stream by exact quota mass. Dynamic has no finite denominator."""

        draw = self._rng.random()
        cumulative = 0.0
        for name, mass in STREAM_QUOTAS.items():
            cumulative += mass
            if draw < cumulative:
                return name
        return DYNAMIC_STREAM

    def propose_example(
        self,
        chunks_by_parent: Mapping[str, Mapping[str, Sequence[str]]],
    ) -> dict[str, object]:
        """Propose one example. Does not advance coverage."""

        stream = self.choose_stream()
        self.proposed += 1
        if stream == DYNAMIC_STREAM:
            return {
                "stream": stream,
                "parent_id": None,
                "chunk_id": None,
                "dynamic": True,
                "counts_for_coverage": False,
            }
        cursor = self.cursors[stream]
        parent_id = cursor.propose_parent(chunks_by_parent[stream], self._rng)
        chunk_id = cursor.consume_parent(parent_id, chunks_by_parent[stream], self._rng)
        return {
            "stream": stream,
            "parent_id": parent_id,
            "chunk_id": chunk_id,
            "dynamic": False,
            "counts_for_coverage": True,
        }

    def acknowledge(
        self,
        example: Mapping[str, object],
        *,
        optimizer_updated: bool,
        loss_weight: float,
        valid: bool,
    ) -> bool:
        """Commit coverage only for a successful optimizer update with nonzero weight."""

        if not optimizer_updated or not valid or loss_weight <= 0:
            return False
        if example.get("dynamic") or not example.get("counts_for_coverage"):
            self.committed += 1
            return False
        corpus = str(example["stream"])
        parent_id = str(example["parent_id"])
        self.coverage.seen.setdefault(corpus, set()).add(parent_id)
        self.committed += 1
        return True


def coverage_plan(denominators: Mapping[str, int]) -> dict[str, object]:
    """Compute the cycle schedule required to cover each finite stream."""

    streams = {}
    slowest_cycles = 0
    for corpus in FINITE_STREAMS:
        count = int(denominators[corpus])
        quota = STREAM_QUOTAS[corpus]
        examples_per_cycle = CYCLE_EXAMPLES * quota
        cycles_needed = 0 if examples_per_cycle <= 0 else int((count + examples_per_cycle - 1) // examples_per_cycle)
        streams[corpus] = {
            "denominator": count,
            "quota": quota,
            "examples_per_cycle": examples_per_cycle,
            "cycles_to_cover_parents": max(cycles_needed, 1 if count else 0),
            "updates_per_cycle": UPDATES_PER_CYCLE,
            "effective_batch": EFFECTIVE_BATCH,
        }
        slowest_cycles = max(slowest_cycles, streams[corpus]["cycles_to_cover_parents"])
    return {
        "streams": streams,
        "slowest_cycles": slowest_cycles,
        "dynamic": {"denominator": None, "quota": STREAM_QUOTAS[DYNAMIC_STREAM]},
    }


def schedule_batch_streams(n_examples: int, seed: int) -> list[str]:
    """Return the exact stream sequence for n proposed examples."""

    rng = random.Random(seed)
    names = list(STREAM_QUOTAS)
    masses = [STREAM_QUOTAS[name] for name in names]
    sequence = []
    for _ in range(n_examples):
        draw = rng.random()
        cumulative = 0.0
        chosen = names[-1]
        for name, mass in zip(names, masses):
            cumulative += mass
            if draw < cumulative:
                chosen = name
                break
        sequence.append(chosen)
    return sequence
