from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ArrayLike = Sequence[Sequence[float]]


def _as_matrix(vectors: ArrayLike) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a 2D array-like object")
    return matrix


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return 1.0 - float(np.dot(a, b) / denom)


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)

    def entropy(x: np.ndarray) -> float:
        x = x[x > 0]
        return float(-np.sum(x * np.log2(x)))

    return entropy(m) - 0.5 * (entropy(p) + entropy(q))


@dataclass
class Occurrence:
    """A contextual embedding for one observed target-word occurrence."""

    vector: np.ndarray
    period: int
    occurrence_id: str
    cluster_id: Optional[int] = None


@dataclass
class Cluster:
    """A WiDiD sense cluster with a stable ID and update history."""

    cluster_id: int
    occurrence_ids: List[str] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    last_updated_period: int = 0

    def refresh(self, occurrences: Dict[str, Occurrence]) -> None:
        if not self.occurrence_ids:
            self.centroid = None
            return
        vectors = np.vstack([occurrences[item_id].vector for item_id in self.occurrence_ids])
        self.centroid = vectors.mean(axis=0)
        self.last_updated_period = max(occurrences[item_id].period for item_id in self.occurrence_ids)


@dataclass
class ShiftScores:
    jsd: float
    pdis: float
    pdiv: float


class SemiDynamicWiDiD:
    """Semi-dynamic WiDiD with thresholded historical cluster refresh.

    This keeps WiDiD's incremental behavior for ordinary updates. After
    ``historical_update_threshold`` new observations, it re-clusters retained
    history, matches the new clusters back to stable cluster IDs, and then
    continues incrementally.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.78,
        historical_update_threshold: int = 500,
        history_window: Optional[int] = None,
        min_cluster_fraction: float = 0.0,
        max_cluster_age: Optional[int] = None,
        ap_preference_quantile: float = 50.0,
        ap_damping: float = 0.9,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if historical_update_threshold < 1:
            raise ValueError("historical_update_threshold must be >= 1")
        if history_window is not None and history_window < 1:
            raise ValueError("history_window must be >= 1 when provided")
        if not 0.0 <= min_cluster_fraction <= 1.0:
            raise ValueError("min_cluster_fraction must be between 0 and 1")
        if not 0.0 <= ap_preference_quantile <= 100.0:
            raise ValueError("ap_preference_quantile must be between 0 and 100")
        if not 0.5 <= ap_damping < 1.0:
            raise ValueError("ap_damping must be >= 0.5 and < 1.0")

        self.similarity_threshold = similarity_threshold
        self.historical_update_threshold = historical_update_threshold
        self.history_window = history_window
        self.min_cluster_fraction = min_cluster_fraction
        self.max_cluster_age = max_cluster_age
        self.ap_preference_quantile = ap_preference_quantile
        self.ap_damping = ap_damping

        self.period = 0
        self._next_cluster_id = 0
        self._next_occurrence_id = 0
        self._updates_since_refresh = 0
        self.refresh_count = 0
        self.occurrences: Dict[str, Occurrence] = {}
        self.clusters: Dict[int, Cluster] = {}

    @property
    def updates_since_refresh(self) -> int:
        return self._updates_since_refresh

    def partial_fit(self, vectors: ArrayLike, period: Optional[int] = None) -> List[int]:
        """Add a new time slice with the APP incremental clustering algorithm."""
        matrix = _as_matrix(vectors)
        if period is None:
            self.period += 1
        else:
            self.period = period

        labels = self._app_update(matrix)

        self._updates_since_refresh += len(labels)
        self._trim_clusters()

        if self._updates_since_refresh >= self.historical_update_threshold:
            self.historical_refresh()

        return labels

    def historical_refresh(self) -> None:
        """Re-cluster retained history with AP and reconcile stable cluster IDs."""
        retained_ids = self._retained_occurrence_ids()
        if not retained_ids:
            return

        matrix = np.vstack([self.occurrences[item_id].vector for item_id in retained_ids])
        new_groups = self._cluster_from_scratch(matrix, retained_ids)
        old_centroids = {
            cluster_id: cluster.centroid
            for cluster_id, cluster in self.clusters.items()
            if cluster.centroid is not None
        }

        assignment = self._match_new_groups_to_old_ids(new_groups, old_centroids)
        refreshed: Dict[int, Cluster] = {}

        for group_idx, occurrence_ids in enumerate(new_groups):
            cluster_id = assignment[group_idx]
            for occurrence_id in occurrence_ids:
                self.occurrences[occurrence_id].cluster_id = cluster_id
            cluster = Cluster(cluster_id=cluster_id, occurrence_ids=list(occurrence_ids))
            cluster.refresh(self.occurrences)
            refreshed[cluster_id] = cluster

        self.clusters = refreshed
        self._trim_clusters()
        self._updates_since_refresh = 0
        self.refresh_count += 1

    def score_shift(self, past_periods: Iterable[int], current_periods: Iterable[int]) -> ShiftScores:
        """Compute WiDiD-style JSD, PDIS, and PDIV between two period groups."""
        past = set(past_periods)
        current = set(current_periods)
        cluster_ids = sorted(self.clusters)

        past_counts = []
        current_counts = []
        past_sense_prototypes = []
        current_sense_prototypes = []

        for cluster_id in cluster_ids:
            cluster = self.clusters[cluster_id]
            past_vectors = [
                self.occurrences[item_id].vector
                for item_id in cluster.occurrence_ids
                if self.occurrences[item_id].period in past
            ]
            current_vectors = [
                self.occurrences[item_id].vector
                for item_id in cluster.occurrence_ids
                if self.occurrences[item_id].period in current
            ]

            past_counts.append(len(past_vectors))
            current_counts.append(len(current_vectors))
            if past_vectors:
                past_sense_prototypes.append(np.vstack(past_vectors).mean(axis=0))
            if current_vectors:
                current_sense_prototypes.append(np.vstack(current_vectors).mean(axis=0))

        jsd = js_divergence(past_counts, current_counts)
        pdis, pdiv = self._prototype_scores(past_sense_prototypes, current_sense_prototypes)
        return ShiftScores(jsd=jsd, pdis=pdis, pdiv=pdiv)

    def snapshot(self) -> List[dict]:
        """Return a compact, inspectable view of the current sense memory."""
        rows = []
        for cluster_id in sorted(self.clusters):
            cluster = self.clusters[cluster_id]
            periods = [self.occurrences[item_id].period for item_id in cluster.occurrence_ids]
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "size": len(cluster.occurrence_ids),
                    "first_period": min(periods) if periods else None,
                    "last_period": max(periods) if periods else None,
                    "last_updated_period": cluster.last_updated_period,
                }
            )
        return rows

    def cluster_type_counts(
        self,
        past_periods: Iterable[int],
        current_periods: Iterable[int],
    ) -> Dict[str, int]:
        past = set(past_periods)
        current = set(current_periods)
        counts = {"c1_only_clusters": 0, "c2_only_clusters": 0, "mixed_clusters": 0}

        for cluster in self.clusters.values():
            has_past = False
            has_current = False
            for occurrence_id in cluster.occurrence_ids:
                period = self.occurrences[occurrence_id].period
                has_past = has_past or period in past
                has_current = has_current or period in current
            if has_past and has_current:
                counts["mixed_clusters"] += 1
            elif has_past:
                counts["c1_only_clusters"] += 1
            elif has_current:
                counts["c2_only_clusters"] += 1

        return counts

    def _app_update(self, matrix: np.ndarray) -> List[int]:
        new_occurrence_ids = []
        for vector in matrix:
            occurrence_id = self._new_occurrence_id()
            self.occurrences[occurrence_id] = Occurrence(
                vector=np.asarray(vector, dtype=float),
                period=self.period,
                occurrence_id=occurrence_id,
            )
            new_occurrence_ids.append(occurrence_id)

        if not self.clusters:
            new_labels = self._affinity_labels(matrix)
            return self._rebuild_from_label_groups(new_occurrence_ids, new_labels)

        old_cluster_ids = [
            cluster_id
            for cluster_id, cluster in sorted(self.clusters.items())
            if cluster.centroid is not None and cluster.occurrence_ids
        ]
        packed_old = np.vstack([self.clusters[cluster_id].centroid for cluster_id in old_cluster_ids])
        packed_matrix = np.vstack([packed_old, matrix])
        packed_labels = self._affinity_labels(packed_matrix)

        old_centroid_labels = packed_labels[: len(old_cluster_ids)]
        new_vector_labels = packed_labels[len(old_cluster_ids) :]
        label_to_cluster_id: Dict[int, int] = {}
        rebuilt: Dict[int, Cluster] = {}

        for old_cluster_id, label in zip(old_cluster_ids, old_centroid_labels):
            label = int(label)
            stable_id = label_to_cluster_id.setdefault(label, old_cluster_id)
            rebuilt.setdefault(stable_id, Cluster(cluster_id=stable_id))
            rebuilt[stable_id].occurrence_ids.extend(self.clusters[old_cluster_id].occurrence_ids)

        returned_labels: List[int] = []
        for occurrence_id, label in zip(new_occurrence_ids, new_vector_labels):
            label = int(label)
            cluster_id = label_to_cluster_id.get(label)
            if cluster_id is None:
                cluster_id = self._next_cluster_id
                self._next_cluster_id += 1
                label_to_cluster_id[label] = cluster_id
                rebuilt[cluster_id] = Cluster(cluster_id=cluster_id)
            rebuilt.setdefault(cluster_id, Cluster(cluster_id=cluster_id))
            rebuilt[cluster_id].occurrence_ids.append(occurrence_id)
            returned_labels.append(cluster_id)

        self.clusters = rebuilt
        for cluster_id, cluster in self.clusters.items():
            for occurrence_id in cluster.occurrence_ids:
                self.occurrences[occurrence_id].cluster_id = cluster_id
            cluster.refresh(self.occurrences)

        return returned_labels

    def _affinity_labels(self, matrix: np.ndarray) -> np.ndarray:
        if len(matrix) == 0:
            return np.asarray([], dtype=int)
        if len(matrix) == 1:
            return np.asarray([0], dtype=int)

        try:
            from sklearn.cluster import AffinityPropagation
            from sklearn.metrics.pairwise import cosine_similarity

            similarity = cosine_similarity(matrix)
            preference = float(np.percentile(similarity, self.ap_preference_quantile))
            model = AffinityPropagation(
                affinity="precomputed",
                damping=self.ap_damping,
                preference=preference,
                random_state=13,
            )
            labels = model.fit_predict(similarity)
            if np.any(labels < 0):
                raise ValueError("Affinity Propagation did not converge to valid labels")
            return labels.astype(int)
        except Exception:
            return self._threshold_labels(matrix)

    def _threshold_labels(self, matrix: np.ndarray) -> np.ndarray:
        groups: List[List[int]] = []
        centroids: List[np.ndarray] = []
        labels = np.empty(len(matrix), dtype=int)

        for idx, vector in enumerate(matrix):
            if not centroids:
                groups.append([idx])
                centroids.append(vector.copy())
                labels[idx] = 0
                continue

            centroid_matrix = np.vstack(centroids)
            sims = _normalize(centroid_matrix) @ _normalize(vector.reshape(1, -1)).ravel()
            best_idx = int(np.argmax(sims))
            if float(sims[best_idx]) >= self.similarity_threshold:
                groups[best_idx].append(idx)
                centroids[best_idx] = matrix[groups[best_idx]].mean(axis=0)
                labels[idx] = best_idx
            else:
                labels[idx] = len(groups)
                groups.append([idx])
                centroids.append(vector.copy())

        return labels

    def _rebuild_from_label_groups(self, occurrence_ids: List[str], labels: np.ndarray) -> List[int]:
        label_to_cluster_id: Dict[int, int] = {}
        rebuilt: Dict[int, Cluster] = {}
        returned_labels: List[int] = []

        for occurrence_id, label in zip(occurrence_ids, labels):
            label = int(label)
            cluster_id = label_to_cluster_id.get(label)
            if cluster_id is None:
                cluster_id = self._new_cluster_id()
                label_to_cluster_id[label] = cluster_id
                rebuilt[cluster_id] = self.clusters[cluster_id]
            rebuilt[cluster_id].occurrence_ids.append(occurrence_id)
            returned_labels.append(cluster_id)

        self.clusters = rebuilt
        for cluster_id, cluster in self.clusters.items():
            for occurrence_id in cluster.occurrence_ids:
                self.occurrences[occurrence_id].cluster_id = cluster_id
            cluster.refresh(self.occurrences)

        return returned_labels

    def _cluster_from_scratch(self, matrix: np.ndarray, occurrence_ids: List[str]) -> List[List[str]]:
        labels = self._affinity_labels(matrix)
        groups_by_label: Dict[int, List[str]] = {}
        for occurrence_id, label in zip(occurrence_ids, labels):
            groups_by_label.setdefault(int(label), []).append(occurrence_id)
        return [groups_by_label[label] for label in sorted(groups_by_label)]

    def _match_new_groups_to_old_ids(
        self,
        new_groups: List[List[str]],
        old_centroids: Dict[int, np.ndarray],
    ) -> Dict[int, int]:
        unmatched_old = set(old_centroids)
        assignment: Dict[int, int] = {}

        candidates: List[Tuple[float, int, int]] = []
        for group_idx, occurrence_ids in enumerate(new_groups):
            group_centroid = np.vstack(
                [self.occurrences[item_id].vector for item_id in occurrence_ids]
            ).mean(axis=0)
            for old_id, old_centroid in old_centroids.items():
                similarity = 1.0 - cosine_distance(group_centroid, old_centroid)
                candidates.append((similarity, group_idx, old_id))

        for similarity, group_idx, old_id in sorted(candidates, reverse=True):
            if group_idx in assignment or old_id not in unmatched_old:
                continue
            if similarity >= self.similarity_threshold:
                assignment[group_idx] = old_id
                unmatched_old.remove(old_id)

        for group_idx in range(len(new_groups)):
            if group_idx not in assignment:
                assignment[group_idx] = self._new_cluster_id()

        return assignment

    def _prototype_scores(
        self,
        past_sense_prototypes: List[np.ndarray],
        current_sense_prototypes: List[np.ndarray],
    ) -> Tuple[float, float]:
        if not past_sense_prototypes or not current_sense_prototypes:
            return 1.0, 1.0

        past_word_prototype = np.vstack(past_sense_prototypes).mean(axis=0)
        current_word_prototype = np.vstack(current_sense_prototypes).mean(axis=0)
        pdis = cosine_distance(past_word_prototype, current_word_prototype)

        past_div = np.mean(
            [cosine_distance(proto, past_word_prototype) for proto in past_sense_prototypes]
        )
        current_div = np.mean(
            [cosine_distance(proto, current_word_prototype) for proto in current_sense_prototypes]
        )
        pdiv = abs(float(past_div) - float(current_div))
        return pdis, pdiv

    def _trim_clusters(self) -> None:
        if not self.clusters:
            return

        total = sum(len(cluster.occurrence_ids) for cluster in self.clusters.values())
        min_size = int(np.ceil(total * self.min_cluster_fraction))
        retained: Dict[int, Cluster] = {}

        for cluster_id, cluster in self.clusters.items():
            if min_size and len(cluster.occurrence_ids) < min_size:
                self._drop_occurrences(cluster.occurrence_ids)
                continue
            if self.max_cluster_age is not None:
                age = self.period - cluster.last_updated_period
                if age > self.max_cluster_age:
                    self._drop_occurrences(cluster.occurrence_ids)
                    continue
            retained[cluster_id] = cluster

        self.clusters = retained

    def _retained_occurrence_ids(self) -> List[str]:
        ids = sorted(
            self.occurrences,
            key=lambda item_id: (self.occurrences[item_id].period, item_id),
        )
        if self.history_window is not None:
            ids = ids[-self.history_window :]
            keep = set(ids)
            drop = [item_id for item_id in self.occurrences if item_id not in keep]
            self._drop_occurrences(drop)
        return ids

    def _drop_occurrences(self, occurrence_ids: Iterable[str]) -> None:
        for occurrence_id in list(occurrence_ids):
            self.occurrences.pop(occurrence_id, None)

    def _new_cluster_id(self) -> int:
        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1
        self.clusters[cluster_id] = Cluster(cluster_id=cluster_id)
        return cluster_id

    def _new_occurrence_id(self) -> str:
        occurrence_id = f"occ-{self._next_occurrence_id}"
        self._next_occurrence_id += 1
        return occurrence_id


class IncrementalWiDiD(SemiDynamicWiDiD):
    """Original WiDiD-style baseline without historical re-clustering."""

    def __init__(
        self,
        similarity_threshold: float = 0.78,
        min_cluster_fraction: float = 0.0,
        max_cluster_age: Optional[int] = None,
        ap_preference_quantile: float = 50.0,
        ap_damping: float = 0.9,
    ) -> None:
        super().__init__(
            similarity_threshold=similarity_threshold,
            historical_update_threshold=10**18,
            history_window=None,
            min_cluster_fraction=min_cluster_fraction,
            max_cluster_age=max_cluster_age,
            ap_preference_quantile=ap_preference_quantile,
            ap_damping=ap_damping,
        )

    def historical_refresh(self) -> None:
        return None
