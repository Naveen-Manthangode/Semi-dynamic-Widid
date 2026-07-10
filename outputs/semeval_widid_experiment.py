from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from semi_dynamic_widid import IncrementalWiDiD, SemiDynamicWiDiD


TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
POS_SUFFIX_RE = re.compile(r"^(.+?)[_.-](nn|noun|vb|verb|v|adj|a|adv|r)$", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def target_aliases(target: str) -> Set[str]:
    """Return common SemEval token variants for a target word.

    SemEval lexical-change files may use plain lemmas in one file and POS-tagged
    forms such as ``word_nn`` or ``word_vb`` in another. Keeping aliases lets the
    runner find occurrences without requiring the user to rewrite the data.
    """

    target = target.lower().strip()
    aliases = {target}
    match = POS_SUFFIX_RE.match(target)
    if match:
        aliases.add(match.group(1))
    else:
        for suffix in ["nn", "vb", "adj", "adv", "noun", "verb", "a", "v", "r"]:
            aliases.add(f"{target}_{suffix}")
            aliases.add(f"{target}.{suffix}")
            aliases.add(f"{target}-{suffix}")
    return aliases


def stable_index(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector
    return vector / norm


def hash_context_embedding(tokens: Sequence[str], target_index: int, window: int, dim: int) -> np.ndarray:

    vector = np.zeros(dim, dtype=float)
    start = max(0, target_index - window)
    end = min(len(tokens), target_index + window + 1)

    for pos in range(start, end):
        if pos == target_index:
            continue
        distance = abs(pos - target_index)
        weight = 1.0 / max(distance, 1)
        vector[stable_index(tokens[pos], dim)] += weight

    if np.linalg.norm(vector) == 0.0:
        vector[stable_index(tokens[target_index], dim)] = 1.0
    return normalize_vector(vector)


class HashEmbeddingBackend:
    def __init__(self, window: int, dim: int) -> None:
        self.window = window
        self.dim = dim

    def extract_document(
        self,
        tokens: Sequence[str],
        alias_to_target: Dict[str, str],
        vectors: Dict[str, List[np.ndarray]],
        max_occurrences_per_target: Optional[int],
    ) -> None:
        for index, token in enumerate(tokens):
            target = alias_to_target.get(token)
            if target is None:
                continue
            if reached_limit(vectors, target, max_occurrences_per_target):
                continue
            vectors[target].append(hash_context_embedding(tokens, index, self.window, self.dim))


class BertEmbeddingBackend:
    def __init__(
        self,
        model_name: str,
        window: int,
        device: Optional[str],
        max_length: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Needed Transformers install torch and transformers for bert"
            ) from exc

        self.torch = torch
        self.window = window
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.hidden_size)

    def extract_document(
        self,
        tokens: Sequence[str],
        alias_to_target: Dict[str, str],
        vectors: Dict[str, List[np.ndarray]],
        max_occurrences_per_target: Optional[int],
    ) -> None:
        for index, token in enumerate(tokens):
            target = alias_to_target.get(token)
            if target is None:
                continue
            if reached_limit(vectors, target, max_occurrences_per_target):
                continue
            embedding = self._embed_occurrence(tokens, index)
            vectors[target].append(embedding)

    def _embed_occurrence(self, tokens: Sequence[str], target_index: int) -> np.ndarray:
        start = max(0, target_index - self.window)
        end = min(len(tokens), target_index + self.window + 1)
        context_tokens = list(tokens[start:end])
        local_target_index = target_index - start

        encoded = self.tokenizer(
            context_tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = encoded.word_ids(batch_index=0)
        target_piece_positions = [
            pos for pos, word_id in enumerate(word_ids) if word_id == local_target_index
        ]
        if not target_piece_positions:
            return hash_context_embedding(tokens, target_index, self.window, self.dim)

        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            output = self.model(**model_inputs)
            hidden = self.torch.stack(output.hidden_states[-4:]).sum(dim=0)[0]
            piece_vectors = hidden[target_piece_positions]
            vector = piece_vectors.mean(dim=0).detach().cpu().numpy()

        return normalize_vector(vector.astype(float))


class Doc2VecEmbeddingBackend:
    def __init__(
        self,
        corpus_paths: Sequence[Path],
        vector_size: int,
        window: int,
        epochs: int,
        min_count: int,
    ) -> None:
        try:
            from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        except ImportError as exc:
            raise RuntimeError(
                "Doc2Vec embeddings require gensim. Install it with: pip install gensim"
            ) from exc

        tagged_docs = []
        self.docs_by_path: Dict[str, List[Tuple[str, List[str]]]] = {}

        for corpus_path in corpus_paths:
            path_docs: List[Tuple[str, List[str]]] = []
            for text in iter_texts(corpus_path):
                tokens = tokenize(text)
                if not tokens:
                    continue
                tag = f"doc-{len(tagged_docs)}"
                tagged_docs.append(TaggedDocument(tokens, [tag]))
                path_docs.append((tag, tokens))
            self.docs_by_path[str(corpus_path.resolve())] = path_docs

        if not tagged_docs:
            raise ValueError("Doc2Vec could not find any documents to train on")

        model = Doc2Vec(
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=1,
            epochs=epochs,
            seed=13,
        )
        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
        self.model = model

    def iter_precomputed_docs(self, corpus_path: Path) -> Iterable[Tuple[List[str], np.ndarray]]:
        for tag, tokens in self.docs_by_path.get(str(corpus_path.resolve()), []):
            yield tokens, normalize_vector(np.asarray(self.model.dv[tag], dtype=float))

    def extract_document(
        self,
        tokens: Sequence[str],
        alias_to_target: Dict[str, str],
        vectors: Dict[str, List[np.ndarray]],
        max_occurrences_per_target: Optional[int],
        document_vector: np.ndarray,
    ) -> None:
        for token in tokens:
            target = alias_to_target.get(token)
            if target is None:
                continue
            if reached_limit(vectors, target, max_occurrences_per_target):
                continue
            vectors[target].append(document_vector.copy())


def reached_limit(
    vectors: Dict[str, List[np.ndarray]],
    target: str,
    max_occurrences_per_target: Optional[int],
) -> bool:
    return (
        max_occurrences_per_target is not None
        and len(vectors[target]) >= max_occurrences_per_target
    )


def read_targets(path: Path) -> List[str]:
    targets = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            targets.append(line.split()[0].lower())
    return targets


def read_gold(path: Optional[Path]) -> Tuple[Dict[str, float], int]:
    if path is None:
        return {}, 0

    gold: Dict[str, float] = {}
    base_entries = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = re.split(r"[\t,; ]+", line)
            row = [cell.strip() for cell in row if cell.strip()]
            if len(row) < 2:
                continue
            try:
                key = row[0].lower()
                value = float(row[-1])
            except ValueError:
                continue
            base_entries += 1
            for alias in target_aliases(key):
                gold[alias] = value
    return gold, base_entries


def iter_texts(path: Path) -> Iterable[str]:
    if path.is_file():
        yield path.read_text(encoding="utf-8", errors="ignore")
        return

    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and not file_path.name.startswith("."):
            yield file_path.read_text(encoding="utf-8", errors="ignore")


def make_alias_map(targets: Sequence[str]) -> Dict[str, str]:
    alias_to_target: Dict[str, str] = {}
    for target in targets:
        for alias in target_aliases(target):
            alias_to_target.setdefault(alias, target)
    return alias_to_target


def extract_target_embeddings(
    corpus_path: Path,
    targets: Sequence[str],
    backend,
    max_occurrences_per_target: Optional[int],
) -> Dict[str, np.ndarray]:
    alias_to_target = make_alias_map(targets)
    vectors: Dict[str, List[np.ndarray]] = {target: [] for target in targets}

    for text in iter_texts(corpus_path):
        tokens = tokenize(text)
        backend.extract_document(tokens, alias_to_target, vectors, max_occurrences_per_target)

    return {
        target: np.vstack(items) if items else np.empty((0, backend.dim), dtype=float)
        for target, items in vectors.items()
    }


def extract_doc2vec_target_embeddings(
    corpus_path: Path,
    targets: Sequence[str],
    backend: Doc2VecEmbeddingBackend,
    max_occurrences_per_target: Optional[int],
) -> Dict[str, np.ndarray]:
    alias_to_target = make_alias_map(targets)
    vectors: Dict[str, List[np.ndarray]] = {target: [] for target in targets}

    for tokens, document_vector in backend.iter_precomputed_docs(corpus_path):
        backend.extract_document(
            tokens,
            alias_to_target,
            vectors,
            max_occurrences_per_target,
            document_vector,
        )

    dim = backend.model.vector_size
    return {
        target: np.vstack(items) if items else np.empty((0, dim), dtype=float)
        for target, items in vectors.items()
    }


def occurrence_diagnostics(
    targets: Sequence[str],
    target_vectors_c1: Dict[str, np.ndarray],
    target_vectors_c2: Dict[str, np.ndarray],
    gold: Dict[str, float],
) -> List[dict]:
    rows = []
    for target in targets:
        c1_count = len(target_vectors_c1.get(target, []))
        c2_count = len(target_vectors_c2.get(target, []))
        rows.append(
            {
                "target": target,
                "occurrences_c1": c1_count,
                "occurrences_c2": c2_count,
                "has_both_periods": c1_count > 0 and c2_count > 0,
                "has_gold": target in gold,
            }
        )
    return rows


def split_batches(vectors: np.ndarray, batches: int) -> List[np.ndarray]:
    if batches < 1:
        raise ValueError("batches must be >= 1")
    if len(vectors) == 0:
        return []
    return [batch for batch in np.array_split(vectors, batches) if len(batch) > 0]


def spearmanr(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    if len(x) < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def rankdata(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def run_one_method(
    method_name: str,
    embedding_backend: str,
    target_vectors_c1: Dict[str, np.ndarray],
    target_vectors_c2: Dict[str, np.ndarray],
    similarity_threshold: float,
    historical_update_threshold: int,
    history_window: Optional[int],
    min_cluster_fraction: float,
    batches_per_corpus: int,
    ap_preference_quantile: float,
    ap_damping: float,
) -> Tuple[List[dict], float]:
    rows = []
    start = time.perf_counter()

    for target in sorted(target_vectors_c1):
        vectors_c1 = target_vectors_c1[target]
        vectors_c2 = target_vectors_c2[target]
        if vectors_c1.size == 0 or vectors_c2.size == 0:
            continue

        if method_name == "widid":
            model = IncrementalWiDiD(
                similarity_threshold=similarity_threshold,
                min_cluster_fraction=min_cluster_fraction,
                ap_preference_quantile=ap_preference_quantile,
                ap_damping=ap_damping,
            )
        else:
            model = SemiDynamicWiDiD(
                similarity_threshold=similarity_threshold,
                historical_update_threshold=historical_update_threshold,
                history_window=history_window,
                min_cluster_fraction=min_cluster_fraction,
                ap_preference_quantile=ap_preference_quantile,
                ap_damping=ap_damping,
            )

        c1_batches = split_batches(vectors_c1, batches_per_corpus)
        c2_batches = split_batches(vectors_c2, batches_per_corpus)
        c1_periods = list(range(1, len(c1_batches) + 1))
        c2_periods = list(range(len(c1_batches) + 1, len(c1_batches) + len(c2_batches) + 1))

        for period, batch in zip(c1_periods, c1_batches):
            model.partial_fit(batch, period=period)
        for period, batch in zip(c2_periods, c2_batches):
            model.partial_fit(batch, period=period)

        scores = model.score_shift(past_periods=c1_periods, current_periods=c2_periods)
        snapshot = model.snapshot()
        cluster_types = model.cluster_type_counts(
            past_periods=c1_periods,
            current_periods=c2_periods,
        )

        rows.append(
            {
                "target": target,
                "method": method_name,
                "embedding_backend": embedding_backend,
                "jsd": scores.jsd,
                "pdis": scores.pdis,
                "pdiv": scores.pdiv,
                "clusters": len(snapshot),
                "occurrences_c1": len(vectors_c1),
                "occurrences_c2": len(vectors_c2),
                "batches_c1": len(c1_batches),
                "batches_c2": len(c2_batches),
                "historical_refreshes": getattr(model, "refresh_count", 0),
                "c1_only_clusters": cluster_types["c1_only_clusters"],
                "c2_only_clusters": cluster_types["c2_only_clusters"],
                "mixed_clusters": cluster_types["mixed_clusters"],
            }
        )

    elapsed = time.perf_counter() - start
    return rows, elapsed


def write_rows(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "target",
        "method",
        "embedding_backend",
        "jsd",
        "pdis",
        "pdiv",
        "clusters",
        "occurrences_c1",
        "occurrences_c2",
        "batches_c1",
        "batches_c2",
        "historical_refreshes",
        "c1_only_clusters",
        "c2_only_clusters",
        "mixed_clusters",
        "gold",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_diagnostics(path: Path, rows: List[dict]) -> None:
    fieldnames = ["target", "occurrences_c1", "occurrences_c2", "has_both_periods", "has_gold"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run original WiDiD and semi-dynamic WiDiD on SemEval-style corpora."
    )
    parser.add_argument("--corpus1", required=True, type=Path, help="Path to the first-period corpus.")
    parser.add_argument("--corpus2", required=True, type=Path, help="Path to the second-period corpus.")
    parser.add_argument("--targets", required=True, type=Path, help="Path to targets.txt.")
    parser.add_argument("--gold", type=Path, help="Optional gold graded-change file.")
    parser.add_argument("--output", type=Path, default=Path("widid_comparison.csv"))
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--window", type=int, default=12, help="Context window around each target.")
    parser.add_argument(
        "--embedding-backend",
        choices=["hash", "bert", "doc2vec"],
        default="hash",
        help="Embedding extractor to use before WiDiD clustering.",
    )
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--bert-device", default=None, help="Example: cpu, cuda, or mps.")
    parser.add_argument("--bert-max-length", type=int, default=128)
    parser.add_argument("--doc2vec-vector-size", type=int, default=100)
    parser.add_argument("--doc2vec-window", type=int, default=10)
    parser.add_argument("--doc2vec-epochs", type=int, default=15)
    parser.add_argument("--doc2vec-min-count", type=int, default=1)
    parser.add_argument("--max-occurrences-per-target", type=int, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument(
        "--ap-preference-quantile",
        type=float,
        default=50.0,
        help="Affinity Propagation preference percentile. Higher usually creates more clusters.",
    )
    parser.add_argument(
        "--ap-damping",
        type=float,
        default=0.9,
        help="Affinity Propagation damping, in [0.5, 1.0).",
    )
    parser.add_argument("--historical-update-threshold", type=int, default=500)
    parser.add_argument("--history-window", type=int, default=None)
    parser.add_argument("--min-cluster-fraction", type=float, default=0.0)
    parser.add_argument(
        "--batches-per-corpus",
        type=int,
        default=1,
        help="Split each target's C1 and C2 occurrences into this many sequential batches.",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="Optional path for per-target occurrence diagnostics.",
    )
    args = parser.parse_args()

    targets = read_targets(args.targets)
    gold, gold_base_entries = read_gold(args.gold)
    print(f"loaded_targets: {len(targets)}")
    print(f"loaded_gold_targets: {gold_base_entries}")
    print(f"embedding_backend: {args.embedding_backend}")

    if args.embedding_backend == "hash":
        backend = HashEmbeddingBackend(window=args.window, dim=args.dim)
        vectors_c1 = extract_target_embeddings(
            args.corpus1,
            targets,
            backend,
            args.max_occurrences_per_target,
        )
        vectors_c2 = extract_target_embeddings(
            args.corpus2,
            targets,
            backend,
            args.max_occurrences_per_target,
        )
    elif args.embedding_backend == "bert":
        backend = BertEmbeddingBackend(
            model_name=args.bert_model,
            window=args.window,
            device=args.bert_device,
            max_length=args.bert_max_length,
        )
        vectors_c1 = extract_target_embeddings(
            args.corpus1,
            targets,
            backend,
            args.max_occurrences_per_target,
        )
        vectors_c2 = extract_target_embeddings(
            args.corpus2,
            targets,
            backend,
            args.max_occurrences_per_target,
        )
    else:
        backend = Doc2VecEmbeddingBackend(
            corpus_paths=[args.corpus1, args.corpus2],
            vector_size=args.doc2vec_vector_size,
            window=args.doc2vec_window,
            epochs=args.doc2vec_epochs,
            min_count=args.doc2vec_min_count,
        )
        vectors_c1 = extract_doc2vec_target_embeddings(
            args.corpus1,
            targets,
            backend,
            args.max_occurrences_per_target,
        )
        vectors_c2 = extract_doc2vec_target_embeddings(
            args.corpus2,
            targets,
            backend,
            args.max_occurrences_per_target,
        )

    diagnostics = occurrence_diagnostics(targets, vectors_c1, vectors_c2, gold)
    both_periods = sum(1 for row in diagnostics if row["has_both_periods"])
    zero_c1 = sum(1 for row in diagnostics if row["occurrences_c1"] == 0)
    zero_c2 = sum(1 for row in diagnostics if row["occurrences_c2"] == 0)
    print(f"targets_with_occurrences_in_both_periods: {both_periods}")
    print(f"targets_missing_in_corpus1: {zero_c1}")
    print(f"targets_missing_in_corpus2: {zero_c2}")

    diagnostics_output = args.diagnostics_output
    if diagnostics_output is None:
        diagnostics_output = args.output.with_name(args.output.stem + "_diagnostics.csv")
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    write_diagnostics(diagnostics_output, diagnostics)

    widid_rows, widid_time = run_one_method(
        "widid",
        args.embedding_backend,
        vectors_c1,
        vectors_c2,
        args.similarity_threshold,
        args.historical_update_threshold,
        args.history_window,
        args.min_cluster_fraction,
        args.batches_per_corpus,
        args.ap_preference_quantile,
        args.ap_damping,
    )
    semi_rows, semi_time = run_one_method(
        "semi_dynamic_widid",
        args.embedding_backend,
        vectors_c1,
        vectors_c2,
        args.similarity_threshold,
        args.historical_update_threshold,
        args.history_window,
        args.min_cluster_fraction,
        args.batches_per_corpus,
        args.ap_preference_quantile,
        args.ap_damping,
    )

    rows = widid_rows + semi_rows
    for row in rows:
        row["gold"] = gold.get(row["target"], "")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output, rows)

    print(f"targets with usable occurrences: {len(widid_rows)}")
    print(f"batches_per_corpus: {args.batches_per_corpus}")
    print(f"ap_preference_quantile: {args.ap_preference_quantile}")
    print(f"ap_damping: {args.ap_damping}")
    print(
        "semi_dynamic_total_historical_refreshes: "
        f"{sum(int(row['historical_refreshes']) for row in semi_rows)}"
    )
    print(f"widid_runtime_seconds: {widid_time:.4f}")
    print(f"semi_dynamic_runtime_seconds: {semi_time:.4f}")
    print(f"wrote: {args.output}")
    print(f"wrote_diagnostics: {diagnostics_output}")

    if gold:
        for method in ["widid", "semi_dynamic_widid"]:
            method_rows = [row for row in rows if row["method"] == method and row["target"] in gold]
            if not method_rows:
                print(f"{method}: no rows matched the gold labels")
                continue
            gold_values = [float(row["gold"]) for row in method_rows]
            for metric in ["jsd", "pdis", "pdiv"]:
                metric_values = [float(row[metric]) for row in method_rows]
                rho = spearmanr(gold_values, metric_values)
                print(f"{method}_{metric}_spearman: {rho:.6f}")


if __name__ == "__main__":
    main()
