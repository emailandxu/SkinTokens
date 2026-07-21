from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor
from transformers import LogitsProcessor, LogitsProcessorList

from .session import SkeletonContext
from .skeleton_stream import (
    children_from_parents,
    longest_chain_indices,
    reorder_skeleton_context,
    subtree_similarity,
    subtree_size,
)
from src.model.tokenrig import TokenRig, decode_multi, encode_mesh_cond
from src.tokenizer.spec import TokenizeInput


SKIN_MODE_SINGLE = "single"
SKIN_MODE_DFS_ENSEMBLE = "dfs-ensemble"


@dataclass(frozen=True)
class SimilarSubtreePair:
    root_a: int
    root_b: int
    nodes_a: tuple[int, ...]
    nodes_b: tuple[int, ...]
    similarity: float


@dataclass(frozen=True)
class DfsOrderCandidate:
    name: str
    order: tuple[int, ...]
    description: str


@dataclass(frozen=True)
class SkinEnsembleOptions:
    max_candidates: int = 8
    batch_size: int = 8
    num_beams: int = 10
    top_k: int = 5
    top_p: float = 0.95
    temperature: float = 1.5
    repetition_penalty: float = 1.2
    do_sample: bool = True
    topk_skin: int = 4
    pair_threshold: float = 0.80
    min_subtree_size: int = 4
    max_regression: float = 0.01
    seed: Optional[int] = None

    @classmethod
    def from_payload(cls, payload: dict, *, default_num_beams: int = 10) -> "SkinEnsembleOptions":
        max_candidates = min(
            max(1, int(payload.get("skin_ensemble_max_candidates", 8))),
            16,
        )
        return cls(
            max_candidates=max_candidates,
            batch_size=min(
                max(1, int(payload.get("skin_ensemble_batch_size", 8))),
                max_candidates,
                8,
            ),
            num_beams=min(
                max(1, int(payload.get("skin_num_beams", default_num_beams))),
                10,
            ),
            top_k=int(payload.get("top_k", 5)),
            top_p=float(payload.get("top_p", 0.95)),
            temperature=float(payload.get("temperature", 1.5)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.2)),
            do_sample=bool(payload.get("do_sample", True)),
            topk_skin=max(1, int(payload.get("topk_skin", 4))),
            pair_threshold=float(payload.get("skin_ensemble_pair_threshold", 0.80)),
            min_subtree_size=max(2, int(payload.get("skin_ensemble_min_subtree_size", 4))),
            max_regression=max(0.0, float(payload.get("skin_ensemble_max_regression", 0.01))),
            seed=None if payload.get("seed") is None else int(payload["seed"]),
        )


@dataclass
class SkinCandidateResult:
    name: str
    description: str
    order: tuple[int, ...]
    sampled_skin: np.ndarray
    output_ids: Tensor
    metrics: dict[str, object]

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "order": list(self.order),
            "metrics": self.metrics,
        }


@dataclass
class SkinEnsembleResult:
    selected: SkinCandidateResult
    candidates: list[SkinCandidateResult]
    candidate_count: int
    batch_size: int
    wall_sec: float
    cuda_peak_allocated_mb: Optional[float]

    def report(self, *, include_candidates: bool = True) -> dict[str, object]:
        report: dict[str, object] = {
            "mode": SKIN_MODE_DFS_ENSEMBLE,
            "candidate_count": self.candidate_count,
            "batch_size": self.batch_size,
            "selected_candidate": self.selected.name,
            "selected_order": list(self.selected.order),
            "selected_metrics": self.selected.metrics,
            "performance": {
                "wall_sec": self.wall_sec,
                "cuda_peak_allocated_mb": self.cuda_peak_allocated_mb,
            },
        }
        if include_candidates:
            report["candidates"] = [candidate.summary() for candidate in self.candidates]
        return report


class SkinTokenOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        mask = torch.full_like(scores, -torch.inf)
        mask[:, self.start : self.end] = 0
        return scores + mask


def _reset_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _descendants(root: int, children: Sequence[Sequence[int]]) -> tuple[int, ...]:
    nodes: list[int] = []

    def visit(node: int) -> None:
        nodes.append(node)
        for child in children[node]:
            visit(int(child))

    visit(root)
    return tuple(nodes)


def find_similar_subtree_pairs(
    joints: np.ndarray,
    parents: np.ndarray,
    *,
    threshold: float = 0.80,
    min_size: int = 4,
) -> list[SimilarSubtreePair]:
    children = children_from_parents(parents)
    candidates: list[tuple[float, int, int]] = []
    for siblings in children:
        eligible = [node for node in siblings if subtree_size(node, children) >= min_size]
        for position, a in enumerate(eligible):
            for b in eligible[position + 1 :]:
                similarity = subtree_similarity(joints, children, a, b)
                if similarity >= threshold:
                    candidates.append((similarity, a, b))

    used: set[int] = set()
    pairs: list[SimilarSubtreePair] = []
    for similarity, a, b in sorted(candidates, reverse=True):
        if a in used or b in used:
            continue
        if b < a:
            a, b = b, a
        if min(
            len(longest_chain_indices(a, children)),
            len(longest_chain_indices(b, children)),
        ) < min_size:
            continue
        pairs.append(
            SimilarSubtreePair(
                root_a=a,
                root_b=b,
                nodes_a=_descendants(a, children),
                nodes_b=_descendants(b, children),
                similarity=similarity,
            )
        )
        used.update((a, b))
    return sorted(pairs, key=lambda pair: pair.root_a)


def _dfs_order(children: Sequence[Sequence[int]], root: int = 0) -> tuple[int, ...]:
    order: list[int] = []

    def visit(node: int) -> None:
        order.append(node)
        for child in children[node]:
            visit(int(child))

    visit(root)
    return tuple(order)


def _move_child_first(parents: np.ndarray, child: int) -> Optional[tuple[int, ...]]:
    parent = int(parents[child])
    if parent < 0:
        return None
    children = [list(row) for row in children_from_parents(parents)]
    siblings = children[parent]
    if child not in siblings or siblings[0] == child:
        return None
    siblings.remove(child)
    children[parent] = [child] + siblings
    return _dfs_order(children)


def _swap_siblings(parents: np.ndarray, a: int, b: int) -> Optional[tuple[int, ...]]:
    parent = int(parents[a])
    if parent < 0 or parent != int(parents[b]):
        return None
    children = [list(row) for row in children_from_parents(parents)]
    siblings = children[parent]
    if a not in siblings or b not in siblings:
        return None
    ia = siblings.index(a)
    ib = siblings.index(b)
    siblings[ia], siblings[ib] = siblings[ib], siblings[ia]
    return _dfs_order(children)


def enumerate_dfs_candidates(
    rig: SkeletonContext,
    options: SkinEnsembleOptions,
) -> tuple[list[DfsOrderCandidate], list[SimilarSubtreePair]]:
    identity = tuple(range(rig.parents.shape[0]))
    results = [DfsOrderCandidate("baseline", identity, "canonical skeleton order")]
    seen = {identity}
    pairs = find_similar_subtree_pairs(
        rig.joints,
        rig.parents,
        threshold=options.pair_threshold,
        min_size=options.min_subtree_size,
    )

    def append(name: str, order: Optional[tuple[int, ...]], description: str) -> None:
        if order is None or order in seen or len(results) >= options.max_candidates:
            return
        seen.add(order)
        results.append(DfsOrderCandidate(name, order, description))

    for pair_index, pair in enumerate(pairs):
        append(
            f"pair{pair_index}_root{pair.root_a}_first",
            _move_child_first(rig.parents, pair.root_a),
            f"move subtree root {pair.root_a} first among siblings",
        )
        append(
            f"pair{pair_index}_root{pair.root_b}_first",
            _move_child_first(rig.parents, pair.root_b),
            f"move subtree root {pair.root_b} first among siblings",
        )
        append(
            f"pair{pair_index}_swap_{pair.root_a}_{pair.root_b}",
            _swap_siblings(rig.parents, pair.root_a, pair.root_b),
            f"swap similar subtree roots {pair.root_a} and {pair.root_b}",
        )

    children = children_from_parents(rig.parents)
    for siblings in children:
        for child in siblings:
            if len(results) >= options.max_candidates:
                break
            if subtree_size(int(child), children) < options.min_subtree_size:
                continue
            append(
                f"subtree{child}_first",
                _move_child_first(rig.parents, int(child)),
                f"move sizeable subtree root {child} first among siblings",
            )
    return results, pairs


def _tokenize_rig(
    model: TokenRig,
    rig: SkeletonContext,
    cls: Optional[str],
) -> np.ndarray:
    return model.tokenizer.tokenize(
        input=TokenizeInput(
            joints=rig.joints,
            parents=rig.parents.tolist(),
            cls=cls,
            joint_names=rig.joint_names,
        )
    )


def remap_skin_to_canonical(skin: np.ndarray, order: Sequence[int]) -> np.ndarray:
    canonical = np.zeros_like(skin)
    canonical[:, np.asarray(order, dtype=np.int64)] = skin
    return canonical


def _normalized_topk_skin(weights: np.ndarray, topk: int) -> np.ndarray:
    weights = np.nan_to_num(weights.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    weights[weights < 0.0] = 0.0
    if 0 < topk < weights.shape[1]:
        keep = np.argpartition(weights, -topk, axis=1)[:, -topk:]
        mask = np.zeros_like(weights, dtype=bool)
        np.put_along_axis(mask, keep, True, axis=1)
        weights[~mask] = 0.0
    row_sum = weights.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= 1e-12
    if np.any(empty):
        weights[empty, 0] = 1.0
        row_sum = weights.sum(axis=1, keepdims=True)
    return weights / row_sum


def _bone_segments(joints: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = joints.copy()
    ends = joints.copy()
    children: list[list[int]] = [[] for _ in parents]
    for child, parent in enumerate(parents):
        if int(parent) >= 0:
            children[int(parent)].append(child)
    for bone_id in range(joints.shape[0]):
        if children[bone_id]:
            ends[bone_id] = joints[children[bone_id][0]]
        elif int(parents[bone_id]) >= 0:
            parent = int(parents[bone_id])
            ends[bone_id] = joints[bone_id] + (joints[bone_id] - joints[parent]) * 0.35
        else:
            ends[bone_id] = joints[bone_id] + np.asarray([0.0, 0.0, 0.1], dtype=np.float32)
    return starts, ends


def _point_segment_distances(points: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    segment = ends - starts
    denominator = np.sum(segment * segment, axis=1)
    offset = points[:, None, :] - starts[None, :, :]
    projection = np.sum(offset * segment[None, :, :], axis=2)
    projection /= np.maximum(denominator[None, :], 1e-12)
    projection = np.clip(projection, 0.0, 1.0)
    closest = starts[None, :, :] + projection[:, :, None] * segment[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)


def calculate_skin_metrics(
    vertices: np.ndarray,
    skin: np.ndarray,
    rig: SkeletonContext,
    pairs: Sequence[SimilarSubtreePair],
) -> dict[str, object]:
    starts, ends = _bone_segments(rig.joints, rig.parents)
    distances = _point_segment_distances(vertices, starts, ends)
    nearest = np.argmin(distances, axis=1)
    best_distance = distances[np.arange(vertices.shape[0]), nearest]
    lengths = np.linalg.norm(ends - starts, axis=1)
    positive = lengths[lengths > 1e-6]
    median_length = float(np.median(positive)) if positive.size else 1.0
    far = distances > best_distance[:, None] + 0.75 * median_length
    far_mass = float(np.mean(np.sum(np.where(far, skin, 0.0), axis=1)))
    regret = float(
        np.mean(np.sum(skin * distances, axis=1) - best_distance)
        / max(median_length, 1e-8)
    )

    pair_metrics: dict[str, object] = {}
    total_wrong = 0.0
    total_vertices = 0
    for pair in pairs:
        nodes_a = np.asarray(pair.nodes_a, dtype=np.int64)
        nodes_b = np.asarray(pair.nodes_b, dtype=np.int64)
        region_a = np.isin(nearest, nodes_a)
        region_b = np.isin(nearest, nodes_b)
        wrong_a = float(np.sum(skin[region_a][:, nodes_b])) if np.any(region_a) else 0.0
        wrong_b = float(np.sum(skin[region_b][:, nodes_a])) if np.any(region_b) else 0.0
        count_a = int(np.sum(region_a))
        count_b = int(np.sum(region_b))
        count = count_a + count_b
        total_wrong += wrong_a + wrong_b
        total_vertices += count
        pair_metrics[f"{pair.root_a}<->{pair.root_b}"] = {
            "vertices_a": count_a,
            "vertices_b": count_b,
            "a_to_b": wrong_a / count_a if count_a else 0.0,
            "b_to_a": wrong_b / count_b if count_b else 0.0,
            "cross_subtree_leak": (wrong_a + wrong_b) / count if count else 0.0,
        }
    return {
        "cross_subtree_leak": total_wrong / total_vertices if total_vertices else 0.0,
        "far_bone_weight_mass": far_mass,
        "distance_regret": regret,
        "pairs": pair_metrics,
    }


def select_skin_candidate(
    candidates: Sequence[SkinCandidateResult],
    *,
    max_regression: float,
) -> SkinCandidateResult:
    baseline = next(candidate for candidate in candidates if candidate.name == "baseline")
    baseline_far = float(baseline.metrics["far_bone_weight_mass"])
    baseline_regret = float(baseline.metrics["distance_regret"])
    feasible = [
        candidate
        for candidate in candidates
        if float(candidate.metrics["far_bone_weight_mass"])
        <= baseline_far * (1.0 + max_regression)
        and float(candidate.metrics["distance_regret"])
        <= baseline_regret * (1.0 + max_regression)
    ]
    if not feasible:
        return baseline
    return min(
        feasible,
        key=lambda candidate: (
            float(candidate.metrics["cross_subtree_leak"]),
            float(candidate.metrics["far_bone_weight_mass"]),
            float(candidate.metrics["distance_regret"]),
        ),
    )


def _cuda_peak_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device=device) / (1024**2))


@torch.no_grad()
def _generate_group(
    model: TokenRig,
    learned_mesh_cond: Tensor,
    skeleton_tokens: Tensor,
    options: SkinEnsembleOptions,
) -> list[Tensor]:
    batch_size = skeleton_tokens.shape[0]
    mesh_condition = learned_mesh_cond.repeat(batch_size, 1, 1)
    skeleton_embed = model.transformer.get_input_embeddings()(skeleton_tokens)
    inputs_embeds = torch.cat([mesh_condition, skeleton_embed], dim=1)
    total_skin_tokens = model.tokens_per_skin * model.tokenizer.bones_in_sequence(
        skeleton_tokens[0].detach().cpu().numpy()
    )
    device_type = skeleton_tokens.device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        generated = model.transformer.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=total_skin_tokens,
            min_new_tokens=total_skin_tokens,
            top_k=options.top_k,
            top_p=options.top_p,
            temperature=options.temperature,
            repetition_penalty=options.repetition_penalty,
            num_return_sequences=1,
            num_beams=options.num_beams,
            do_sample=options.do_sample,
            use_cache=True,
            bos_token_id=model.tokenizer.bos,
            eos_token_id=model.eos,
            pad_token_id=model.tokenizer.pad,
            logits_processor=LogitsProcessorList(
                [SkinTokenOnlyLogitsProcessor(model.tokenizer.vocab_size, model.eos)]
            ),
        )
    skin = generated[:, -total_skin_tokens:]
    if generated.shape[0] != batch_size:
        raise RuntimeError(f"unexpected generated batch size: {tuple(generated.shape)}")
    if torch.any(skin < model.tokenizer.vocab_size) or torch.any(skin >= model.eos):
        raise RuntimeError("ensemble generation returned a non-skin token")
    return [
        torch.cat([skeleton_tokens[index], skin[index]], dim=0)
        for index in range(batch_size)
    ]


@torch.no_grad()
def generate_skin_ensemble(
    model: TokenRig,
    *,
    vertices: Tensor,
    normals: Tensor,
    rig: SkeletonContext,
    cls: Optional[str],
    options: SkinEnsembleOptions,
    learned_mesh_cond: Optional[Tensor] = None,
    cond_latents: Optional[Tensor] = None,
) -> SkinEnsembleResult:
    if vertices.dim() != 2 or normals.dim() != 2:
        raise ValueError("skin ensemble expects unbatched vertices and normals")
    device = vertices.device
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    _reset_seed(options.seed)

    cond = torch.cat([vertices, normals], dim=-1).unsqueeze(0)
    device_type = device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        if learned_mesh_cond is None:
            learned_mesh_cond = encode_mesh_cond(
                model.mesh_encoder,
                model.output_proj,
                model.tokens_skin_cond,
                {"vertices": vertices, "normals": normals},
            )
        if cond_latents is None:
            _, cond_latents = model.vae.model._encode(
                x=None,
                cond=cond,
                num_tokens=model.tokens_per_skin,
                cond_tokens=model.tokens_skin_cond,
                seed=options.seed,
                return_z=False,
            )
    if cond_latents is None:
        raise RuntimeError("VAE did not return mesh condition tokens")

    order_candidates, pairs = enumerate_dfs_candidates(rig, options)
    prepared: list[tuple[DfsOrderCandidate, np.ndarray]] = []
    for candidate in order_candidates:
        candidate_rig = (
            rig
            if candidate.name == "baseline"
            else reorder_skeleton_context(rig, candidate.order)
        )
        prepared.append((candidate, _tokenize_rig(model, candidate_rig, cls)))

    groups: dict[int, list[tuple[DfsOrderCandidate, np.ndarray]]] = {}
    for item in prepared:
        groups.setdefault(item[1].shape[0], []).append(item)

    results: list[SkinCandidateResult] = []
    for token_length in sorted(groups):
        group = groups[token_length]
        for start in range(0, len(group), options.batch_size):
            chunk = group[start : start + options.batch_size]
            skeleton_tokens = torch.as_tensor(
                np.stack([tokens for _, tokens in chunk]),
                dtype=torch.long,
                device=device,
            )
            output_ids = _generate_group(model, learned_mesh_cond, skeleton_tokens, options)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                decoded = decode_multi(
                    cond=cond[0],
                    cond_latents=cond_latents[0],
                    inputs_ids=output_ids,
                    tokenizer=model.tokenizer,
                    tokens_per_skin=model.tokens_per_skin,
                    vae=model.vae,
                    is_numpy=True,
                )
            sampled_vertices = vertices.detach().float().cpu().numpy()
            for (candidate, _), ids, decoded_item in zip(chunk, output_ids, decoded):
                raw_skin = decoded_item["skin_pred"]
                if raw_skin is None:
                    raise RuntimeError(f"candidate {candidate.name} failed to decode")
                candidate_skin = _normalized_topk_skin(np.asarray(raw_skin), options.topk_skin)
                canonical_skin = remap_skin_to_canonical(candidate_skin, candidate.order)
                results.append(
                    SkinCandidateResult(
                        name=candidate.name,
                        description=candidate.description,
                        order=candidate.order,
                        sampled_skin=canonical_skin,
                        output_ids=ids,
                        metrics=calculate_skin_metrics(
                            sampled_vertices,
                            canonical_skin,
                            rig,
                            pairs,
                        ),
                    )
                )

    selected = select_skin_candidate(results, max_regression=options.max_regression)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return SkinEnsembleResult(
        selected=selected,
        candidates=results,
        candidate_count=len(results),
        batch_size=options.batch_size,
        wall_sec=time.perf_counter() - start_time,
        cuda_peak_allocated_mb=_cuda_peak_mb(device),
    )
