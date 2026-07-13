from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch.nn.functional import pad
from transformers import LogitsProcessor, LogitsProcessorList

from src.model.spec import TokenRigResult
from src.model.tokenrig import VocabSwitchingLogitsProcessor, decode
from src.tokenizer.tokenizer_part import discretize
from src.tokenizer.spec import TokenizeInput

from .session import InteractiveSession, SkeletonContext


class ForceCoordinateStepsProcessor(LogitsProcessor):
    def __init__(self, num_discrete: int, steps: int):
        self.num_discrete = num_discrete
        self.steps = steps

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] >= self.steps:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        mask[:, : self.num_discrete] = 0
        return scores + mask


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 16
    top_k: int = 5
    top_p: float = 0.95
    temperature: float = 1.5
    repetition_penalty: float = 1.2
    num_beams: int = 1
    do_sample: bool = True

    @classmethod
    def from_payload(cls, payload: dict) -> "GenerationOptions":
        return cls(
            max_new_tokens=int(payload.get("max_new_tokens", 16)),
            top_k=int(payload.get("top_k", 5)),
            top_p=float(payload.get("top_p", 0.95)),
            temperature=float(payload.get("temperature", 1.5)),
            repetition_penalty=float(payload.get("repetition_penalty", 1.2)),
            num_beams=int(payload.get("num_beams", 1)),
            do_sample=bool(payload.get("do_sample", True)),
        )

    def to_generate_kwargs(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "num_return_sequences": 1,
            "num_beams": self.num_beams,
            "do_sample": self.do_sample,
        }


def context_to_tokens(model, session: InteractiveSession, context: SkeletonContext) -> np.ndarray:
    if context.joints.shape[0] == 0:
        return np.asarray(
            model.tokenizer.make_cls_head(cls=session.cls),
            dtype=np.int64,
        )
    joints = session.normalize_points(context.joints)
    return model.tokenizer.tokenize(
        input=TokenizeInput(
            joints=joints,
            parents=context.parents.astype(int).tolist(),
            cls=session.cls,
            joint_names=context.joint_names,
        )
    )


def start_tokens_without_eos(model, session: InteractiveSession, context: SkeletonContext) -> np.ndarray:
    tokens = context_to_tokens(model, session, context)
    if tokens.shape[0] == 0 or tokens[0] != model.tokenizer.bos:
        tokens = np.concatenate([[model.tokenizer.bos], tokens]).astype(np.int64)
    if tokens[-1] == model.tokenizer.eos:
        tokens = tokens[:-1]
    return tokens.astype(np.int64)


def trim_to_next_unit(model, prefix: np.ndarray, generated: np.ndarray) -> np.ndarray:
    tokenizer = model.tokenizer
    start_count = tokenizer.bones_in_sequence(prefix)
    out: List[int] = prefix.astype(int).tolist()
    for token in generated.astype(int).tolist():
        out.append(token)
        arr = np.asarray(out, dtype=np.int64)
        if token == tokenizer.eos:
            return arr
        try:
            count = tokenizer.bones_in_sequence(arr)
        except Exception:
            continue
        if count > start_count:
            return arr
    return np.asarray(out, dtype=np.int64)


def dfs_stack(parents: np.ndarray | list[int]) -> list[int]:
    if len(parents) == 0:
        return []
    stack = []
    current = len(parents) - 1
    while current != -1:
        stack.append(int(current))
        current = int(parents[current])
    stack.reverse()
    return stack


def prefix_with_branch_parent(
    model,
    session: InteractiveSession,
    context: SkeletonContext,
    branch_parent: Optional[int],
) -> np.ndarray:
    prefix = start_tokens_without_eos(model, session, context)
    if branch_parent is None or context.joints.shape[0] == 0:
        return prefix

    parent = int(branch_parent)
    last = context.joints.shape[0] - 1
    if parent == last:
        return prefix
    stack = dfs_stack(context.parents)
    if parent not in stack:
        raise ValueError(f"branch_parent {parent} is not in current DFS stack {stack}")

    parent_joint = session.normalize_points(context.joints[parent : parent + 1])[0]
    parent_tokens = discretize(
        t=parent_joint,
        continuous_range=model.tokenizer.continuous_range,
        num_discrete=model.tokenizer.num_discrete,
    )
    return np.concatenate(
        [
            prefix,
            np.asarray([model.tokenizer.token_id_branch], dtype=np.int64),
            parent_tokens.astype(np.int64),
        ]
    )


def decode_skeleton_tokens(model, session: InteractiveSession, tokens: np.ndarray) -> SkeletonContext:
    tokenizer = model.tokenizer
    done = bool(tokens.shape[0] > 0 and tokens[-1] == tokenizer.eos)
    if done and tokenizer.bones_in_sequence(tokens) == 0:
        return SkeletonContext.empty(done=True)
    decode_tokens = tokens
    if decode_tokens[-1] != tokenizer.eos:
        decode_tokens = np.concatenate([decode_tokens, [tokenizer.eos]]).astype(np.int64)
    detok = tokenizer.detokenize(ids=decode_tokens)
    normalized_joints = detok.joints.astype(np.float32)
    obj_joints = session.denormalize_points(normalized_joints)
    names = list(detok.joint_names) if detok.joint_names is not None else [
        f"bone_{i}" for i in range(normalized_joints.shape[0])
    ]
    return SkeletonContext(
        joints=obj_joints.astype(np.float32),
        parents=np.asarray(detok.parents, dtype=np.int32),
        joint_names=names,
        done=done,
    )


@torch.no_grad()
def generate_next(
    model,
    session: InteractiveSession,
    context: SkeletonContext,
    options: GenerationOptions,
    branch_parent: Optional[int] = None,
) -> tuple[SkeletonContext, np.ndarray]:
    if context.done:
        return context, context_to_tokens(model, session, context)

    prefix_before_branch = start_tokens_without_eos(model, session, context)
    prefix = prefix_with_branch_parent(model, session, context, branch_parent)
    force_child_coordinates = branch_parent is not None and context.joints.shape[0] > 0
    device = session.vertices.device
    start_tokens = torch.as_tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
    start_embed = model.transformer.get_input_embeddings()(start_tokens)
    inputs_embeds = torch.cat([session.learned_mesh_cond, start_embed], dim=1)
    processors = [
        VocabSwitchingLogitsProcessor(
            tokenizer=model.tokenizer,
            switch_token_id=model.tokenizer.eos,
            eos_token_id=model.eos,
            tokens_per_skin=model.tokens_per_skin,
            init=start_tokens[0],
        )
    ]
    if force_child_coordinates:
        processors.append(ForceCoordinateStepsProcessor(model.tokenizer.num_discrete, steps=3))
    logits_processor = LogitsProcessorList(processors)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        results = model.transformer.generate(
            inputs_embeds=inputs_embeds,
            bos_token_id=model.tokenizer.bos,
            eos_token_id=model.tokenizer.eos,
            pad_token_id=model.tokenizer.pad,
            logits_processor=logits_processor,
            **options.to_generate_kwargs(),
        )
    generated = results[0].detach().cpu().numpy().astype(np.int64)
    next_tokens = trim_to_next_unit(model, prefix=prefix, generated=generated)
    next_context = decode_skeleton_tokens(model, session, next_tokens)
    return next_context, next_tokens


@torch.no_grad()
def generate_skin(model, session: InteractiveSession, context: SkeletonContext, options: GenerationOptions) -> TokenRigResult:
    skeleton_tokens = context_to_tokens(model, session, context)
    if skeleton_tokens[-1] != model.tokenizer.eos:
        skeleton_tokens = np.concatenate([skeleton_tokens, [model.tokenizer.eos]]).astype(np.int64)
    device = session.vertices.device
    start_tokens = torch.as_tensor(skeleton_tokens, dtype=torch.long, device=device).unsqueeze(0)
    start_embed = model.transformer.get_input_embeddings()(start_tokens)
    inputs_embeds = torch.cat([session.learned_mesh_cond, start_embed], dim=1)
    logits_processor = LogitsProcessorList([
        VocabSwitchingLogitsProcessor(
            tokenizer=model.tokenizer,
            switch_token_id=model.tokenizer.eos,
            eos_token_id=model.eos,
            tokens_per_skin=model.tokens_per_skin,
            init=start_tokens[0],
        )
    ])
    skin_options = GenerationOptions(
        max_new_tokens=max(options.max_new_tokens, model.tokens_per_skin * max(1, context.joints.shape[0]) + 8),
        top_k=options.top_k,
        top_p=options.top_p,
        temperature=options.temperature,
        repetition_penalty=options.repetition_penalty,
        num_beams=options.num_beams,
        do_sample=options.do_sample,
    )
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        results = model.transformer.generate(
            inputs_embeds=inputs_embeds,
            bos_token_id=model.tokenizer.bos,
            eos_token_id=model.eos,
            pad_token_id=model.tokenizer.pad,
            logits_processor=logits_processor,
            **skin_options.to_generate_kwargs(),
        )
    output_ids = results[0, :]
    for token in reversed(start_tokens[0]):
        output_ids = pad(output_ids, (1, 0), value=int(token.item()))
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        decoded = decode(
            cond=torch.cat([session.vertices, session.normals], dim=-1),
            cond_latents=session.cond_latents[0],
            inputs_ids=output_ids,
            tokenizer=model.tokenizer,
            tokens_per_skin=model.tokens_per_skin,
            vae=model.vae,
        )
    res = TokenRigResult()
    res.input_ids = start_tokens[0]
    res.output_ids = output_ids
    res.cond = torch.cat([session.vertices, session.normals], dim=-1)
    res.cond_latents = session.cond_latents[0]
    res.detokenize_output = decoded["detokenize_output"]
    res.skin_pred = decoded["skin_pred"]
    return res
