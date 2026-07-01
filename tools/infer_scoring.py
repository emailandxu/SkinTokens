from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor

from src.model.tokenrig import TokenRig, encode_mesh_cond


def generation_kwargs_from_args(args: argparse.Namespace) -> dict:
    return dict(
        max_length=2048,
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        temperature=float(args.temperature),
        repetition_penalty=float(args.repetition_penalty),
        num_return_sequences=1,
        num_beams=int(args.num_beams),
        do_sample=True,
    )


@torch.no_grad()
def score_tokens(model: TokenRig, batch: dict, tokens: Tensor) -> dict:
    vertices = batch["vertices"][0]
    normals = batch["normals"][0]
    tokens = tokens.to(vertices.device).long()
    if tokens.numel() < 2:
        raise ValueError("token sequence is too short to score")

    device_type = "cuda" if vertices.device.type == "cuda" else "cpu"
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        learned_mesh_cond = encode_mesh_cond(
            model.mesh_encoder,
            model.output_proj,
            model.tokens_skin_cond,
            {"vertices": vertices, "normals": normals},
        )
        token_embed = model.transformer.get_input_embeddings()(tokens.unsqueeze(0))
        outputs = model.transformer(
            inputs_embeds=torch.cat([learned_mesh_cond, token_embed], dim=1),
            use_cache=False,
        )

    cond_len = learned_mesh_cond.shape[1]
    logits = outputs.logits[0, cond_len : cond_len + tokens.numel() - 1].float()
    targets = tokens[1:]
    token_logprobs = F.log_softmax(logits, dim=-1).gather(
        dim=-1,
        index=targets[:, None],
    )[:, 0]

    target_positions = torch.arange(1, tokens.numel(), device=tokens.device)
    joint_mask = (
        (targets < model.tokenizer.num_discrete)
        | (targets == model.tokenizer.token_id_branch)
    )
    joint_logprobs = token_logprobs[joint_mask]

    return {
        "tokens": tokens.detach().cpu().numpy(),
        "target_positions": target_positions.detach().cpu().numpy(),
        "token_logprobs": token_logprobs.detach().cpu().numpy(),
        "joint_positions": target_positions[joint_mask].detach().cpu().numpy(),
        "joint_logprobs": joint_logprobs.detach().cpu().numpy(),
        "total_logprob": float(token_logprobs.sum().item()),
        "mean_nll": float((-token_logprobs).mean().item()),
        "num_scored_tokens": int(token_logprobs.numel()),
        "joint_total_logprob": float(joint_logprobs.sum().item()),
        "joint_mean_nll": float((-joint_logprobs).mean().item()),
        "num_joint_tokens": int(joint_logprobs.numel()),
    }


def get_skeleton_tokens_from_batch(batch: dict) -> Tensor:
    if "skeleton_tokens" not in batch or "skeleton_mask" not in batch:
        raise ValueError("--score-skeleton requires --txt")
    mask = batch["skeleton_mask"][0] == 1
    return batch["skeleton_tokens"][0][mask].long()


def score_skeleton_tokens(model: TokenRig, batch: dict) -> dict:
    return score_tokens(model, batch, get_skeleton_tokens_from_batch(batch))


@torch.no_grad()
def generate_output_ids(model: TokenRig, batch: dict, skeleton_tokens: Tensor) -> Tensor:
    pred = model.predict_step(
        batch,
        skeleton_tokens=[skeleton_tokens.detach().cpu().numpy()],
        make_asset=False,
    )["results"][0]
    if pred.output_ids is None:
        raise RuntimeError("SkinTokens did not return output ids")
    return pred.output_ids


def score_generated_skin_tokens(model: TokenRig, batch: dict, output_ids: Tensor) -> dict:
    score = score_tokens(model, batch, output_ids)
    tokens = torch.as_tensor(score["tokens"])
    token_logprobs = torch.as_tensor(score["token_logprobs"])
    where_skeleton_eos = torch.where(tokens == model.tokenizer.eos)[0]
    if where_skeleton_eos.numel() == 0:
        raise RuntimeError("generated ids do not contain skeleton eos")
    skin_start = int(where_skeleton_eos[0].item()) + 1
    where_final_eos = torch.where(tokens == model.eos)[0]
    skin_end = int(where_final_eos[0].item()) if where_final_eos.numel() else int(tokens.numel())
    if skin_start >= skin_end:
        raise RuntimeError("generated ids do not contain skin tokens")

    target_positions = torch.arange(1, tokens.numel())
    skin_mask = (target_positions >= skin_start) & (target_positions < skin_end)
    skin_logprobs = token_logprobs[skin_mask]
    score.update(
        {
            "num_skin_tokens": int(skin_logprobs.numel()),
            "skin_total_logprob": float(skin_logprobs.sum().item()),
            "skin_mean_nll": float((-skin_logprobs).mean().item()),
        }
    )
    return score


def print_skeleton_score(score: dict) -> None:
    print("[infer] skeleton score")
    print(f"[infer] scored tokens: {score['num_scored_tokens']}")
    print(f"[infer] total logprob: {score['total_logprob']:.6f}")
    print(f"[infer] mean nll: {score['mean_nll']:.6f}")
    print(f"[infer] joint tokens: {score['num_joint_tokens']}")
    print(f"[infer] joint total logprob: {score['joint_total_logprob']:.6f}")
    print(f"[infer] joint mean nll: {score['joint_mean_nll']:.6f}")


def print_skin_score(score: dict) -> None:
    print("[infer] generated skin score")
    print(f"[infer] skin tokens: {score['num_skin_tokens']}")
    print(f"[infer] skin total logprob: {score['skin_total_logprob']:.6f}")
    print(f"[infer] skin mean nll: {score['skin_mean_nll']:.6f}")
