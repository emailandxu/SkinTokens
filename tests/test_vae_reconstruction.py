from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from torch import nn

from interactive.skin_postprocess import (
    VaeReconstructionOptions,
    VaeSkinReconstructor,
    normalized_topk_weights,
)
from interactive import server as interactive_server
from interactive.protocol import decode_float32_array, encode_float32_array
from interactive.session import SkeletonContext
from src.rig_package.info.asset import Asset


class FakeFsq(nn.Module):
    def forward(self, z):
        indices = torch.zeros(z.shape[:2], dtype=torch.int32, device=z.device)
        return z, indices, None


class FakeVaeInner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.FSQ = FakeFsq()
        self.encoded_batch_sizes: list[int] = []

    def _encode(
        self,
        x,
        cond,
        *,
        num_tokens,
        cond_tokens=1,
        seed=None,
        return_z=True,
        return_cond=True,
    ):
        if return_z:
            self.encoded_batch_sizes.append(int(x.shape[0]))
            z = x[..., -1].mean(dim=1)[:, None, None]
        else:
            z = None
        if return_cond:
            encoded_cond = torch.zeros(
                (cond.shape[0], cond_tokens, 1),
                dtype=cond.dtype,
                device=cond.device,
            )
        else:
            encoded_cond = None
        return z, encoded_cond

    def _decode(self, z, cond, sampled_points, num_chunks=None):
        return z[:, :1].expand(-1, sampled_points.shape[1], -1)


class FakeVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.model = FakeVaeInner()
        self.sample_tokens = 1
        self.cond_tokens = 1
        self.transform_config = {
            "predict_transform": {
                "sampler": {
                    "__target__": "mix",
                    "num_samples": 4,
                    "num_vertex_samples": 2,
                    "num_skin_samples": 4,
                    "all_skeleton": True,
                    "max_distance": 0.1,
                    "rate_distance": 0.1,
                }
            }
        }


class VaeReconstructionTest(unittest.TestCase):
    def test_normalized_topk_weights_limits_influences(self) -> None:
        weights = np.asarray([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]], dtype=np.float32)

        result = normalized_topk_weights(weights, topk=2)

        np.testing.assert_allclose(result.sum(axis=1), 1.0)
        self.assertEqual(int(np.count_nonzero(result[0])), 2)
        np.testing.assert_array_equal(result[1], [1.0, 0.0, 0.0])

    def test_reconstructor_processes_every_bone_in_batches(self) -> None:
        asset = Asset.from_data(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            skin=np.asarray(
                [[0.8, 0.2, 0.0], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]],
                dtype=np.float32,
            ),
            joints=np.zeros((3, 3), dtype=np.float32),
            parents=np.asarray([-1, 0, 0], dtype=np.int32),
            joint_names=["root", "left", "right"],
        )
        vae = FakeVae()

        result, report = VaeSkinReconstructor(
            vae,
            VaeReconstructionOptions(topk=2, batch_size=2, seed=7, decode_chunk=2),
        )(asset)

        self.assertEqual(result.shape, (4, 3))
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)
        self.assertEqual(vae.model.encoded_batch_sizes, [2, 1])
        self.assertEqual(report["bones"], 3)
        self.assertEqual(report["mode"], "vae-reconstruction")

    def test_reconstructor_processes_only_selected_bones(self) -> None:
        asset = Asset.from_data(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            skin=np.asarray(
                [[0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.2, 0.1, 0.7]],
                dtype=np.float32,
            ),
            joints=np.zeros((3, 3), dtype=np.float32),
            parents=np.asarray([-1, 0, 0], dtype=np.int32),
            joint_names=["root", "left", "right"],
        )
        vae = FakeVae()

        result, report = VaeSkinReconstructor(
            vae,
            VaeReconstructionOptions(topk=3, batch_size=2),
        )(
            asset,
            bone_indices=[2],
            mesh_cond_tokens=torch.zeros((1, 1, 1)),
        )

        self.assertEqual(result.shape, (3, 3))
        self.assertEqual(vae.model.encoded_batch_sizes, [1])
        self.assertEqual(report["bones"], 1)
        self.assertEqual(report["total_bones"], 3)
        self.assertEqual(report["bone_indices"], [2])
        self.assertTrue(report["reused_mesh_condition"])
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)

    def test_float32_skin_payload_roundtrip(self) -> None:
        source = np.arange(24, dtype=np.float32).reshape(6, 4) / 7.0

        restored = decode_float32_array(
            encode_float32_array(source),
            source.shape,
        )

        np.testing.assert_array_equal(restored, source)

    def test_interactive_cli_accepts_vae_reconstruction(self) -> None:
        args = interactive_server.build_parser().parse_args(
            ["--skin-postprocess", "vae-reconstruction"]
        )
        self.assertEqual(args.skin_postprocess, "vae-reconstruction")

    def test_interactive_dispatches_vae_reconstruction(self) -> None:
        asset = Asset.from_data(
            vertices=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=np.float32,
            ),
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            skin=np.asarray([[1.0], [1.0], [1.0]], dtype=np.float32),
            joints=np.zeros((1, 3), dtype=np.float32),
            parents=np.asarray([-1], dtype=np.int32),
            joint_names=["root"],
        )
        model = type("Model", (), {"vae": FakeVae()})()

        applied = interactive_server._apply_skin_postprocess(
            asset,
            {
                "skin_postprocess": "vae-reconstruction",
                "topk_skin": 1,
                "vae_reconstruction_batch_size": 1,
            },
            model,
            "cpu",
        )

        self.assertEqual(applied, "vae-reconstruction")
        np.testing.assert_allclose(asset.skin.sum(axis=1), 1.0)

    def test_interactive_reconstructs_selected_bone_fields(self) -> None:
        context = SkeletonContext(
            joints=np.zeros((2, 3), dtype=np.float32),
            parents=np.asarray([-1, 0], dtype=np.int32),
            joint_names=["root", "selected"],
        )
        vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        session = SimpleNamespace(
            session_id="session",
            normalized_vertices_cpu=vertices,
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            cls="articulation",
            obj_path=Path("mesh.obj"),
            context=context,
            normalize_points=lambda points: np.asarray(points, dtype=np.float32),
        )
        vae = FakeVae()
        service = interactive_server.InteractiveModelServer.__new__(
            interactive_server.InteractiveModelServer
        )
        service.model = SimpleNamespace(vae=vae)
        service._session = Mock(return_value=session)
        service._context = Mock(return_value=context)
        skin = np.asarray(
            [[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]],
            dtype=np.float32,
        )

        response = service.reconstruct(
            {
                "session_id": "session",
                "bone_names": ["selected"],
                "skin": encode_float32_array(skin),
                "topk_skin": 2,
            }
        )

        self.assertTrue(response["ok"])
        self.assertEqual(response["bone_names"], ["selected"])
        self.assertEqual(vae.model.encoded_batch_sizes, [1])
        reconstructed = decode_float32_array(response["skin"], skin.shape)
        np.testing.assert_allclose(reconstructed.sum(axis=1), 1.0, atol=1e-6)

    def test_interactive_returns_reconstruction_trajectory(self) -> None:
        context = SkeletonContext(
            joints=np.zeros((2, 3), dtype=np.float32),
            parents=np.asarray([-1, 0], dtype=np.int32),
            joint_names=["root", "selected"],
        )
        vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        session = SimpleNamespace(
            session_id="session",
            normalized_vertices_cpu=vertices,
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            cls="articulation",
            obj_path=Path("mesh.obj"),
            context=context,
            normalize_points=lambda points: np.asarray(points, dtype=np.float32),
        )
        vae = FakeVae()
        service = interactive_server.InteractiveModelServer.__new__(
            interactive_server.InteractiveModelServer
        )
        service.model = SimpleNamespace(vae=vae)
        service._session = Mock(return_value=session)
        service._context = Mock(return_value=context)
        skin = np.asarray(
            [[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]],
            dtype=np.float32,
        )

        response = service.reconstruct(
            {
                "session_id": "session",
                "bone_names": ["selected"],
                "skin": encode_float32_array(skin),
                "topk_skin": 2,
                "trajectory_levels": 3,
            }
        )

        self.assertTrue(response["ok"])
        self.assertEqual(response["skin_fields_shape"], [4, 3, 1])
        self.assertEqual(response["vae_reconstruction"]["levels"], 3)
        self.assertEqual(vae.model.encoded_batch_sizes, [1, 1, 1])
        trajectory = decode_float32_array(
            response["skin_fields"],
            (4, 3, 1),
        )
        np.testing.assert_array_equal(trajectory[0, :, 0], skin[:, 1])
        np.testing.assert_allclose(
            decode_float32_array(response["skin"], skin.shape).sum(axis=1),
            1.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
