from __future__ import annotations

import unittest

import numpy as np
import torch

from interactive.session import SkeletonContext
from interactive.skin_generation import (
    SkinCandidateResult,
    SkinEnsembleOptions,
    enumerate_dfs_candidates,
    remap_skin_to_canonical,
    select_skin_candidate,
)


def symmetric_rig() -> SkeletonContext:
    joints = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, 2.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=np.float32,
    )
    parents = np.asarray([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8], dtype=np.int32)
    names = [f"bone_{index}" for index in range(parents.shape[0])]
    return SkeletonContext(
        joints=joints,
        parents=parents,
        joint_names=names,
    )


def candidate(
    name: str,
    *,
    leak: float,
    far_mass: float,
    regret: float,
) -> SkinCandidateResult:
    return SkinCandidateResult(
        name=name,
        description=name,
        order=(0,),
        sampled_skin=np.ones((1, 1), dtype=np.float32),
        output_ids=torch.zeros(1, dtype=torch.long),
        metrics={
            "cross_subtree_leak": leak,
            "far_bone_weight_mass": far_mass,
            "distance_regret": regret,
            "pairs": {},
        },
    )


class SkinEnsembleTest(unittest.TestCase):
    def test_enumeration_is_deduplicated_valid_and_capped(self) -> None:
        rig = symmetric_rig()
        options = SkinEnsembleOptions(max_candidates=3, min_subtree_size=3)

        candidates, pairs = enumerate_dfs_candidates(rig, options)

        self.assertGreaterEqual(len(pairs), 1)
        self.assertEqual(candidates[0].name, "baseline")
        self.assertLessEqual(len(candidates), 3)
        orders = [item.order for item in candidates]
        self.assertEqual(len(orders), len(set(orders)))
        for order in orders:
            self.assertEqual(sorted(order), list(range(rig.parents.shape[0])))

    def test_remap_skin_restores_canonical_joint_columns(self) -> None:
        order = (0, 2, 1)
        reordered_skin = np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32)

        canonical = remap_skin_to_canonical(reordered_skin, order)

        np.testing.assert_array_equal(
            canonical,
            np.asarray([[10.0, 30.0, 20.0]], dtype=np.float32),
        )

    def test_selector_rejects_geoleak_gain_with_metric_regression(self) -> None:
        baseline = candidate("baseline", leak=0.20, far_mass=0.10, regret=0.20)
        safe = candidate("safe", leak=0.10, far_mass=0.1005, regret=0.201)
        unsafe = candidate("unsafe", leak=0.01, far_mass=0.12, regret=0.20)

        selected = select_skin_candidate(
            [baseline, safe, unsafe],
            max_regression=0.01,
        )

        self.assertEqual(selected.name, "safe")

    def test_payload_uses_skin_specific_beam_count(self) -> None:
        options = SkinEnsembleOptions.from_payload(
            {
                "num_beams": 1,
                "skin_num_beams": 99,
                "skin_ensemble_max_candidates": 99,
                "skin_ensemble_batch_size": 99,
            },
        )

        self.assertEqual(options.num_beams, 10)
        self.assertEqual(options.max_candidates, 16)
        self.assertEqual(options.batch_size, 8)


if __name__ == "__main__":
    unittest.main()
