from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from interactive.blender.apply_skin import _tail_for_joint, parse_rig_txt
from interactive.blender.rig_postprocess import (
    MIRROR_AXIS_X,
    MIRROR_AXIS_Y,
    MIRROR_AXIS_Z,
    MIRROR_DISTANCE_AVERAGE,
    MIRROR_DISTANCE_FARTHEST,
    MIRROR_DISTANCE_NEAREST,
    TEMPLATE_BIPED,
    TEMPLATE_QUADRUPED,
    RigBoneSnapshot,
    RigPostprocessError,
    _BranchPair,
    _RigAnalysis,
    _symmetric_bone_targets,
    _symmetric_points,
    build_naming_plan,
    infer_selected_mirror_pairs,
)


ROOT = Path(__file__).resolve().parents[1]


def _transform_point(
    point: tuple[float, float, float],
    *,
    angle_degrees: float,
    offset: tuple[float, float, float],
) -> tuple[float, float, float]:
    angle = math.radians(angle_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    x, y, z = point
    ox, oy, oz = offset
    return (
        cosine * x - sine * y + ox,
        sine * x + cosine * y + oy,
        z + oz,
    )


def _snapshots(
    rows: list[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ],
    *,
    angle_degrees: float,
    offset: tuple[float, float, float],
) -> tuple[RigBoneSnapshot, ...]:
    return tuple(
        RigBoneSnapshot(
            joint_id=f"joint-{index}",
            name=f"bone_{index}",
            parent=parent,
            head=_transform_point(
                head,
                angle_degrees=angle_degrees,
                offset=offset,
            ),
            tail=_transform_point(
                tail,
                angle_degrees=angle_degrees,
                offset=offset,
            ),
        )
        for index, (parent, head, tail) in enumerate(rows)
    )


def make_biped(
    *,
    angle_degrees: float = 0.0,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[RigBoneSnapshot, ...]:
    # Main path: pelvis -> chest -> neck -> head.
    # Limbs are intentionally disconnected at their attachment bones, as generated
    # SkinTokens rigs can be, while preserving the parent graph.
    rows = [
        (-1, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        (0, (0.0, 0.0, 1.0), (0.0, 0.0, 2.0)),
        (1, (0.0, 0.0, 2.0), (0.0, 0.0, 2.7)),
        (2, (0.0, 0.0, 2.7), (0.0, 0.0, 3.2)),
        (0, (0.65, 0.0, 0.1), (0.75, 0.0, -0.9)),
        (4, (0.75, 0.0, -0.9), (0.85, 0.05, -1.9)),
        (0, (-0.65, 0.0, 0.1), (-0.75, 0.0, -0.9)),
        (6, (-0.75, 0.0, -0.9), (-0.85, 0.05, -1.9)),
        (1, (0.55, 0.0, 1.75), (1.35, 0.0, 1.55)),
        (8, (1.35, 0.0, 1.55), (2.15, 0.05, 1.35)),
        (1, (-0.55, 0.0, 1.75), (-1.35, 0.0, 1.55)),
        (10, (-1.35, 0.0, 1.55), (-2.15, 0.05, 1.35)),
        (0, (0.0, 0.35, 0.1), (0.0, 0.9, 0.0)),
        (12, (0.0, 0.9, 0.0), (0.0, 1.45, -0.1)),
    ]
    return _snapshots(
        rows,
        angle_degrees=angle_degrees,
        offset=offset,
    )


def make_quadruped(
    *,
    angle_degrees: float = 0.0,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[RigBoneSnapshot, ...]:
    # Main path runs from the pelvis towards the head along local +Y.
    rows = [
        (-1, (0.0, 0.0, 1.2), (0.0, 0.8, 1.25)),
        (0, (0.0, 0.8, 1.25), (0.0, 1.7, 1.3)),
        (1, (0.0, 1.7, 1.3), (0.0, 2.5, 1.45)),
        (2, (0.0, 2.5, 1.45), (0.0, 3.15, 1.7)),
        (3, (0.0, 3.15, 1.7), (0.0, 3.65, 1.8)),
        (0, (0.65, 0.05, 1.15), (0.72, 0.05, 0.45)),
        (5, (0.72, 0.05, 0.45), (0.78, 0.15, -0.2)),
        (0, (-0.65, 0.05, 1.15), (-0.72, 0.05, 0.45)),
        (7, (-0.72, 0.05, 0.45), (-0.78, 0.15, -0.2)),
        (2, (0.62, 2.45, 1.35), (0.7, 2.45, 0.55)),
        (9, (0.7, 2.45, 0.55), (0.77, 2.55, -0.2)),
        (2, (-0.62, 2.45, 1.35), (-0.7, 2.45, 0.55)),
        (11, (-0.7, 2.45, 0.55), (-0.77, 2.55, -0.2)),
        (0, (0.0, -0.45, 1.2), (0.0, -1.1, 1.15)),
        (13, (0.0, -1.1, 1.15), (0.0, -1.75, 1.05)),
    ]
    return _snapshots(
        rows,
        angle_degrees=angle_degrees,
        offset=offset,
    )


def make_quadruped_with_hanging_tail() -> tuple[RigBoneSnapshot, ...]:
    bones = list(make_quadruped())
    bones[13] = RigBoneSnapshot(
        joint_id="joint-13",
        name="bone_13",
        parent=0,
        head=(0.0, -0.08, 1.2),
        tail=(0.0, -0.1, 0.55),
    )
    bones[14] = RigBoneSnapshot(
        joint_id="joint-14",
        name="bone_14",
        parent=13,
        head=(0.0, -0.1, 0.55),
        tail=(0.0, -0.12, -0.1),
    )
    return tuple(bones)


def make_tailless_quadruped_with_short_center_attachment(
) -> tuple[RigBoneSnapshot, ...]:
    bones = list(make_quadruped()[:13])
    bones.append(
        RigBoneSnapshot(
            joint_id="joint-13",
            name="bone_13",
            parent=0,
            head=(0.0, -0.08, 1.2),
            tail=(0.0, -0.1, 1.05),
        )
    )
    return tuple(bones)


def make_quadruped_with_raised_front_limbs_and_ears(
) -> tuple[RigBoneSnapshot, ...]:
    rows = [
        (-1, (0.0, 0.0, 1.0), (0.0, 0.8, 1.0)),
        (0, (0.0, 0.8, 1.0), (0.0, 1.6, 1.05)),
        (1, (0.0, 1.6, 1.05), (0.0, 2.4, 1.15)),
        (2, (0.0, 2.4, 1.15), (0.0, 3.0, 1.3)),
        (3, (0.0, 3.0, 1.3), (0.0, 3.4, 1.4)),
        (0, (0.55, 0.05, 0.95), (0.62, 0.1, 0.25)),
        (5, (0.62, 0.1, 0.25), (0.68, 0.2, -0.35)),
        (0, (-0.55, 0.05, 0.95), (-0.62, 0.1, 0.25)),
        (7, (-0.62, 0.1, 0.25), (-0.68, 0.2, -0.35)),
        # The front limbs rise in world Z, reproducing name_skeleton.blend.
        (2, (0.52, 2.35, 1.1), (0.65, 2.45, 1.75)),
        (9, (0.65, 2.45, 1.75), (0.72, 2.6, 2.35)),
        (2, (-0.52, 2.35, 1.1), (-0.65, 2.45, 1.75)),
        (11, (-0.65, 2.45, 1.75), (-0.72, 2.6, 2.35)),
        # A short symmetric ear pair must not displace either limb pair.
        (4, (0.32, 3.35, 1.42), (0.72, 3.6, 1.6)),
        (4, (-0.32, 3.35, 1.42), (-0.72, 3.6, 1.6)),
        (0, (0.0, -0.4, 1.0), (0.0, -1.0, 0.95)),
        (15, (0.0, -1.0, 0.95), (0.0, -1.55, 0.9)),
    ]
    return _snapshots(rows, angle_degrees=0.0, offset=(0.0, 0.0, 0.0))


def make_multisegment_ears_with_shared_attachment(
) -> tuple[RigBoneSnapshot, ...]:
    # This mirrors the generated ear topology in name_skeleton.blend: the
    # attachment belongs to one side, while both three-bone ear chains are its
    # children. A distant unselected bone preserves the full-rig score scale.
    rows = [
        (-1, (0.003, 0.391, 0.306), (0.003, 0.462, 0.365)),
        (0, (0.003, 0.462, 0.365), (-0.063, 0.549, 0.369)),
        (1, (-0.063, 0.549, 0.369), (-0.080, 0.566, 0.373)),
        (2, (-0.080, 0.566, 0.373), (-0.093, 0.578, 0.377)),
        (3, (-0.093, 0.578, 0.377), (-0.097, 0.583, 0.379)),
        (1, (0.074, 0.549, 0.369), (0.091, 0.566, 0.373)),
        (5, (0.091, 0.566, 0.373), (0.107, 0.578, 0.377)),
        (6, (0.107, 0.578, 0.377), (0.113, 0.583, 0.379)),
        (0, (0.0, -0.5, -0.5), (0.0, -1.0, -0.6)),
    ]
    return _snapshots(rows, angle_degrees=0.0, offset=(0.0, 0.0, 0.0))


def _with_arbitrary_names(
    bones: tuple[RigBoneSnapshot, ...],
) -> tuple[RigBoneSnapshot, ...]:
    names = (
        "anchor",
        "cobalt",
        "quartz",
        "ember",
        "crown",
        "willow",
        "cinder",
        "marble",
        "opal",
        "rivet",
        "signal",
        "velvet",
        "linen",
        "bramble",
        "oxide",
        "keel",
        "wake",
    )
    if len(bones) != len(names):
        raise AssertionError("arbitrary-name fixture size changed")
    return tuple(
        RigBoneSnapshot(
            joint_id=f"opaque-joint-{index}",
            name=names[index],
            parent=bone.parent,
            head=bone.head,
            tail=bone.tail,
        )
        for index, bone in enumerate(bones)
    )


class SemanticNamingTest(unittest.TestCase):
    def _assert_reciprocal_pairs(self, plan) -> None:
        assignments_by_id = {
            assignment.joint_id: assignment for assignment in plan.assignments
        }
        left_pairs = 0
        for assignment in plan.assignments:
            if not assignment.mirror_joint_id:
                continue
            partner = assignments_by_id[assignment.mirror_joint_id]
            self.assertEqual(partner.mirror_joint_id, assignment.joint_id)
            self.assertEqual(partner.region, assignment.region)
            self.assertEqual(partner.group, assignment.group)
            self.assertEqual(partner.index, assignment.index)
            self.assertNotEqual(partner.side, assignment.side)
            left_pairs += assignment.side == "l"
        self.assertEqual(left_pairs, plan.mirror_pair_count)

    def test_biped_names_survive_z_rotation_and_center_offset(self) -> None:
        plan = build_naming_plan(
            make_biped(
                angle_degrees=31.0,
                offset=(4.5, -2.75, 6.0),
            ),
            TEMPLATE_BIPED,
        )
        names = {assignment.target_name for assignment in plan.assignments}
        self.assertTrue({"spine_1", "spine_2"}.issubset(names))
        self.assertTrue({"head_1", "head_2"}.issubset(names))
        self.assertTrue({"tail_1", "tail_2"}.issubset(names))
        self.assertTrue(
            {
                "leg_l_1",
                "leg_l_2",
                "leg_r_1",
                "leg_r_2",
                "arm_l_1",
                "arm_l_2",
                "arm_r_1",
                "arm_r_2",
            }.issubset(names)
        )
        self.assertEqual(plan.mirror_pair_count, 4)
        self.assertEqual(plan.rename_count, len(plan.assignments))
        self._assert_reciprocal_pairs(plan)
        lateral = np.asarray(plan.lateral_axis)
        self.assertAlmostEqual(float(np.linalg.norm(lateral)), 1.0, places=7)
        root = np.asarray(
            make_biped(
                angle_degrees=31.0,
                offset=(4.5, -2.75, 6.0),
            )[0].head
        )
        self.assertAlmostEqual(plan.center, float(np.dot(root, lateral)), places=7)

    def test_existing_quadruped_samples_keep_anatomical_roots(self) -> None:
        expectations = {
            "xiaobaozi_skin.txt": {
                "bone_0": "spine_1",
                "bone_1": "tail_1",
                "bone_8": "leg_hind_r_1",
                "bone_43": "leg_hind_l_1",
                "bone_27": "leg_front_r_1",
                "bone_35": "leg_front_l_1",
            },
            "cat5595_skin.txt": {
                "bone_0": "spine_1",
                "bone_45": "tail_1",
                "bone_21": "leg_front_l_1",
                "bone_27": "leg_front_r_1",
                "bone_33": "leg_hind_l_1",
                "bone_39": "leg_hind_r_1",
            },
            "lusia_skin.txt": {
                "bone_0": "spine_1",
                "bone_25": "tail_1",
                "bone_9": "leg_front_l_1",
                "bone_13": "leg_front_r_1",
                "bone_17": "leg_hind_l_1",
                "bone_21": "leg_hind_r_1",
            },
        }
        for filename, expected in expectations.items():
            with self.subTest(filename=filename):
                joints, parents, names = parse_rig_txt(
                    ROOT / "examples" / "samples" / filename
                )
                bones = tuple(
                    RigBoneSnapshot(
                        joint_id=f"sample-{index}",
                        name=name,
                        parent=parents[index],
                        head=tuple(joints[index]),
                        tail=tuple(_tail_for_joint(joints, parents, index)),
                    )
                    for index, name in enumerate(names)
                )
                plan = build_naming_plan(bones, TEMPLATE_QUADRUPED)
                actual = {
                    assignment.old_name: assignment.semantic_name
                    for assignment in plan.assignments
                }
                self.assertEqual(
                    {name: actual.get(name) for name in expected},
                    expected,
                )

    def test_quadruped_distinguishes_front_and_hind_after_transform(self) -> None:
        plan = build_naming_plan(
            make_quadruped(
                angle_degrees=47.0,
                offset=(-3.25, 5.5, -1.5),
            ),
            TEMPLATE_QUADRUPED,
        )
        names = {assignment.target_name for assignment in plan.assignments}
        expected_limbs = {
            f"leg_{group}_{side}_{segment}"
            for group in ("front", "hind")
            for side in ("l", "r")
            for segment in (1, 2)
        }
        self.assertTrue(expected_limbs.issubset(names))
        self.assertTrue({"spine_1", "spine_2", "spine_3"}.issubset(names))
        self.assertTrue({"head_1", "head_2"}.issubset(names))
        self.assertTrue({"tail_1", "tail_2"}.issubset(names))
        self.assertEqual(plan.mirror_pair_count, 4)
        self._assert_reciprocal_pairs(plan)

    def test_split_bone_names_are_renamed_after_session_metadata_is_lost(self) -> None:
        for split_name in ("bone_16_split", "bone_16_split_1"):
            with self.subTest(split_name=split_name):
                bones = list(make_quadruped())
                source = bones[1]
                bones[1] = RigBoneSnapshot(
                    joint_id=source.joint_id,
                    name=split_name,
                    parent=source.parent,
                    head=source.head,
                    tail=source.tail,
                )

                plan = build_naming_plan(tuple(bones), TEMPLATE_QUADRUPED)
                assignment = next(
                    item for item in plan.assignments if item.old_name == split_name
                )

                self.assertEqual(assignment.semantic_name, "spine_2")
                self.assertEqual(assignment.target_name, "spine_2")

    def test_quadruped_tail_may_hang_instead_of_pointing_backward(self) -> None:
        plan = build_naming_plan(
            make_quadruped_with_hanging_tail(),
            TEMPLATE_QUADRUPED,
        )
        actual = {
            assignment.old_name: assignment.semantic_name
            for assignment in plan.assignments
        }

        self.assertEqual(actual["bone_13"], "tail_1")
        self.assertEqual(actual["bone_14"], "tail_2")

    def test_short_center_attachment_is_not_mislabeled_as_tail(self) -> None:
        plan = build_naming_plan(
            make_tailless_quadruped_with_short_center_attachment(),
            TEMPLATE_QUADRUPED,
        )
        actual = {
            assignment.old_name: assignment.semantic_name
            for assignment in plan.assignments
        }

        self.assertNotIn("bone_13", actual)
        self.assertIn("未识别到尾部骨骼", plan.warnings)

    def test_quadruped_uses_large_pairs_when_front_limbs_do_not_point_down(self) -> None:
        plan = build_naming_plan(
            make_quadruped_with_raised_front_limbs_and_ears(),
            TEMPLATE_QUADRUPED,
        )
        actual = {
            assignment.old_name: assignment.semantic_name
            for assignment in plan.assignments
        }
        self.assertEqual(actual["bone_5"], "leg_hind_r_1")
        self.assertEqual(actual["bone_7"], "leg_hind_l_1")
        self.assertEqual(actual["bone_9"], "leg_front_r_1")
        self.assertEqual(actual["bone_11"], "leg_front_l_1")
        self.assertFalse(actual["bone_13"].startswith("leg_"))
        self.assertFalse(actual["bone_14"].startswith("leg_"))

    def test_third_pair_ambiguity_starts_at_eighty_five_percent(self) -> None:
        analysis = _RigAnalysis(make_quadruped(), TEMPLATE_QUADRUPED)
        main_path = (0, 1, 2, 3, 4)
        first = _BranchPair(
            attachment=0,
            left_root=5,
            right_root=7,
            similarity=1.0,
            extent=1.0,
        )
        second = _BranchPair(
            attachment=2,
            left_root=9,
            right_root=11,
            similarity=1.0,
            extent=1.0,
        )
        unambiguous_third = _BranchPair(
            attachment=3,
            left_root=13,
            right_root=14,
            similarity=1.0,
            extent=0.849,
        )
        selected = analysis.major_limb_pairs(
            main_path,
            (first, second, unambiguous_third),
        )
        self.assertEqual(set(selected), {first, second})

        ambiguous_third = _BranchPair(
            attachment=3,
            left_root=13,
            right_root=14,
            similarity=1.0,
            extent=0.85,
        )
        with self.assertRaisesRegex(RigPostprocessError, "过于接近"):
            analysis.major_limb_pairs(
                main_path,
                (first, second, ambiguous_third),
            )


class MirrorPairInferenceTest(unittest.TestCase):
    def test_two_arbitrarily_named_bones_are_an_explicit_pair(self) -> None:
        bones = (
            RigBoneSnapshot(
                joint_id="root-id",
                name="unrelated_root",
                parent=-1,
                head=(0.0, 0.0, 0.0),
                tail=(0.0, 0.0, 1.0),
            ),
            RigBoneSnapshot(
                joint_id="first-id",
                name="artist_control_B",
                parent=0,
                head=(1.4, 0.1, 0.8),
                tail=(1.8, 0.4, -0.2),
            ),
            RigBoneSnapshot(
                joint_id="second-id",
                name="helper.904",
                parent=0,
                head=(-0.8, 0.6, 1.1),
                tail=(-1.1, 0.9, 0.0),
            ),
        )

        inference = infer_selected_mirror_pairs(
            bones,
            ("helper.904", "artist_control_B"),
            axis=MIRROR_AXIS_X,
            center=0.0,
        )

        self.assertEqual(len(inference.pairs), 1)
        self.assertEqual(
            (
                inference.pairs[0].positive_name,
                inference.pairs[0].negative_name,
            ),
            ("artist_control_B", "helper.904"),
        )
        self.assertEqual(inference.unmatched_names, ())
        self.assertEqual(inference.ambiguous_names, ())

    def test_selected_limb_chains_and_ears_form_geometry_pairs(self) -> None:
        bones = _with_arbitrary_names(
            make_quadruped_with_raised_front_limbs_and_ears()
        )
        selected_indices = tuple(range(5, 15))

        inference = infer_selected_mirror_pairs(
            bones,
            tuple(bones[index].name for index in selected_indices),
            axis=MIRROR_AXIS_X,
            center=0.0,
        )

        actual = {
            (pair.positive_name, pair.negative_name)
            for pair in inference.pairs
        }
        expected = {
            (bones[5].name, bones[7].name),
            (bones[6].name, bones[8].name),
            (bones[9].name, bones[11].name),
            (bones[10].name, bones[12].name),
            (bones[13].name, bones[14].name),
        }
        self.assertEqual(actual, expected)
        self.assertEqual(inference.unmatched_names, ())
        self.assertEqual(inference.ambiguous_names, ())

    def test_center_and_unmatched_selected_bones_are_skipped(self) -> None:
        bones = _with_arbitrary_names(
            make_quadruped_with_raised_front_limbs_and_ears()
        )
        selected_indices = (0, 5, 7, 13)

        inference = infer_selected_mirror_pairs(
            bones,
            tuple(bones[index].name for index in selected_indices),
            axis=MIRROR_AXIS_X,
            center=0.0,
        )

        self.assertEqual(
            {
                (pair.positive_name, pair.negative_name)
                for pair in inference.pairs
            },
            {(bones[5].name, bones[7].name)},
        )
        self.assertEqual(
            set(inference.unmatched_names),
            {bones[0].name, bones[13].name},
        )
        self.assertEqual(inference.ambiguous_names, ())

    def test_multisegment_ears_keep_the_middle_pair(self) -> None:
        bones = make_multisegment_ears_with_shared_attachment()

        inference = infer_selected_mirror_pairs(
            bones,
            tuple(bones[index].name for index in range(1, 8)),
            axis=MIRROR_AXIS_X,
            center=0.0,
        )

        self.assertEqual(
            {
                (pair.positive_name, pair.negative_name)
                for pair in inference.pairs
            },
            {
                (bones[5].name, bones[2].name),
                (bones[6].name, bones[3].name),
                (bones[7].name, bones[4].name),
            },
        )
        self.assertEqual(inference.unmatched_names, (bones[1].name,))
        self.assertEqual(inference.ambiguous_names, ())


class MirrorFormulaTest(unittest.TestCase):
    def test_opposite_bone_directions_are_rejected_before_alignment(self) -> None:
        with self.assertRaisesRegex(RigPostprocessError, "方向不一致"):
            _symmetric_bone_targets(
                np.asarray((1.0, 0.0, 0.0)),
                np.asarray((1.0, 1.0, 0.0)),
                np.asarray((-1.0, 0.0, 0.0)),
                np.asarray((-1.0, -1.0, 0.0)),
                MIRROR_AXIS_X,
                0.0,
                MIRROR_DISTANCE_AVERAGE,
            )

    def test_symmetric_points_support_each_axis_with_offset_center(self) -> None:
        center = 1.5
        for axis, axis_index in (
            (MIRROR_AXIS_X, 0),
            (MIRROR_AXIS_Y, 1),
            (MIRROR_AXIS_Z, 2),
        ):
            with self.subTest(axis=axis):
                left = np.asarray((2.0, 4.0, 6.0))
                right = np.asarray((8.0, 10.0, 12.0))
                left[axis_index] = center + 4.0
                right[axis_index] = center - 2.0

                mirrored_left, mirrored_right = _symmetric_points(
                    left,
                    right,
                    axis,
                    center,
                    MIRROR_DISTANCE_AVERAGE,
                )

                self.assertAlmostEqual(
                    mirrored_left[axis_index],
                    center + 3.0,
                    places=7,
                )
                self.assertAlmostEqual(
                    mirrored_right[axis_index],
                    center - 3.0,
                    places=7,
                )
                tangent_indices = [index for index in range(3) if index != axis_index]
                expected_tangent = 0.5 * (left + right)
                np.testing.assert_allclose(
                    mirrored_left[tangent_indices],
                    expected_tangent[tangent_indices],
                    atol=1e-9,
                )
                np.testing.assert_allclose(
                    mirrored_right[tangent_indices],
                    expected_tangent[tangent_indices],
                    atol=1e-9,
                )

    def test_distance_mode_selects_farthest_nearest_or_average(self) -> None:
        center = -0.75
        left = np.asarray((center + 4.0, 2.0, 6.0))
        right = np.asarray((center - 2.0, 8.0, 10.0))
        expectations = {
            MIRROR_DISTANCE_FARTHEST: 4.0,
            MIRROR_DISTANCE_NEAREST: 2.0,
            MIRROR_DISTANCE_AVERAGE: 3.0,
        }
        for distance_mode, expected_distance in expectations.items():
            with self.subTest(distance_mode=distance_mode):
                mirrored_left, mirrored_right = _symmetric_points(
                    left,
                    right,
                    MIRROR_AXIS_X,
                    center,
                    distance_mode,
                )
                self.assertAlmostEqual(
                    mirrored_left[0] - center,
                    expected_distance,
                    places=7,
                )
                self.assertAlmostEqual(
                    center - mirrored_right[0],
                    expected_distance,
                    places=7,
                )

    def test_rotated_or_invalid_axis_is_rejected(self) -> None:
        left = np.asarray((1.0, 0.0, 0.0))
        right = np.asarray((-1.0, 0.0, 0.0))
        for invalid_axis in ("XY", "", np.asarray((1.0, 1.0, 0.0))):
            with self.subTest(axis=repr(invalid_axis)):
                with self.assertRaises(RigPostprocessError):
                    _symmetric_points(
                        left,
                        right,
                        invalid_axis,
                        0.0,
                        MIRROR_DISTANCE_AVERAGE,
                    )


if __name__ == "__main__":
    unittest.main()
