# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestImportUsdWorldRoots(unittest.TestCase):
    @staticmethod
    def _stage():
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        support = UsdGeom.Xform.Define(stage, "/Support")
        support.AddTranslateOp().Set(Gf.Vec3d(3.0, -2.0, 1.0))
        support.AddOrientOp().Set(Gf.Quatf(np.cos(0.3), 0.0, 0.0, np.sin(0.3)))
        UsdPhysics.ArticulationRootAPI.Apply(support.GetPrim())
        # A world-welded support is an ordinary static Xform, not a rigid body.
        offsets = {"Left": (0.5, 0.2, 0.4), "Right": (-0.4, 0.7, 0.8), "Tip": (0.5, 0.2, 1.1)}
        anchors = {"Left": (0.5, 0.2, 0.0), "Right": (-0.4, 0.7, 0.0), "Tip": (0.0, 0.0, 0.3)}
        for name, offset in offsets.items():
            body = UsdGeom.Xform.Define(stage, f"/Support/{name}")
            body.AddTranslateOp().Set(Gf.Vec3d(*offset))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
            mass.CreateMassAttr(1.0)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Support/{name}Joint")
            parent = "/Support/Left" if name == "Tip" else "/Support"
            joint.CreateBody0Rel().SetTargets([parent])
            joint.CreateBody1Rel().SetTargets([body.GetPath()])
            joint.CreateAxisAttr("Z")
            joint.CreateLocalPos0Attr(Gf.Vec3f(*anchors[name]))
            parent_offset = np.array(offsets["Left"]) if name == "Tip" else np.zeros(3)
            child_anchor = parent_offset + np.array(anchors[name]) - np.array(offset)
            joint.CreateLocalPos1Attr(Gf.Vec3f(*child_anchor))
        return stage, offsets

    def _assert_pose(self, actual, expected):
        np.testing.assert_allclose(actual[:3], np.array(expected.p), atol=3.0e-6, rtol=0.0)
        self.assertAlmostEqual(abs(np.dot(actual[3:], np.array(expected.q))), 1.0, delta=2.0e-6)

    def test_all_world_roots_preserve_placement_and_reference_pose(self):
        incoming = wp.transform((1.2, 2.3, -0.6), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.4))
        support = wp.transform((3.0, -2.0, 1.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.6))
        for override in (False, True):
            with self.subTest(override_root_xform=override), wp.ScopedDevice("cpu"):
                stage, offsets = self._stage()
                builder = newton.ModelBuilder()
                builder.add_usd(stage, xform=incoming, override_root_xform=override)
                self.assertEqual(builder.articulation_count, 1)
                self.assertNotIn("/Support", builder.body_label)
                self.assertEqual(builder.joint_parent.count(-1), 2)
                self.assertTrue(all(kind == newton.JointType.REVOLUTE for kind in builder.joint_type))
                model = builder.finalize()
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                # Native equality anchors also use FK at qref, not imported body_q.
                ref_q = SolverMuJoCo._build_ref_q(model)
                ref_poses = SolverMuJoCo._compute_body_poses_at_qref(model, ref_q).numpy()
                solver = SolverMuJoCo(model, use_mujoco_cpu=True)
                mujoco = SolverMuJoCo.import_mujoco()[0]
                solver.mj_data.qpos[:] = solver.mj_model.qpos0
                mujoco.mj_forward(solver.mj_model, solver.mj_data)
                placement = incoming if override else incoming * support
                for name, offset in offsets.items():
                    label = f"/Support/{name}"
                    body = builder.body_label.index(label)
                    expected = placement * wp.transform(offset, wp.quat_identity())
                    for poses in (model.body_q.numpy(), state.body_q.numpy(), ref_poses):
                        self._assert_pose(poses[body], expected)
                    native_body = solver.mj_model.body(label.replace("/", "_")).id
                    native_pose = np.concatenate(
                        (solver.mj_data.xpos[native_body], solver.mj_data.xquat[native_body][[1, 2, 3, 0]])
                    )
                    self._assert_pose(native_pose, expected)
                    joint = builder.joint_label.index(f"/Support/{name}Joint")
                    parent = builder.joint_parent[joint]
                    parent_pose = wp.transform_identity()
                    if parent != -1:
                        pose = state.body_q.numpy()[parent]
                        parent_pose = wp.transform(pose[:3], pose[3:])
                    anchor = parent_pose * builder.joint_X_p[joint]
                    child_anchor = expected * builder.joint_X_c[joint]
                    np.testing.assert_allclose(np.array(anchor.p), np.array(child_anchor.p), atol=3.0e-6, rtol=0.0)
                    native_joint = np.flatnonzero(solver.mj_model.jnt_bodyid == native_body)
                    self.assertEqual(len(native_joint), 1)
                    np.testing.assert_allclose(
                        solver.mj_data.xanchor[native_joint[0]], np.array(child_anchor.p), atol=3.0e-6, rtol=0.0
                    )

    def test_mobility_override_only_replaces_first_root(self):
        for floating in (False, True):
            with self.subTest(floating=floating), wp.ScopedDevice("cpu"):
                stage, _ = self._stage()
                builder = newton.ModelBuilder()
                builder.add_usd(stage, xform=wp.transform((2.0, 3.0, 4.0), wp.quat_identity()), floating=floating)
                roots = [i for i, parent in enumerate(builder.joint_parent) if parent == -1]
                self.assertEqual(len(roots), 2)
                expected_type = newton.JointType.FREE if floating else newton.JointType.FIXED
                self.assertEqual(builder.joint_type[roots[0]], expected_type)
                self.assertEqual(builder.joint_type[roots[1]], newton.JointType.REVOLUTE)
                tip = builder.joint_label.index("/Support/TipJoint")
                self.assertEqual(builder.joint_type[tip], newton.JointType.REVOLUTE)
                self.assertEqual(builder.joint_parent[tip], builder.body_label.index("/Support/Left"))
                model = builder.finalize()
                state = model.state()
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)
                for actual, expected in zip(state.body_q.numpy(), model.body_q.numpy(), strict=True):
                    self._assert_pose(actual, wp.transform(expected[:3], expected[3:]))


if __name__ == "__main__":
    unittest.main()
