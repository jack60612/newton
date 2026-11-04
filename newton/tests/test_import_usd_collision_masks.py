# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test authored MuJoCo collision masks imported from USD."""

import itertools
import unittest

import numpy as np

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE

if USD_AVAILABLE:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestUsdCollisionMasks(unittest.TestCase):
    def _stage(self, body_masks):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        for body_index, masks in enumerate(body_masks):
            body = UsdGeom.Xform.Define(stage, f"/body{body_index}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            for shape_index, mask in enumerate(masks):
                sphere = UsdGeom.Sphere.Define(stage, f"/body{body_index}/shape{shape_index}")
                sphere.CreateRadiusAttr(0.1)
                prim = sphere.GetPrim()
                UsdPhysics.CollisionAPI.Apply(prim)
                if mask is not None:
                    for name, value in zip(("mjc:contype", "mjc:conaffinity"), mask, strict=True):
                        prim.CreateAttribute(name, Sdf.ValueTypeNames.Int, custom=True).Set(value)
        return stage

    def _builder(self):
        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        return builder

    def _solver(self, model):
        return SolverMuJoCo(model, use_mujoco_contacts=True, use_mujoco_cpu=True)

    def _masks_allow(self, solver, shape_a, shape_b):
        shape_to_geom = {
            int(shape): geom for geom, shape in enumerate(solver.mjc_geom_to_newton_shape.numpy()[0]) if shape >= 0
        }
        geom_a, geom_b = shape_to_geom[shape_a], shape_to_geom[shape_b]
        contype, conaffinity = solver.mj_model.geom_contype, solver.mj_model.geom_conaffinity
        return bool(
            (int(contype[geom_a]) & int(conaffinity[geom_b])) or (int(contype[geom_b]) & int(conaffinity[geom_a]))
        )

    def test_forward_authored_masks(self):
        """Forward raw asymmetric masks, including zero and signed high bits."""
        for masks in (((0, 1), (1, 0), (1, 0)), ((-2147483648, 0), (0, -2147483648), (-1, 0))):
            with self.subTest(masks=masks):
                builder = self._builder()
                result = builder.add_usd(self._stage([[mask] for mask in masks]), load_visual_shapes=False)
                model = builder.finalize(device="cpu")
                domains = model.mujoco.collision_mask_domain.numpy()
                self.assertTrue(np.all(domains >= 0))
                self.assertEqual(len(np.unique(domains)), 1)
                shapes = [result["path_shape_map"][f"/body{i}/shape0"] for i in range(3)]
                expected_pairs = {tuple(sorted((shapes[0], shapes[1]))), tuple(sorted((shapes[1], shapes[2])))}
                if masks[0][0] == 0:
                    expected_pairs = {tuple(sorted((shapes[0], shapes[1]))), tuple(sorted((shapes[0], shapes[2])))}
                self.assertEqual(set(map(tuple, model.shape_contact_pairs.numpy())), expected_pairs)
                solver = self._solver(model)
                mapping = solver.mjc_geom_to_newton_shape.numpy()[0]
                for geom, shape in enumerate(mapping):
                    if shape >= 0:
                        index = shapes.index(int(shape))
                        self.assertEqual(int(solver.mj_model.geom_contype[geom]), masks[index][0])
                        self.assertEqual(int(solver.mj_model.geom_conaffinity[geom]), masks[index][1])

    def test_filtered_pairs_narrow_masks(self):
        """Keep explicit filtered pairs and mask restrictions during solver fallback."""
        stage = self._stage([[(1, 0), (2, 0)], [(0, 1), (0, 3)]])
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/body0/shape0")).CreateFilteredPairsRel().AddTarget(
            Sdf.Path("/body1/shape0")
        )
        builder = self._builder()
        result = builder.add_usd(stage, load_visual_shapes=False)
        shapes = result["path_shape_map"]
        model = builder.finalize(device="cpu")
        expected = {
            tuple(sorted((shapes["/body0/shape0"], shapes["/body1/shape1"]))),
            tuple(sorted((shapes["/body0/shape1"], shapes["/body1/shape1"]))),
        }
        self.assertEqual(set(map(tuple, model.shape_contact_pairs.numpy())), expected)
        solver = self._solver(model)
        for a in range(2):
            for b in range(2):
                shape_a, shape_b = shapes[f"/body0/shape{a}"], shapes[f"/body1/shape{b}"]
                self.assertEqual(self._masks_allow(solver, shape_a, shape_b), b == 1)

    def test_separate_import_domains(self):
        """Keep repeated USD imports independent when compiling collision masks."""
        builder = self._builder()
        stage = self._stage([[(0, 1)], [(1, 0)], [(1, 0)]])
        builder.add_usd(stage, load_visual_shapes=False)
        builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize(device="cpu")
        domains = model.mujoco.collision_mask_domain.numpy()
        self.assertEqual(len(np.unique(domains)), 2)
        solver = self._solver(model)
        expected = set(map(tuple, model.shape_contact_pairs.numpy()))
        for a in range(model.shape_count):
            for b in range(a + 1, model.shape_count):
                self.assertEqual(self._masks_allow(solver, a, b), (a, b) in expected)

    def test_large_authored_masks_preserve_filtered_static_hardware(self):
        """Preserve all native pairs above 256 shapes and 32 independent fruit bodies."""
        import mujoco

        for apple_apple, apple_wood in itertools.product((False, True), repeat=2):
            with self.subTest(apple_apple=apple_apple, apple_wood=apple_wood):
                wood = (1, 2)
                apple = (4, 2 | (4 if apple_apple else 0) | (1 if apple_wood else 0))
                hardware = (2, 1)
                body_masks = [[wood] * 112, [wood] * 113] + [[apple] for _ in range(34)]
                body_masks += [[hardware], [hardware]]
                stage = self._stage(body_masks)
                # Compile separated bodies so constructor forward dynamics does
                # not solve thousands of deliberately overlapping contacts.
                for body_index in range(36):
                    UsdGeom.Xformable(stage.GetPrimAtPath(f"/body{body_index}")).AddTranslateOp().Set(
                        Gf.Vec3d(float(body_index), 0.0, 0.0)
                    )
                for index in (36, 37):
                    stage.GetPrimAtPath(f"/body{index}").RemoveAPI(UsdPhysics.RigidBodyAPI)
                UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/body0")).CreateFilteredPairsRel().AddTarget(
                    Sdf.Path("/body36/shape0")
                )
                builder = self._builder()
                result = builder.add_usd(stage, load_visual_shapes=False)
                model = builder.finalize(device="cpu")
                self.assertGreater(model.shape_count, 256)
                solver = self._solver(model)
                shape_to_geom = {
                    int(shape): geom for geom, shape in enumerate(solver.mjc_geom_to_newton_shape.numpy()[0])
                }
                shapes = result["path_shape_map"]
                for body_index, masks in enumerate(body_masks):
                    for shape_index, mask in enumerate(masks):
                        geom = shape_to_geom[shapes[f"/body{body_index}/shape{shape_index}"]]
                        self.assertEqual(int(solver.mj_model.geom_contype[geom]), mask[0])
                        self.assertEqual(int(solver.mj_model.geom_conaffinity[geom]), mask[1])
                wire = shapes["/body36/shape0"]
                self.assertEqual(int(model.shape_body.numpy()[wire]), -1)
                self.assertGreater(int(solver.mj_model.geom_bodyid[shape_to_geom[wire]]), 0)

                # Then overlap every sphere for collision detection only: verify
                # all allowed/forbidden pairs without an unrelated dense solve.
                # Static hardware remains at its compiled origin/BVH throughout.
                data = mujoco.MjData(solver.mj_model)
                for joint, kind in enumerate(solver.mj_model.jnt_type):
                    self.assertEqual(int(kind), int(mujoco.mjtJoint.mjJNT_FREE))
                    address = int(solver.mj_model.jnt_qposadr[joint])
                    data.qpos[address : address + 3] = 0.0
                mujoco.mj_kinematics(solver.mj_model, data)
                mujoco.mj_collision(solver.mj_model, data)
                mapping = solver.mjc_geom_to_newton_shape.numpy()[0]
                actual = {
                    tuple(sorted((int(mapping[contact.geom1]), int(mapping[contact.geom2]))))
                    for contact in data.contact[: data.ncon]
                }
                expected = set(map(tuple, model.shape_contact_pairs.numpy()))
                self.assertEqual(actual, expected)

    def test_partial_static_filter_above_compiler_cap_fails_closed(self):
        """Reject an unrepresentable partial-owner filter without enabling extra contacts."""
        stage = self._stage([[(1, 0)] * 257, [(0, 1)]])
        stage.GetPrimAtPath("/body1").RemoveAPI(UsdPhysics.RigidBodyAPI)
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/body0/shape0")).CreateFilteredPairsRel().AddTarget(
            Sdf.Path("/body1/shape0")
        )
        builder = self._builder()
        builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "Cannot represent Newton collision filters exactly"):
            self._solver(model)

    def test_connect_anchors_with_filtered_static_wire(self):
        """Keep CONNECT world anchors unchanged when a filtered static wire gains a native body."""
        stage = self._stage([[(1, 0)], [(0, 1)], [(1, 0)]])
        stage.GetPrimAtPath("/body1").RemoveAPI(UsdPhysics.RigidBodyAPI)
        UsdGeom.Xformable(stage.GetPrimAtPath("/body0")).AddTranslateOp().Set(Gf.Vec3d(0.0, 2.0, 3.0))
        UsdGeom.Xformable(stage.GetPrimAtPath("/body1")).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath("/body0")).CreateFilteredPairsRel().AddTarget(
            Sdf.Path("/body1/shape0")
        )
        connect = UsdPhysics.SphericalJoint.Define(stage, "/connect")
        connect.CreateBody0Rel().SetTargets([Sdf.Path("/body0")])
        connect.CreateBody1Rel().SetTargets([Sdf.Path("/body1")])
        connect.CreateLocalPos0Attr(Gf.Vec3f(1.0, 0.0, 0.0))
        connect.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        connect.CreateExcludeFromArticulationAttr(True)
        connect.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["MjcEqualityConnectAPI"]))
        builder = self._builder()
        result = builder.add_usd(stage, load_visual_shapes=False)
        model = builder.finalize(device="cpu")
        solver = self._solver(model)
        self.assertEqual(solver.mj_model.neq, 1)
        np.testing.assert_allclose(solver.mj_model.eq_data[0, :3], [1.0, 0.0, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(solver.mj_model.eq_data[0, 3:6], [1.0, 2.0, 3.0], atol=1.0e-6)
        self.assertEqual(int(solver.mj_model.eq_obj2id[0]), 0)
        wire = result["path_shape_map"]["/body1/shape0"]
        shape_to_geom = {int(shape): geom for geom, shape in enumerate(solver.mjc_geom_to_newton_shape.numpy()[0])}
        wire_body = int(solver.mj_model.geom_bodyid[shape_to_geom[wire]])
        self.assertGreater(wire_body, 0)
        self.assertEqual(int(solver.mj_model.body_weldid[wire_body]), 0)

    def test_unauthored_masks_have_no_domain(self):
        """Leave unauthored shapes outside preserved collision mask domains."""
        builder = self._builder()
        result = builder.add_usd(self._stage([[(0, 1)], [None]]), load_visual_shapes=False)
        model = builder.finalize(device="cpu")
        shapes = result["path_shape_map"]
        domains = model.mujoco.collision_mask_domain.numpy()
        self.assertGreaterEqual(int(domains[shapes["/body0/shape0"]]), 0)
        self.assertEqual(int(domains[shapes["/body1/shape0"]]), -1)
        solver = self._solver(model)
        self.assertTrue(self._masks_allow(solver, shapes["/body0/shape0"], shapes["/body1/shape0"]))


if __name__ == "__main__":
    unittest.main()
