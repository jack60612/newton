# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo
from newton.tests.unittest_utils import USD_AVAILABLE


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestImportUsdArmature(unittest.TestCase):
    def _stage(self, *, d6=False, axes=None, scalar=None):
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        body = UsdGeom.Xform.Define(stage, "/Body")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(2.0)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.2, 0.3, 0.4))
        if d6:
            joint = UsdPhysics.Joint.Define(stage, "/Joint")
            joint.CreateBody1Rel().SetTargets([body.GetPath()])
            prim = joint.GetPrim()
            # Explicitly author six unlocked axes. Without LimitAPI instances,
            # USD's joint description does not supply D6 axis records.
            for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
                limit = UsdPhysics.LimitAPI.Apply(prim, axis)
                limit.CreateLowAttr(-1.0)
                limit.CreateHighAttr(1.0)
        else:
            prim = body.GetPrim()
        if scalar is not None:
            prim.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float, True).Set(scalar)
        for name, value in (axes or {}).items():
            prim.CreateAttribute(f"newton:{name}:armature", Sdf.ValueTypeNames.Float, True).Set(value)
        return stage

    def test_free_six_axis_armature_matches_c_mujoco(self):
        """USD authors native rotor inertia before import; no postload repair."""
        names = ("transX", "transY", "transZ", "rotX", "rotY", "rotZ")
        expected = np.array([0.0, 0.0, 0.0, 0.001, 0.002, 0.004])
        builder = newton.ModelBuilder()
        builder.add_usd(self._stage(axes=dict(zip(names, expected, strict=True))))
        self.assertEqual(builder.joint_type, [newton.JointType.FREE])
        np.testing.assert_allclose(builder.joint_armature, expected, rtol=1e-6, atol=0.0)
        with wp.ScopedDevice("cpu"):
            model = builder.finalize()
            solver = SolverMuJoCo(model, use_mujoco_cpu=True)
        np.testing.assert_allclose(solver.mj_model.dof_armature, expected, rtol=1e-6, atol=0.0)
        np.testing.assert_allclose(solver.mjw_model.dof_armature.numpy()[0], expected, rtol=1e-6, atol=0.0)

        mujoco = SolverMuJoCo.import_mujoco()[0]
        reference = mujoco.MjModel.from_xml_string('''
        <mujoco><option gravity="0 0 0"/><worldbody><body>
          <freejoint/>
          <inertial pos="0 0 0" mass="2" diaginertia="0.2 0.3 0.4"/>
        </body></worldbody></mujoco>
        ''')
        # MJCF only has scalar armature; C MuJoCo's native per-DOF data
        # is the independent reference for anisotropic FREE joints.
        reference.dof_armature[:] = expected
        reference_data = mujoco.MjData(reference)
        mujoco.mj_setConst(reference, reference_data)
        solver.mj_model.opt.gravity[:] = 0.0
        native_data = mujoco.MjData(solver.mj_model)
        force = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        reference_data.qfrc_applied[:] = force
        native_data.qfrc_applied[:] = force
        mujoco.mj_forward(reference, reference_data)
        mujoco.mj_forward(solver.mj_model, native_data)
        np.testing.assert_allclose(native_data.qacc, reference_data.qacc, rtol=1e-6, atol=1e-8)

    def test_d6_axis_override_preserves_scalar_fallback(self):
        builder = newton.ModelBuilder()
        builder.add_usd(self._stage(d6=True, scalar=0.5, axes={"transX": 0.0, "rotY": 0.125}))
        self.assertEqual(builder.joint_type, [newton.JointType.D6])
        self.assertEqual(builder.joint_dof_dim, [(3, 3)])
        np.testing.assert_allclose(builder.joint_armature, [0.0, 0.5, 0.5, 0.5, 0.125, 0.5])

    def test_free_unauthored_axes_remain_zero(self):
        builder = newton.ModelBuilder()
        builder.add_usd(self._stage(axes={"rotZ": 0.125}, scalar=42.0))
        # Body-level scalar armature remains ignored; it is not body inertia.
        np.testing.assert_allclose(builder.joint_armature, [0.0, 0.0, 0.0, 0.0, 0.0, 0.125])


if __name__ == "__main__":
    unittest.main()
