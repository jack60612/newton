# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton._src.solvers.mujoco.sizing import _contact_points, _scene_constraint_capacity, _structural_constraint_rows
from newton.solvers import SolverMuJoCo


class TestMuJoCoSceneSizing(unittest.TestCase):
    def capacity(self, model):
        _, mujoco_warp = SolverMuJoCo.import_mujoco()
        with wp.ScopedDevice("cpu"):
            return _scene_constraint_capacity(model, mujoco_warp.put_model(model))

    def test_no_dispatch_pairs_reserve_broadphase_capacity(self):
        mujoco, mujoco_warp = SolverMuJoCo.import_mujoco()
        model = mujoco.MjModel.from_xml_string("""
            <mujoco><asset>
            <hfield name="terrain" nrow="2" ncol="2" size="1 1 1 .1"/>
            </asset><worldbody>
            <geom name="floor" type="plane" size="1 1 .1"/>
            <body mocap="true"><geom name="terrain" type="hfield" hfield="terrain"/></body>
            </worldbody><contact>
            <pair geom1="floor" geom2="terrain"/>
            </contact></mujoco>
        """)
        self.assertEqual(_contact_points(model, 0, 1), 0)
        # Explicit contact pairs override DOF-less body filtering in the
        # broadphase inventory, even when the pair has no narrowphase dispatch.
        with wp.ScopedDevice("cpu"):
            device_model = mujoco_warp.put_model(model)
            candidates = len(device_model.nxn_geom_pair_filtered.numpy())
            self.assertGreater(candidates, 0)
            rows, contacts = _scene_constraint_capacity(model, device_model)
            self.assertEqual(rows, 1)
            self.assertEqual(contacts, candidates)
            data = mujoco_warp.put_data(model, mujoco.MjData(model), njmax=rows, nconmax=contacts)
            self.assertEqual(data.naconmax, candidates)

    def test_native_inactive_equality_is_not_a_capacity_bound(self):
        mujoco, _ = SolverMuJoCo.import_mujoco()
        # Independent C MuJoCo model: no Newton conversion or farm-sim code.
        model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><body name="b"><freejoint/>
            <geom type="sphere" size=".1" contype="0" conaffinity="0"/>
            </body></worldbody><equality>
            <connect body1="b" anchor="0 0 0" active="false"/>
            </equality></mujoco>
        """)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(data.nefc, 0)
        self.assertEqual(self.capacity(model), (4, 0))
        data.eq_active[0] = True
        mujoco.mj_forward(model, data)
        self.assertEqual(data.nefc, 3)

    def test_future_contact_and_explicit_pair_dimension(self):
        mujoco, _ = SolverMuJoCo.import_mujoco()
        model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody>
            <geom name="floor" type="plane" size="1 1 .1"/>
            <body pos="0 0 2"><freejoint/>
            <geom name="box" type="box" size=".1 .1 .1"/>
            </body></worldbody><contact>
            <pair geom1="floor" geom2="box" condim="6"/>
            </contact></mujoco>
        """)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertEqual(data.ncon, 0)
        self.assertEqual(self.capacity(model), (80, 8))
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.assertEqual(self.capacity(model), (48, 8))
        data.qpos[2] = 0.05
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)
        self.assertLessEqual(data.ncon, 8)
        self.assertLessEqual(data.nefc, 48)

    def test_inactive_limits_and_friction(self):
        mujoco, _ = SolverMuJoCo.import_mujoco()
        model = mujoco.MjModel.from_xml_string("""
            <mujoco><compiler angle="radian"/><worldbody>
            <body><joint name="j" range="-1 1" frictionloss=".1"/>
            <geom type="sphere" size=".1" contype="0" conaffinity="0"/>
            </body></worldbody><tendon>
            <fixed name="t" limited="true" range="-2 2" frictionloss=".1">
            <joint joint="j" coef="1"/></fixed></tendon></mujoco>
        """)
        self.assertEqual(_structural_constraint_rows(model), 4)
        self.assertEqual(self.capacity(model), (4, 0))

    def test_native_solver_allocates_future_contacts_per_world(self):
        template = newton.ModelBuilder()
        body = template.add_link(xform=wp.transform(wp.vec3(0, 0, 2), wp.quat_identity()))
        joint = template.add_joint_free(body)
        template.add_articulation([joint])
        template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        template.add_ground_plane()
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        with wp.ScopedDevice("cpu"):
            solver = SolverMuJoCo(builder.finalize(device="cpu"), separate_worlds=True)
        expected_rows, expected_contacts = _scene_constraint_capacity(solver.mj_model, solver.mjw_model)
        self.assertEqual(solver.mjw_data.njmax, expected_rows)
        self.assertEqual(solver.mjw_data.naconmax, 2 * expected_contacts)
        self.assertGreater(expected_contacts, solver.mj_data.ncon)

    def test_explicit_capacities_are_preserved(self):
        builder = newton.ModelBuilder()
        body = builder.add_link()
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])
        builder.add_shape_sphere(body, radius=0.1)
        with wp.ScopedDevice("cpu"):
            model = builder.finalize(device="cpu")
            solver = SolverMuJoCo(model, njmax=17, nconmax=19)
        self.assertEqual(solver.mjw_data.njmax, 17)
        self.assertEqual(solver.mjw_data.naconmax, 19)

    def test_zero_constraint_scene_keeps_pinned_integrator_path(self):
        mujoco, _ = SolverMuJoCo.import_mujoco()
        model = mujoco.MjModel.from_xml_string("""
            <mujoco><worldbody><body><freejoint/>
            <geom size=".1" contype="0" conaffinity="0"/>
            </body></worldbody></mujoco>
        """)
        self.assertEqual(self.capacity(model), (1, 0))


if __name__ == "__main__":
    unittest.main()
