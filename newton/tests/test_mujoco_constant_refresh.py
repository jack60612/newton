# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Numeric release refresh parity with the existing host notification path."""

from unittest import mock

import numpy as np
import pytest
import warp as wp

import newton
from newton import ModelFlags
from newton.solvers import SolverMuJoCo

_MJCF = """
<mujoco>
  <worldbody>
    <site name="reference" pos="0 0 1"/>
    <site name="anchor" pos="0 0 2"/>
    <body name="root" pos="0 0 1">
      <freejoint name="root_free"/>
      <geom type="box" size="0.05 0.07 0.1" mass="1"/>
      <site name="tip" pos="0 0 0.2"/>
      <body name="child" pos="0 0 0.3">
        <joint name="hinge" type="hinge" limited="true" range="-0.5 0.5"/>
        <geom type="sphere" size="0.05" mass="0.2"/>
      </body>
    </body>
  </worldbody>
  <equality><connect body1="root" anchor="0 0 1"/></equality>
  <tendon><spatial name="stem" stiffness="20" damping="0.1">
    <site site="anchor"/><site site="tip"/>
  </spatial></tendon>
  <actuator>
    <general name="rotation" site="tip" refsite="reference" gear="0 0 0 1 0 0"
             gainprm="10" biastype="affine" biasprm="0 -10 -0.2"/>
    <position name="hinge_drive" joint="hinge" kp="10" dampratio="1"/>
    <general name="retained_dampratio" site="tip" refsite="reference" gear="0 0 0 1 0 0"
             gainprm="10" biastype="affine" biasprm="0 -10 1"/>
  </actuator>
</mujoco>
"""


@wp.kernel
def _release_rotor(mask: wp.array[wp.bool], mapping: wp.array2d[int], source: wp.array[float], live: wp.array2d[float]):
    world, dof = wp.tid()
    if mask[world] and dof < 6:
        source[mapping[world, dof]] = 0.0
        live[world, dof] = 0.0


@wp.kernel
def _zero_float(mask: wp.array[wp.bool], width: int, values: wp.array[float]):
    i = wp.tid()
    if mask[i // width]:
        values[i] = 0.0


@wp.kernel
def _zero_first_actuator(mask: wp.array[wp.bool], width: int, values: wp.array[float]):
    i = wp.tid()
    if mask[i // width] and i % width < 10:
        values[i] = 0.0


@wp.kernel
def _zero_bool(mask: wp.array[wp.bool], width: int, values: wp.array[wp.bool]):
    i = wp.tid()
    if mask[i // width]:
        values[i] = False


def _flat(array):
    count = array.size * wp.types.type_size_in_bytes(array.dtype) // 4
    return wp.array(ptr=array.ptr, dtype=float, shape=(count,), device=array.device)


def _make_solver(device, kinematic):
    with wp.ScopedDevice(device):
        template = newton.ModelBuilder()
        template.add_mjcf(_MJCF, ctrl_direct=True, convert_mjc_equality_constraints=False)
        if kinematic:
            body = template.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=True)
            template.add_shape_sphere(body, radius=0.05)
        builder = newton.ModelBuilder()
        builder.replicate(template, 2)
        model = builder.finalize()
        rotor = model.joint_armature.numpy()
        width = len(rotor) // 2
        for world in range(2):
            rotor[world * width + 3 : world * width + 6] = 0.0014435225
        model.joint_armature.assign(rotor)
        return SolverMuJoCo(model, iterations=2, disable_contacts=False, enable_sleeping=False)


def _release(solver, mask):
    m = solver.mjw_model
    wp.launch(
        _release_rotor,
        m.dof_armature.shape,
        inputs=[mask, solver.mjc_dof_to_newton_dof, solver.model.joint_armature, m.dof_armature],
    )
    # These are the source/live numeric writes a release caller owns. Configuration
    # and layout flags remain fixed. Host notification performs their mapping again.
    for source, live in (
        (solver.model.joint_damping, m.dof_damping),
        (solver.model.mujoco.dof_passive_stiffness, m.jnt_stiffness),
        (solver.model.mujoco.tendon_stiffness, m.tendon_stiffness),
        (solver.model.mujoco.tendon_damping, m.tendon_damping),
    ):
        for array in (source, live):
            flat = _flat(array)
            wp.launch(_zero_float, flat.size, inputs=[mask, flat.size // 2, flat])
    # Retain the hinge and second site/refsite dampratio actuator. Their damping
    # must be re-resolved against the released FREE root's changed reflected mass.
    for source, live in (
        (solver.model.mujoco.actuator_gainprm, m.actuator_gainprm),
        (solver.model.mujoco.actuator_biasprm, m.actuator_biasprm),
    ):
        for array in (source, live):
            flat = _flat(array)
            wp.launch(_zero_first_actuator, flat.size, inputs=[mask, flat.size // 2, flat])
    for array in (solver.model.mujoco.equality_constraint_enabled, solver.mjw_data.eq_active):
        flat = array.reshape((array.size,))
        wp.launch(_zero_bool, flat.size, inputs=[mask, flat.size // 2, flat])


def _derived(solver):
    m = solver.mjw_model
    result = {
        name: getattr(m, name).numpy().copy()
        for name in (
            "dof_invweight0",
            "body_invweight0",
            "tendon_invweight0",
            "tendon_length0",
            "actuator_acc0",
            "actuator_biasprm",
            "actuator_lengthrange",
            "eq_data",
            "jnt_solref",
            "cam_pos0",
            "cam_poscom0",
            "cam_mat0",
            "light_pos0",
            "light_poscom0",
            "light_dir0",
        )
    }
    result["meaninertia"] = m.stat.meaninertia.numpy().copy()
    return result


def _notify(solver):
    solver.notify_model_changed(
        ModelFlags.JOINT_DOF_PROPERTIES
        | ModelFlags.TENDON_PROPERTIES
        | ModelFlags.ACTUATOR_PROPERTIES
        | ModelFlags.CONSTRAINT_PROPERTIES
    )


@pytest.mark.parametrize("kinematic", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_release_constants_host_parity(device, kinematic):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    with wp.ScopedDevice(device):
        solver = _make_solver(device, kinematic)
        reference = _make_solver(device, kinematic)
        refresh = solver.prepare_model_constants_refresh()
        reference.prepare_model_constants_refresh()
        original = _derived(solver)
        original_rotor = solver.model.joint_armature.numpy().copy()
        assert np.all(solver.mjw_model.dof_armature.numpy()[:, 3:6] > 0)
        addresses = [values.ptr for values, _, _ in refresh.saved]
        mask = wp.array([True, False], dtype=wp.bool)
        _release(solver, mask)
        _release(reference, mask)
        refresh(mask)
        _notify(reference)
        for name, value in _derived(solver).items():
            np.testing.assert_array_equal(value, _derived(reference)[name], err_msg=name)
            np.testing.assert_array_equal(value[1], original[name][1], err_msg=f"untouched world: {name}")
        assert np.all(solver.mjw_model.dof_armature.numpy()[0, 3:6] == 0)
        assert not np.array_equal(original["body_invweight0"][0], solver.mjw_model.body_invweight0.numpy()[0])
        assert addresses == [values.ptr for values, _, _ in refresh.saved]
        # Ordinary reset notification restores authored rotors. The same prepared
        # workspace remains valid for the next independent release in world 1.
        solver.model.joint_armature.assign(original_rotor)
        reference.model.joint_armature.assign(original_rotor)
        _notify(solver)
        _notify(reference)
        mask.assign([False, True])
        _release(solver, mask)
        _release(reference, mask)
        refresh(mask)
        _notify(reference)
        for name, value in _derived(solver).items():
            np.testing.assert_array_equal(value, _derived(reference)[name], err_msg=f"reset: {name}")


def test_release_constants_cuda_replay():
    if not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    with wp.ScopedDevice("cuda:0"):
        solver = _make_solver("cuda:0", True)
        reference = _make_solver("cuda:0", True)
        refresh = solver.prepare_model_constants_refresh()
        reference.prepare_model_constants_refresh()
        mask = wp.array([False, False], dtype=wp.bool)
        # Warm kernels without releasing, then record. Recording must not change
        # constants or consume the first release that arrives only on replay.
        _release(solver, mask)
        refresh(mask)
        before = _derived(solver)
        with wp.ScopedCapture() as capture:
            _release(solver, mask)
            refresh(mask)
            solver._mujoco_warp.step(solver.mjw_model, solver.mjw_data)
        for name, value in _derived(solver).items():
            np.testing.assert_array_equal(value, before[name])
        mask.assign([True, False])
        _release(reference, mask)
        _notify(reference)
        reference._mujoco_warp.step(reference.mjw_model, reference.mjw_data)
        with mock.patch.object(solver, "notify_model_changed", side_effect=AssertionError("host notification")):
            wp.capture_launch(capture.graph)
            wp.synchronize()
        for name, value in _derived(solver).items():
            np.testing.assert_array_equal(value, _derived(reference)[name], err_msg=name)
        np.testing.assert_array_equal(solver.mjw_data.qpos.numpy(), reference.mjw_data.qpos.numpy())
        np.testing.assert_array_equal(solver.mjw_data.qvel.numpy(), reference.mjw_data.qvel.numpy())
        # A device-empty tick must not even invalidate an armed contact cache.
        mask.fill_(False)
        solver._last_contact_generation.fill_(17)
        wp.capture_launch(capture.graph)
        assert solver._last_contact_generation.numpy()[0] == 17
        reference._mujoco_warp.step(reference.mjw_model, reference.mjw_data)
        np.testing.assert_array_equal(solver.mjw_data.qpos.numpy(), reference.mjw_data.qpos.numpy())
        np.testing.assert_array_equal(solver.mjw_data.qvel.numpy(), reference.mjw_data.qvel.numpy())
        # Restore authored rotor through ordinary notification, then replay the
        # already recorded graph for an independent release in world 1.
        for item in (solver, reference):
            rotor = item.model.joint_armature.numpy()
            mapping = item.mjc_dof_to_newton_dof.numpy()
            rotor[mapping[:, 3:6]] = 0.0014435225
            item.model.joint_armature.assign(rotor)
            _notify(item)
        mask.assign([False, True])
        _release(reference, mask)
        _notify(reference)
        reference._mujoco_warp.step(reference.mjw_model, reference.mjw_data)
        wp.capture_launch(capture.graph)
        for name, value in _derived(solver).items():
            np.testing.assert_array_equal(value, _derived(reference)[name], err_msg=f"replay after reset: {name}")
        np.testing.assert_array_equal(solver.mjw_data.qpos.numpy(), reference.mjw_data.qpos.numpy())
        np.testing.assert_array_equal(solver.mjw_data.qvel.numpy(), reference.mjw_data.qvel.numpy())


def test_release_constants_cpu_without_ambient_device_scope():
    solver = _make_solver("cpu", False)
    refresh = solver.prepare_model_constants_refresh()
    before = _derived(solver)
    mask = wp.array([True, False], dtype=wp.bool, device="cpu")
    # Do not enter ScopedDevice: refresh must select its own solver device.
    refresh(mask)
    for name, value in _derived(solver).items():
        np.testing.assert_array_equal(value, before[name], err_msg=name)
