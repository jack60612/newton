# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepared fixed-layout numeric-property refresh for captured simulation."""

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .solver_mujoco import SolverMuJoCo


@wp.kernel
def _any_world(mask: wp.array[wp.bool], condition: wp.array[int]):
    world = wp.tid()
    if mask[world]:
        wp.atomic_max(condition, 0, 1)


@wp.kernel
def _restore_unchanged(mask: wp.array[wp.bool], width: int, saved: wp.array[float], values: wp.array[float]):
    index = wp.tid()
    if not mask[index // width]:
        values[index] = saved[index]


# Every per-world model field written by ``set_const_0`` and by the numeric
# post-processing of an inertial/actuator notification.
_MODEL_FIELDS = (
    "tendon_length0",
    "eq_data",
    "dof_invweight0",
    "body_invweight0",
    "tendon_invweight0",
    "cam_pos0",
    "cam_poscom0",
    "cam_mat0",
    "light_pos0",
    "light_poscom0",
    "light_dir0",
    "actuator_acc0",
    "actuator_biasprm",
    "actuator_gainprm",
    "actuator_dynprm",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "actuator_actrange",
    "actuator_gear",
    "actuator_cranklength",
    "jnt_solref",
    "dof_armature",
    "tendon_solref_lim",
    "tendon_range",
)


class ModelConstantsRefresh:
    """Callable workspace returned by ``prepare_model_constants_refresh``.

    Not reentrant. Recreate after model buffers or topology are replaced. The mask
    is borrowed, never cleared, and must have one bool per solver world.
    """

    def __init__(self, solver: "SolverMuJoCo"):
        self.solver = solver
        m, d = solver.mjw_model, solver.mjw_data
        self.nworld = d.nworld
        self.condition = wp.zeros(1, dtype=int, device=solver.device)
        self.saved = []
        # Expand broadcast storage before capture so no world shares derived state.
        targets = [(m, field) for field in _MODEL_FIELDS]
        if solver._actuator_dampratio is not None:
            # Published dampratio placeholders follow the actuator_biasprm row count.
            targets.append((solver, "_actuator_dampratio"))
        for owner, name in targets:
            array = getattr(owner, name)
            if array.shape[0] != self.nworld:
                values = np.repeat(array.numpy(), self.nworld, axis=0)
                array = wp.array(values, dtype=array.dtype, device=array.device)
                setattr(owner, name, array)
            count = array.size * wp.types.type_size_in_bytes(array.dtype) // 4
            flat = wp.array(ptr=array.ptr, dtype=float, shape=(count,), device=array.device)
            self.saved.append((flat, wp.empty_like(flat), count // self.nworld))
        self.workspace = solver._mujoco_warp.Const0Workspace(m, d)

    def __call__(self, world_mask: wp.array[wp.bool]) -> None:
        """Refresh selected worlds after live and source numeric-property writes.

        Supports eager execution and CUDA graph recording/replay. No device
        allocation occurs here. During CUDA capture a conditional node skips
        all work when the device mask is empty. Eager CUDA execution computes all
        worlds and commits only selected worlds, avoiding condition readback.
        Topology, references, limits, body flags and buffer identities must remain
        unchanged since preparation.
        """
        if world_mask.shape != (self.nworld,) or world_mask.dtype != wp.bool:
            raise ValueError("world_mask must be a bool array with one entry per solver world")
        with wp.ScopedDevice(self.solver.device), self.solver._scoped_mujoco_warp_execution():
            self.condition.zero_()
            wp.launch(_any_world, self.nworld, inputs=[world_mask, self.condition], device=self.solver.device)
            if self.solver.device.is_cuda and not self.solver.device.is_capturing:
                self._refresh(world_mask)
            else:
                wp.capture_if(self.condition, self._refresh, world_mask=world_mask)

    def _refresh(self, world_mask: wp.array[wp.bool]) -> None:
        s = self.solver
        for values, saved, _ in self.saved:
            wp.copy(saved, values)
        # Mirror the inertial and actuator notifications: publish the source
        # armature and reupload raw dampratio before resolving it against the
        # changed reflected mass.
        wp.copy(s._notify_joint_armature, s.model.joint_armature)
        s._update_dof_armature()
        s._update_actuator_properties()
        s._set_const_0(workspace=self.workspace)
        s._update_solref_from_invweight0()
        s._update_tendon_limit_gains()
        # Release does not change reference poses: reuse the relative transforms
        # prepared by ordinary notification, stored in place for the solver lifetime.
        s._notify_connect_constraints_changed(False, False)
        for values, saved, width in self.saved:
            if values.size:
                wp.launch(_restore_unchanged, values.size, inputs=[world_mask, width, saved, values], device=s.device)
        s._invalidate_contact_fast_path()
        s._wake_sleeping_worlds(world_mask)
