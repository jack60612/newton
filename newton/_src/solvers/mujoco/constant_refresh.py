# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepared fixed-layout numeric-property refresh for captured simulation."""

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...sim import BodyFlags

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


class ModelConstantsRefresh:
    """Callable workspace returned by ``prepare_model_constants_refresh``.

    Not reentrant. Recreate after model buffers or topology are replaced. The mask
    is borrowed, never cleared, and must have one bool per solver world.
    """

    def __init__(self, solver: "SolverMuJoCo"):
        self.solver = solver
        m, d = solver.mjw_model, solver.mjw_data
        self.nworld = d.nworld
        self.has_kinematic_bodies = bool(np.any((solver.model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0))
        self.condition = wp.zeros(1, dtype=int, device=solver.device)
        self.saved = []
        # Every field written by set_const_0, plus the notification's numeric
        # postprocessing. Expand before capture so no world shares derived state.
        fields = (
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
            "actuator_lengthrange",
            "jnt_solref",
            "dof_armature",
        )
        for owner, name in [(m, field) for field in fields] + [(m.stat, "meaninertia")]:
            array = getattr(owner, name)
            if array.shape[0] != self.nworld:
                values = np.repeat(array.numpy(), self.nworld, axis=0)
                array = wp.array(values, dtype=array.dtype, device=array.device)
                setattr(owner, name, array)
            count = array.size * wp.types.type_size_in_bytes(array.dtype) // 4
            flat = wp.array(ptr=array.ptr, dtype=float, shape=(count,), device=array.device)
            self.saved.append((flat, wp.empty_like(flat), count // self.nworld))
        self.workspace = solver._mujoco_warp.Const0Workspace(m, d)
        self.actuator_biasprm = wp.empty_like(m.actuator_biasprm)
        self.physical_meaninertia = wp.empty_like(m.stat.meaninertia)

    def __call__(self, world_mask: wp.array[wp.bool]) -> None:
        """Refresh selected worlds after live and source numeric-property writes.

        Supports eager execution and CUDA graph recording/replay. No host reads or
        device allocation occur here. During CUDA capture a conditional node skips
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
        m, d = s.mjw_model, s.mjw_data
        for values, saved, _ in self.saved:
            wp.copy(saved, values)
        # Host ACTUATOR_PROPERTIES reuploads raw dampratio before resolving it
        # against the changed reflected mass. Reuse its mapping, including for
        # retained actuators; a previously resolved negative bias is not a ratio.
        s._update_actuator_properties()
        s._mujoco_warp.set_length_range(m, d)
        s._set_const_0_with_physical_meaninertia(self)
        s._launch_solref_from_invweight0()
        # Release does not change reference poses: reuse the reference transforms
        # prepared by ordinary notification, rather than allocate/recompute FK.
        if s.has_connect_constraints:
            s._update_connect_constraint_anchors(
                s.model, m, s.mjc_eq_to_newton_eq, s.connect_constraint_q_rel, s.connect_constraint_t_rel
            )
        if s.has_jnt_connect_constraints:
            s._update_jnt_connect_constraint_anchors(
                s.model,
                m,
                s.mjc_eq_to_newton_jnt,
                s.jnt_eq_anchor1,
                s.jnt_connect_constraint_q_rel,
                s.jnt_connect_constraint_t_rel,
            )
        for values, saved, width in self.saved:
            if values.size:
                wp.launch(_restore_unchanged, values.size, inputs=[world_mask, width, saved, values], device=s.device)
        s._invalidate_contact_fast_path()
        s._wake_sleeping_worlds(world_mask)
