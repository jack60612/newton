# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Construction-time storage bounds for rigid MuJoCo-Warp scenes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco
    import mujoco_warp


def _structural_constraint_rows(model: mujoco.MjModel) -> int:
    """Count potential rows, including inactive equalities and unengaged limits."""
    import mujoco

    equality_widths = {
        int(mujoco.mjtEq.mjEQ_CONNECT): 3,
        int(mujoco.mjtEq.mjEQ_WELD): 6,
        int(mujoco.mjtEq.mjEQ_JOINT): 1,
        int(mujoco.mjtEq.mjEQ_TENDON): 1,
    }
    rows = sum(equality_widths[int(kind)] for kind in model.eq_type)
    # MuJoCo-Warp 3.12 rejects exactly-fitting CONNECT/WELD blocks.
    rows += int(any(equality_widths[int(kind)] > 1 for kind in model.eq_type))
    rows += int(np.count_nonzero(model.jnt_limited))
    rows += int(np.count_nonzero(model.tendon_limited))
    rows += int(np.count_nonzero(model.dof_frictionloss > 0))
    rows += int(np.count_nonzero(model.tendon_frictionloss > 0))
    return rows


def _contact_points(model: mujoco.MjModel, first: int, second: int) -> int:
    """Bound a rigid pair's manifold using MuJoCo-Warp's collision dispatch."""
    import mujoco
    from mujoco_warp._src.collision_driver import MJ_COLLISION_TABLE

    geom = mujoco.mjtGeom
    a, b = sorted((int(model.geom_type[first]), int(model.geom_type[second])))
    if geom.mjGEOM_SDF in (a, b):
        return int(model.opt.sdf_initpoints)
    if (a, b) not in MJ_COLLISION_TABLE:
        # Broadphase eligibility does not imply a narrowphase dispatch.
        return 0
    if geom.mjGEOM_HFIELD in (a, b):
        return int(mujoco.mjMAXCONPAIR)
    if a == geom.mjGEOM_PLANE:
        return {
            int(geom.mjGEOM_SPHERE): 1,
            int(geom.mjGEOM_CAPSULE): 2,
            int(geom.mjGEOM_ELLIPSOID): 1,
            int(geom.mjGEOM_CYLINDER): 4,
            int(geom.mjGEOM_BOX): 8,
            int(geom.mjGEOM_MESH): 4,
        }[b]
    nativeccd = not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_NATIVECCD
    multiccd = not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
    if a == b == geom.mjGEOM_BOX:
        return 4 if nativeccd else 8
    if a in (geom.mjGEOM_BOX, geom.mjGEOM_MESH) and b in (geom.mjGEOM_BOX, geom.mjGEOM_MESH) and multiccd:
        return 4
    if a == geom.mjGEOM_CAPSULE and b in (geom.mjGEOM_CAPSULE, geom.mjGEOM_BOX):
        return 2
    return 1


def _scene_constraint_capacity(model: mujoco.MjModel, device_model: mujoco_warp.Model) -> tuple[int, int]:
    """Return per-world (constraint rows, contacts) for finalized rigid topology.

    Uses the native filtered pair list, including explicit pairs and sensor-only
    pairs. This is a worst-case bound, not a prediction of simultaneous contacts.
    """
    import mujoco

    if model.nflex:
        raise ValueError("Topology-based capacity estimation only supports rigid MuJoCo models.")
    rows = _structural_constraint_rows(model)
    contacts = 0
    if not model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT:
        pairs = device_model.nxn_geom_pair_filtered.numpy()
        for (first, second), (pair, _) in zip(
            pairs, device_model.nxn_pairid_filtered.numpy(), strict=True
        ):
            points = _contact_points(model, int(first), int(second))
            if pair >= 0:
                dimension = int(model.pair_dim[pair])
            elif model.geom_priority[first] > model.geom_priority[second]:
                dimension = int(model.geom_condim[first])
            elif model.geom_priority[first] < model.geom_priority[second]:
                dimension = int(model.geom_condim[second])
            else:
                dimension = int(max(model.geom_condim[first], model.geom_condim[second]))
            width = dimension if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else max(1, 2 * (dimension - 1))
            contacts += points
            rows += points * width
        # Broadphase stores candidates in buffers capped by naconmax as well.
        # No-op dispatch pairs consume candidate slots, but no constraint rows.
        contacts = max(contacts, len(pairs))
    # Keep the implicit integrator's force-RHS path on pinned MuJoCo-Warp.
    return max(1, rows), contacts
