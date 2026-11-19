# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material cutouts for queries against a model's shared shape BVH.

The table belongs to the model, not to a sensor.  Its entries retain the
texture and UV allocations referenced by the device structs.  Variant rows
select their current entry on the device, so selection is graph-capturable.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ..utils import load_texture, normalize_texture
from .types import GeoType, Mesh

if TYPE_CHECKING:
    from ..sim.model import Model


@wp.struct
class RayOpacity:
    uvs: wp.array[wp.vec2f]
    texture: wp.Texture2D
    threshold: wp.float32
    has_texture: wp.int32
    double_sided: wp.int32


@wp.struct
class RayOpacityView:
    shapes: wp.array[wp.int32]
    variants: wp.array[wp.int32]
    shape_variant: wp.array[wp.int32]
    shape_opacity: wp.array[wp.float32]
    materials: wp.array[RayOpacity]


class RayOpacityTable:
    """Device representation of the cutout materials in a model's shape sources."""

    def __init__(self, model: Model):
        self._resources = []
        materials = []
        source_ids: dict[int, int] = {}
        material_ids: dict[tuple, int] = {}
        uv_arrays: dict[tuple, wp.array] = {}
        textures: dict[tuple, wp.Texture2D] = {}
        array_ids: dict[int, tuple] = {}
        array_candidates: dict[tuple, list[tuple[np.ndarray, tuple]]] = {}
        array_representatives = 0

        def texture_identity(texture):
            nonlocal array_representatives
            if isinstance(texture, str):
                return ("path", texture)
            identity = id(texture)
            if identity in array_ids:
                return array_ids[identity]
            digest = hashlib.blake2b(memoryview(texture).cast("B"), digest_size=16).digest()
            fingerprint = (texture.shape, texture.dtype.str, digest)
            candidates = array_candidates.setdefault(fingerprint, [])
            for previous, key in candidates:
                # The digest only narrows candidates: a collision must never alias distinct alpha masks.
                if np.array_equal(previous.view(np.uint8), texture.view(np.uint8)):
                    array_ids[identity] = key
                    return key
            key = ("array", array_representatives)
            array_representatives += 1
            candidates.append((texture, key))
            array_ids[identity] = key
            return key

        def material_id(source):
            if not isinstance(source, Mesh) or not source.opacity_threshold:
                return -1
            source_key = id(source)
            if source_key in source_ids:
                return source_ids[source_key]
            opacity_texture = source.opacity_texture
            texture_key = None
            uv_key = None
            if opacity_texture is not None:
                if source.uvs is None or len(source.uvs) != len(source.vertices):
                    raise ValueError("An opacity texture requires one UV coordinate per mesh vertex")
                texture_key = texture_identity(opacity_texture)
                uv_key = (source.uvs.shape, source.uvs.tobytes(), source.texture_transform)
            key = (source.opacity_threshold, source.double_sided, texture_key, uv_key)
            if key in material_ids:
                source_ids[source_key] = material_ids[key]
                return material_ids[key]
            item = RayOpacity()
            item.threshold = source.opacity_threshold
            item.double_sided = int(source.double_sided)
            if opacity_texture is not None:
                assert texture_key is not None and uv_key is not None
                if texture_key not in textures:
                    pixels = load_texture(opacity_texture)
                    if pixels is None:
                        raise ValueError(f"Failed to load opacity texture: {opacity_texture}")
                    pixels = normalize_texture(pixels, require_channels=True)
                    if pixels.shape[2] != 4:
                        raise ValueError("An opacity texture must have an alpha channel")
                    textures[texture_key] = wp.Texture2D(
                        pixels,
                        filter_mode=wp.TextureFilterMode.LINEAR,
                        address_mode=wp.TextureAddressMode.WRAP,
                        normalized_coords=True,
                        dtype=wp.uint8,
                        num_channels=4,
                        device=model.device,
                    )
                if uv_key not in uv_arrays:
                    transform = np.asarray(source.texture_transform, dtype=np.float32)
                    uv = np.asarray(source.uvs, dtype=np.float32) @ transform[:, :2].T + transform[:, 2]
                    uv_arrays[uv_key] = wp.array(uv, dtype=wp.vec2f, device=model.device)
                    self._resources.append(uv_arrays[uv_key])
                item.uvs = uv_arrays[uv_key]
                item.texture = textures[texture_key]
                item.has_texture = 1
            material_ids[key] = len(materials)
            source_ids[source_key] = material_ids[key]
            materials.append(item)
            return material_ids[key]

        # Do not allocate device tables or load any textures for models without
        # authored cutouts.  The ordinary raycast and camera kernels then keep
        # their existing compile-time-specialized path.
        shape_ids = np.fromiter((material_id(source) for source in model.shape_source), dtype=np.int32)
        variant_ids = np.fromiter((material_id(source) for source in model._variant_shape_source), dtype=np.int32)
        self.has_cutouts = bool(materials)
        if self.has_cutouts:
            shape_types = model.shape_type.numpy()
            unsupported = (shape_ids >= 0) & (shape_types != int(GeoType.MESH))
            if model._has_variant_rows:
                unsupported &= ~model._shape_variant_row.numpy()
            if np.any(unsupported):
                raise ValueError("Opacity textures are supported only on triangle mesh shapes")
            if len(variant_ids):
                variant_types = model._variant_shape_geometry["shape_type"].numpy()
                if np.any((variant_ids >= 0) & (variant_types != int(GeoType.MESH))):
                    raise ValueError("Opacity textures are supported only on triangle mesh variants")
            self.shapes = wp.array(shape_ids, dtype=wp.int32, device=model.device)
            self.variants = wp.array(variant_ids, dtype=wp.int32, device=model.device)
            self.materials = wp.array(materials, dtype=RayOpacity, device=model.device)
            self._resources.extend(textures.values())
        else:
            self.shapes = None
            self.variants = None
            self.materials = None
        self.view = RayOpacityView()
        self.view.shapes = self.shapes
        self.view.variants = self.variants
        self.view.shape_variant = model._shape_variant_shape
        self.view.shape_opacity = model.shape_opacity
        self.view.materials = self.materials


@wp.func
def material_index(
    shape: wp.int32,
    shape_material: wp.array[wp.int32],
    variant_material: wp.array[wp.int32],
    shape_variant: wp.array[wp.int32],
) -> wp.int32:
    result = shape_material[shape]
    if shape_variant.shape[0] > 0:
        variant = shape_variant[shape]
        if variant >= 0:
            result = variant_material[variant]
    return result


@wp.func
def accepts_hit(
    material: RayOpacity,
    scalar_opacity: wp.float32,
    mesh_id: wp.uint64,
    face: wp.int32,
    bary_u: wp.float32,
    bary_v: wp.float32,
) -> wp.bool:
    opacity = scalar_opacity
    if material.has_texture != 0:
        i0 = wp.mesh_get_index(mesh_id, face * 3)
        i1 = wp.mesh_get_index(mesh_id, face * 3 + 1)
        i2 = wp.mesh_get_index(mesh_id, face * 3 + 2)
        uv = material.uvs[i0] * bary_u + material.uvs[i1] * bary_v + material.uvs[i2] * (1.0 - bary_u - bary_v)
        opacity = wp.texture_sample(material.texture, wp.vec2f(uv[0], 1.0 - uv[1]), dtype=wp.vec4f)[3]
    return opacity >= material.threshold


@wp.func
def _next_float(value: wp.float32, positive: wp.bool) -> wp.float32:
    """Adjacent float in the requested direction (the value is finite)."""
    bits = wp.cast(value, wp.uint32)
    if value == 0.0:
        bits = wp.uint32(1) if positive else wp.uint32(0x80000001)
    elif (value > 0.0) == positive:
        bits += wp.uint32(1)
    else:
        bits -= wp.uint32(1)
    return wp.cast(bits, wp.float32)


def make_mesh_cutout_intersection(compute_normal: bool):
    """Build a closest opaque triangle query using Warp's native mesh raycast.

    A bounded number of rejected triangles may be traversed.  Since Warp's
    native mesh query has no minimum-distance parameter, each continuation
    advances the ray parameter by one representable float.  Exactly coplanar
    triangles and surfaces within its rounding interval cannot be distinguished.
    Exhaustion is reported to the caller rather than treated as a miss.
    """

    @wp.func
    def intersect(
        origin: wp.vec3f,
        direction: wp.vec3f,
        scale: wp.vec3f,
        mesh_id: wp.uint64,
        cull_backfaces: wp.bool,
        max_t: wp.float32,
        material: RayOpacity,
        scalar_opacity: wp.float32,
        max_rejections: wp.int32,
    ) -> tuple[wp.float32, wp.vec3f, wp.float32, wp.float32, wp.int32, wp.int32, wp.bool]:
        if mesh_id == wp.uint64(0):
            return -1.0, wp.vec3f(0.0), 0.0, 0.0, -1, 0, False
        rejected = wp.int32(0)
        offset = wp.float32(0.0)
        while offset < max_t:
            result = wp.mesh_query_ray(mesh_id, origin + direction * offset, direction, max_t - offset)
            if not result.result:
                break
            hit_t = offset + result.t
            if (not cull_backfaces or wp.dot(direction, result.normal) < 0.0) and accepts_hit(
                material, scalar_opacity, mesh_id, result.face, result.u, result.v
            ):
                normal = wp.vec3f(0.0)
                if wp.static(compute_normal):
                    inv_scale = wp.vec3f(
                        1.0 / wp.where(scale[0] != 0.0, scale[0], 1.0e-6),
                        1.0 / wp.where(scale[1] != 0.0, scale[1], 1.0e-6),
                        1.0 / wp.where(scale[2] != 0.0, scale[2], 1.0e-6),
                    )
                    normal = wp.normalize(wp.cw_mul(result.normal, inv_scale))
                return hit_t, normal, result.u, result.v, result.face, rejected, False
            if rejected >= max_rejections:
                return -1.0, wp.vec3f(0.0), 0.0, 0.0, -1, rejected, True
            rejected += 1
            offset = _next_float(hit_t, True)
        return -1.0, wp.vec3f(0.0), 0.0, 0.0, -1, rejected, False

    return intersect


ray_intersect_mesh_cutout = make_mesh_cutout_intersection(True)
ray_intersect_mesh_cutout_no_normal = make_mesh_cutout_intersection(False)
