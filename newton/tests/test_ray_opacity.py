# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared material cutouts for model raycasts, camera depth and shadows."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorCamera
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestRayOpacity(unittest.TestCase):
    pass


def _rgba_mask():
    texture = np.full((8, 8, 4), 255, dtype=np.uint8)
    texture[:, :4, 3] = 0
    return texture


def _quad(z, *, cutout=False, transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
    return newton.Mesh(
        np.array([[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]], dtype=np.float32),
        np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        texture_transform=transform,
        opacity_texture=_rgba_mask() if cutout else None,
        opacity_threshold=0.5 if cutout else None,
        double_sided=True,
        compute_inertia=False,
    )


def _cloned_scene(device, *, transform=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
    prototype = newton.ModelBuilder()
    prototype.add_shape_mesh(-1, mesh=_quad(0.0, cutout=True, transform=transform), label="leaf")
    prototype.add_shape_mesh(-1, mesh=_quad(-1.0), color=(1.0, 1.0, 1.0), label="background")
    builder = newton.ModelBuilder()
    builder.add_world(prototype)
    builder.add_world(prototype)
    return builder.finalize(device=device)


def _query_arrays(device):
    origins = np.array([[-0.5, 0.0, 2.0], [0.5, 0.0, 2.0]] * 2, dtype=np.float32)
    directions = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (4, 1))
    return (
        wp.array(origins, dtype=wp.vec3f, device=device),
        wp.array(directions, dtype=wp.vec3f, device=device),
        wp.array([0, 0, 1, 1], dtype=wp.int32, device=device),
    )


def _camera_rays(device, *, from_below=False):
    z, dz = (-2.0, 1.0) if from_below else (2.0, -1.0)
    rays = np.zeros((1, 2, 1, 2, 3), dtype=np.float32)
    rays[0, :, 0, 0] = [[-0.5, 0.0, z], [0.5, 0.0, z]]
    rays[0, :, 0, 1] = [0.0, 0.0, dz]
    return wp.array(rays, dtype=wp.vec3f, device=device)


def _transforms(count, device):
    return wp.array(np.tile([0, 0, 0, 0, 0, 0, 1], (count, 1)), dtype=wp.transformf, device=device)


def test_raycast_camera_and_cloned_worlds(test: TestRayOpacity, device: str):
    for transform, expected in (
        (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), [3.0, 2.0]),
        (((1.0, 0.0, 0.5), (0.0, 1.0, 0.0)), [2.0, 3.0]),
    ):
        model = _cloned_scene(device, transform=transform)
        test.assertTrue(model._ray_opacity.has_cutouts)
        origins, directions, worlds = _query_arrays(device)
        distances = wp.empty(4, dtype=wp.float32, device=device)
        hit_shapes = wp.empty(4, dtype=wp.int32, device=device)
        truncated = wp.empty(4, dtype=wp.int32, device=device)
        camera = SensorCamera(model)
        camera_rays = _camera_rays(device)
        state = model.state()
        poses = _transforms(2, device)
        depth = camera.allocate_image_depth(2, 2, 1)
        shape = camera.allocate_image_shape_index(2, 2, 1)

        def cast_and_render():
            newton.intersect_ray(
                model,
                ray_origins=origins,
                ray_directions=directions,
                ray_worlds=worlds,
                out_dist=distances,
                out_shape_id=hit_shapes,
                out_truncated=truncated,
            )
            camera.update(state, poses, camera_rays, depth_image=depth, shape_index_image=shape)

        cast_and_render()
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                cast_and_render()
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(truncated.numpy(), [0, 0, 0, 0])
        np.testing.assert_allclose(distances.numpy(), expected * 2, atol=1.0e-5)
        np.testing.assert_allclose(depth.numpy().reshape(-1), expected * 2, atol=1.0e-5)
        np.testing.assert_array_equal(hit_shapes.numpy().astype(np.uint32), shape.numpy().reshape(-1))
        np.testing.assert_array_equal(model.shape_world.numpy()[hit_shapes.numpy()], [0, 0, 1, 1])


def test_alpha_cutout_shadow_matches_ray(test: TestRayOpacity, device: str):
    model = _cloned_scene(device)
    camera = SensorCamera(model)
    camera.create_default_light(enable_shadows=True, direction=wp.vec3f(0.0, 0.0, -1.0))
    hdr = camera.allocate_image_color_hdr(2, 2, 1)
    camera.update(
        model.state(),
        _transforms(2, device),
        _camera_rays(device, from_below=True),
        hdr_color_image=hdr,
        render_config=SensorCamera.RenderConfig(
            enable_shadows=True, enable_ambient_lighting=False, enable_backface_culling=False
        ),
    )
    values = hdr.numpy()[:, 0, :, 0]
    # Light arrives from above through the leaf, while the camera observes the
    # receiver from below.  The same left texel that the raycast passes must
    # receive the unshadowed directional light in both cloned worlds.
    np.testing.assert_allclose(values[:, 0], [1.0, 1.0], atol=1.0e-5)
    np.testing.assert_allclose(values[:, 1], [0.3, 0.3], atol=1.0e-5)


def test_rejected_triangle_does_not_hide_later_triangle_in_same_mesh(test: TestRayOpacity, device: str):
    # One mesh owns two separate quads.  The nearer left texel is transparent;
    # every point of the farther quad maps to the opaque half of the texture.
    vertices = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
        ],
        dtype=np.float32,
    )
    indices = np.array([0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7], dtype=np.int32)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.75, 0], [0.75, 0], [0.75, 1], [0.75, 1]], dtype=np.float32)
    mesh = newton.Mesh(
        vertices,
        indices,
        uvs=uv,
        opacity_texture=_rgba_mask(),
        opacity_threshold=0.5,
        double_sided=True,
        compute_inertia=False,
    )
    builder = newton.ModelBuilder()
    builder.begin_world()
    shape = builder.add_shape_mesh(-1, mesh=mesh)
    builder.end_world()
    model = builder.finalize(device=device)
    origins = wp.array([[-0.5, 0.0, 2.0], [0.5, 0.0, 2.0]], dtype=wp.vec3f, device=device)
    directions = wp.array([[0, 0, -1], [0, 0, -1]], dtype=wp.vec3f, device=device)
    worlds = wp.array([0, 0], dtype=wp.int32, device=device)
    distance = wp.empty(2, dtype=wp.float32, device=device)
    ids = wp.empty(2, dtype=wp.int32, device=device)
    truncated = wp.empty(2, dtype=wp.int32, device=device)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=distance,
        out_shape_id=ids,
        out_truncated=truncated,
    )
    np.testing.assert_allclose(distance.numpy(), [3.0, 2.0], atol=1.0e-5)
    np.testing.assert_array_equal(ids.numpy(), [shape, shape])
    np.testing.assert_array_equal(truncated.numpy(), [0, 0])


def test_double_sided_cutout_camera_and_raycast(test: TestRayOpacity, device: str):
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_shape_mesh(-1, mesh=_quad(0.0, cutout=True))
    builder.end_world()
    model = builder.finalize(device=device)
    origins = wp.array([[-0.5, 0.0, -2.0], [0.5, 0.0, -2.0]], dtype=wp.vec3f, device=device)
    directions = wp.array([[0, 0, 1], [0, 0, 1]], dtype=wp.vec3f, device=device)
    worlds = wp.array([0, 0], dtype=wp.int32, device=device)
    distances = wp.empty(2, dtype=wp.float32, device=device)
    truncated = wp.empty(2, dtype=wp.int32, device=device)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=distances,
        out_truncated=truncated,
    )
    camera = SensorCamera(model)
    depth = camera.allocate_image_depth(1, 2, 1)
    camera.update(model.state(), _transforms(1, device), _camera_rays(device, from_below=True), depth_image=depth)
    np.testing.assert_allclose(distances.numpy(), [-1.0, 2.0], atol=1.0e-5)
    np.testing.assert_allclose(depth.numpy().reshape(-1), [0.0, 2.0], atol=1.0e-5)
    np.testing.assert_array_equal(truncated.numpy(), [0, 0])


def test_scalar_threshold_keeps_equality(test: TestRayOpacity, device: str):
    for opacity, expected in ((0.49, -1.0), (0.5, 2.0)):
        mesh = _quad(0.0)
        mesh.opacity = opacity
        mesh.opacity_threshold = 0.5
        builder = newton.ModelBuilder()
        builder.begin_world()
        builder.add_shape_mesh(-1, mesh=mesh)
        builder.end_world()
        model = builder.finalize(device=device)
        origins = wp.array([[0.0, 0.0, 2.0]], dtype=wp.vec3f, device=device)
        directions = wp.array([[0.0, 0.0, -1.0]], dtype=wp.vec3f, device=device)
        worlds = wp.array([0], dtype=wp.int32, device=device)
        distances = wp.empty(1, dtype=wp.float32, device=device)
        truncated = wp.empty(1, dtype=wp.int32, device=device)
        newton.intersect_ray(
            model,
            ray_origins=origins,
            ray_directions=directions,
            ray_worlds=worlds,
            out_dist=distances,
            out_truncated=truncated,
        )
        np.testing.assert_allclose(distances.numpy(), [expected], atol=1.0e-5)
        np.testing.assert_array_equal(truncated.numpy(), [0])


def _layered_mesh(levels):
    vertices = []
    indices = []
    uvs = []
    for layer, (z, opaque) in enumerate(levels):
        vertices.extend([[-1, -1, z], [1, -1, z], [1, 1, z], [-1, 1, z]])
        indices.extend([4 * layer, 4 * layer + 1, 4 * layer + 2, 4 * layer, 4 * layer + 2, 4 * layer + 3])
        u = 0.75 if opaque else 0.25
        uvs.extend([[u, 0], [u, 0], [u, 1], [u, 1]])
    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        uvs=np.asarray(uvs, dtype=np.float32),
        opacity_texture=_rgba_mask(),
        opacity_threshold=0.5,
        double_sided=True,
        compute_inertia=False,
    )


def test_alpha_budget_fails_closed_and_reports_each_ray(test: TestRayOpacity, device: str):
    builder = newton.ModelBuilder()
    builder.begin_world()
    shape = builder.add_shape_mesh(-1, mesh=_layered_mesh([(0, False), (-0.1, False), (-0.2, True)]))
    builder.end_world()
    model = builder.finalize(device=device)
    origins = wp.array([[0.1, 0.0, 2.0]], dtype=wp.vec3f, device=device)
    directions = wp.array([[0, 0, -1]], dtype=wp.vec3f, device=device)
    worlds = wp.array([0], dtype=wp.int32, device=device)
    dist = wp.empty(1, dtype=wp.float32, device=device)
    ids = wp.empty(1, dtype=wp.int32, device=device)
    normal = wp.empty(1, dtype=wp.vec3f, device=device)
    truncated = wp.empty(1, dtype=wp.int32, device=device)
    with test.assertRaisesRegex(ValueError, "out_truncated"):
        newton.intersect_ray(model, ray_origins=origins, ray_directions=directions, ray_worlds=worlds)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=dist,
        out_shape_id=ids,
        out_normal=normal,
        out_truncated=truncated,
        max_alpha_rejections=1,
    )
    np.testing.assert_array_equal(truncated.numpy(), [1])
    np.testing.assert_array_equal(dist.numpy(), [-1.0])
    np.testing.assert_array_equal(ids.numpy(), [-1])
    np.testing.assert_array_equal(normal.numpy(), [[0.0, 0.0, 0.0]])
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=dist,
        out_shape_id=ids,
        out_normal=normal,
        out_truncated=truncated,
        max_alpha_rejections=8,
    )
    np.testing.assert_array_equal(truncated.numpy(), [0])
    np.testing.assert_array_equal(ids.numpy(), [shape])
    np.testing.assert_allclose(dist.numpy(), [2.2], atol=1.0e-5)


def test_alpha_continuation_resolves_known_small_gap(test: TestRayOpacity, device: str):
    # The gap is 0.1 mm (> 8 ULP at 100 m), well below a LiDAR's cm precision.
    gap = 0.0001
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_shape_mesh(-1, mesh=_layered_mesh([(0, False), (-gap, True)]))
    builder.end_world()
    model = builder.finalize(device=device)
    origins = wp.array([[0.1, 0.0, 20.0], [0.1, 0.0, 100.0]], dtype=wp.vec3f, device=device)
    directions = wp.array([[0, 0, -1], [0, 0, -1]], dtype=wp.vec3f, device=device)
    worlds = wp.array([0, 0], dtype=wp.int32, device=device)
    dist = wp.empty(2, dtype=wp.float32, device=device)
    truncated = wp.empty(2, dtype=wp.int32, device=device)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=dist,
        out_truncated=truncated,
    )
    np.testing.assert_array_equal(truncated.numpy(), [0, 0])
    np.testing.assert_allclose(dist.numpy(), [20.0 + gap, 100.0 + gap], atol=1.0e-5, rtol=0)


def test_disabled_ray_mask_returns_miss_without_truncation(test: TestRayOpacity, device: str):
    model = _cloned_scene(device)
    origins, directions, worlds = _query_arrays(device)
    enabled = wp.array([False, True, True, False], dtype=wp.bool, device=device)
    dist = wp.full(4, 99.0, dtype=wp.float32, device=device)
    ids = wp.full(4, 99, dtype=wp.int32, device=device)
    normal = wp.full(4, wp.vec3f(1.0, 1.0, 1.0), dtype=wp.vec3f, device=device)
    truncated = wp.full(4, 1, dtype=wp.int32, device=device)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        ray_enabled=enabled,
        out_dist=dist,
        out_shape_id=ids,
        out_normal=normal,
        out_truncated=truncated,
    )
    np.testing.assert_allclose(dist.numpy(), [-1.0, 2.0, 3.0, -1.0], atol=1.0e-5)
    np.testing.assert_array_equal(ids.numpy()[[0, 3]], [-1, -1])
    np.testing.assert_array_equal(normal.numpy()[[0, 3]], np.zeros((2, 3), dtype=np.float32))
    np.testing.assert_array_equal(truncated.numpy(), [0, 0, 0, 0])


devices = get_test_devices()
add_function_test(
    TestRayOpacity, "test_raycast_camera_and_cloned_worlds", test_raycast_camera_and_cloned_worlds, devices=devices
)
add_function_test(
    TestRayOpacity, "test_alpha_cutout_shadow_matches_ray", test_alpha_cutout_shadow_matches_ray, devices=devices
)
add_function_test(
    TestRayOpacity,
    "test_rejected_triangle_does_not_hide_later_triangle_in_same_mesh",
    test_rejected_triangle_does_not_hide_later_triangle_in_same_mesh,
    devices=devices,
)
add_function_test(
    TestRayOpacity,
    "test_double_sided_cutout_camera_and_raycast",
    test_double_sided_cutout_camera_and_raycast,
    devices=devices,
)
add_function_test(
    TestRayOpacity, "test_scalar_threshold_keeps_equality", test_scalar_threshold_keeps_equality, devices=devices
)
add_function_test(
    TestRayOpacity,
    "test_alpha_budget_fails_closed_and_reports_each_ray",
    test_alpha_budget_fails_closed_and_reports_each_ray,
    devices=devices,
)
add_function_test(
    TestRayOpacity,
    "test_alpha_continuation_resolves_known_small_gap",
    test_alpha_continuation_resolves_known_small_gap,
    devices=devices,
)
add_function_test(
    TestRayOpacity,
    "test_disabled_ray_mask_returns_miss_without_truncation",
    test_disabled_ray_mask_returns_miss_without_truncation,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
