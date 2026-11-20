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


def _check_raycast_camera_and_cloned_worlds(test, device, transform, expected):
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


def test_raycast_camera_and_cloned_worlds(test: TestRayOpacity, device: str):
    for transform, expected in (
        (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), [3.0, 2.0]),
        (((1.0, 0.0, 0.5), (0.0, 1.0, 0.0)), [2.0, 3.0]),
    ):
        _check_raycast_camera_and_cloned_worlds(test, device, transform, expected)


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
    rejections = wp.full(1, 99, dtype=wp.int32, device=device)
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
        out_alpha_rejections=rejections,
        max_alpha_rejections=1,
    )
    np.testing.assert_array_equal(truncated.numpy(), [1])
    np.testing.assert_array_equal(rejections.numpy(), [1])
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
        out_alpha_rejections=rejections,
        max_alpha_rejections=2,
    )
    np.testing.assert_array_equal(truncated.numpy(), [0])
    np.testing.assert_array_equal(rejections.numpy(), [2])
    np.testing.assert_array_equal(ids.numpy(), [shape])
    np.testing.assert_allclose(dist.numpy(), [2.2], atol=1.0e-5)

    # A zero budget reports truncation at the first transparent candidate
    # without counting a rejection beyond the bound.
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=dist,
        out_shape_id=ids,
        out_normal=normal,
        out_truncated=truncated,
        out_alpha_rejections=rejections,
        max_alpha_rejections=0,
    )
    np.testing.assert_array_equal(truncated.numpy(), [1])
    np.testing.assert_array_equal(rejections.numpy(), [0])
    np.testing.assert_array_equal(dist.numpy(), [-1.0])
    np.testing.assert_array_equal(ids.numpy(), [-1])
    np.testing.assert_array_equal(normal.numpy(), [[0.0, 0.0, 0.0]])


def test_alpha_rejection_counts_per_ray_and_graph_capture(test: TestRayOpacity, device: str):
    builder = newton.ModelBuilder()
    builder.begin_world()
    shape = builder.add_shape_mesh(-1, mesh=_layered_mesh([(0, False), (-0.1, False), (-0.2, False), (-0.3, True)]))
    builder.end_world()
    model = builder.finalize(device=device)

    # The rays see three, two, one, and zero transparent layers before an
    # opaque hit; then an empty miss, a miss after one rejection, and a
    # disabled ray which would otherwise see all three layers.
    origins = wp.array(
        [
            [0.1, 0, 2],
            [0.1, 0, -0.05],
            [0.1, 0, -0.15],
            [0.1, 0, -0.25],
            [2, 0, 2],
            [0.1, 0, -0.05],
            [0.1, 0, 2],
        ],
        dtype=wp.vec3f,
        device=device,
    )
    directions = wp.array([[0, 0, -1]] * 5 + [[0, 0, 1], [0, 0, -1]], dtype=wp.vec3f, device=device)
    worlds = wp.array([0] * 7, dtype=wp.int32, device=device)
    enabled = wp.array([True] * 6 + [False], dtype=wp.bool, device=device)
    dist = wp.full(7, 99.0, dtype=wp.float32, device=device)
    ids = wp.full(7, 99, dtype=wp.int32, device=device)
    normal = wp.full(7, wp.vec3f(1.0, 1.0, 1.0), dtype=wp.vec3f, device=device)
    truncated = wp.full(7, 99, dtype=wp.int32, device=device)
    rejections = wp.full(7, 99, dtype=wp.int32, device=device)

    def cast(budget=8):
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
            out_alpha_rejections=rejections,
            max_alpha_rejections=budget,
        )

    def reset():
        dist.fill_(99.0)
        ids.fill_(99)
        normal.fill_(wp.vec3f(1.0, 1.0, 1.0))
        truncated.fill_(99)
        rejections.fill_(99)

    def check():
        np.testing.assert_array_equal(rejections.numpy(), [3, 2, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(truncated.numpy(), [0] * 7)
        np.testing.assert_array_equal(ids.numpy(), [shape, shape, shape, shape, -1, -1, -1])
        np.testing.assert_allclose(dist.numpy(), [2.3, 0.25, 0.15, 0.05, -1, -1, -1], atol=1.0e-5)
        np.testing.assert_array_equal(normal.numpy()[[4, 5, 6]], np.zeros((3, 3), dtype=np.float32))

    cast()
    check()
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            cast()
        reset()
        wp.capture_launch(capture.graph)
        check()
        original_enabled = wp.array([True] * 6 + [False], dtype=wp.bool, device=device)
        enabled.fill_(True)
        reset()
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(rejections.numpy(), [3, 2, 1, 0, 0, 1, 3])
        np.testing.assert_array_equal(truncated.numpy(), [0] * 7)
        test.assertEqual(int(ids.numpy()[6]), shape)
        np.testing.assert_allclose(dist.numpy()[6], 2.3, atol=1.0e-5)
        wp.copy(enabled, original_enabled)
        reset()
        wp.capture_launch(capture.graph)
        check()

    # Omitting the optional output must not alter any existing result.
    dist_without = wp.empty(7, dtype=wp.float32, device=device)
    ids_without = wp.empty(7, dtype=wp.int32, device=device)
    normal_without = wp.empty(7, dtype=wp.vec3f, device=device)
    truncated_without = wp.empty(7, dtype=wp.int32, device=device)

    def check_omitted(budget):
        newton.intersect_ray(
            model,
            ray_origins=origins,
            ray_directions=directions,
            ray_worlds=worlds,
            ray_enabled=enabled,
            out_dist=dist_without,
            out_shape_id=ids_without,
            out_normal=normal_without,
            out_truncated=truncated_without,
            max_alpha_rejections=budget,
        )
        np.testing.assert_array_equal(dist_without.numpy(), dist.numpy())
        np.testing.assert_array_equal(ids_without.numpy(), ids.numpy())
        np.testing.assert_array_equal(normal_without.numpy(), normal.numpy())
        np.testing.assert_array_equal(truncated_without.numpy(), truncated.numpy())

    check_omitted(8)

    # At a lower bound, only the first ray exhausts its budget. The count on
    # that ray includes the rejection that reached the bound.
    def check_limited():
        np.testing.assert_array_equal(rejections.numpy(), [2, 2, 1, 0, 0, 1, 0])
        np.testing.assert_array_equal(truncated.numpy(), [1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(ids.numpy(), [-1, shape, shape, shape, -1, -1, -1])
        np.testing.assert_allclose(dist.numpy(), [-1, 0.25, 0.15, 0.05, -1, -1, -1], atol=1.0e-5)
        np.testing.assert_array_equal(normal.numpy()[[0, 4, 5, 6]], np.zeros((4, 3), dtype=np.float32))

    cast(2)
    check_limited()
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            cast(2)
        reset()
        wp.capture_launch(capture.graph)
        check_limited()
    check_omitted(2)


def test_alpha_continuation_rejects_rounded_self_hit_once(test: TestRayOpacity, device: str):
    # Advancing the ray parameter by just one ULP rounded the origin back onto
    # the transparent z=-0.2 triangle and exhausted this one-rejection budget.
    builder = newton.ModelBuilder()
    builder.begin_world()
    shape = builder.add_shape_mesh(-1, mesh=_layered_mesh([(0, False), (-0.1, False), (-0.2, False), (-0.3, True)]))
    builder.end_world()
    model = builder.finalize(device=device)
    origins = wp.array([[0.1, 0, -0.15]], dtype=wp.vec3f, device=device)
    directions = wp.array([[0, 0, -1]], dtype=wp.vec3f, device=device)
    worlds = wp.array([0], dtype=wp.int32, device=device)
    dist = wp.empty(1, dtype=wp.float32, device=device)
    ids = wp.empty(1, dtype=wp.int32, device=device)
    truncated = wp.empty(1, dtype=wp.int32, device=device)
    rejections = wp.empty(1, dtype=wp.int32, device=device)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=dist,
        out_shape_id=ids,
        out_truncated=truncated,
        out_alpha_rejections=rejections,
        max_alpha_rejections=1,
    )
    np.testing.assert_array_equal(rejections.numpy(), [1])
    np.testing.assert_array_equal(truncated.numpy(), [0])
    np.testing.assert_allclose(dist.numpy(), [0.15], atol=1.0e-5)
    np.testing.assert_array_equal(ids.numpy(), [shape])

    # The shared traversal is also used by the camera's depth-only and
    # full-output paths.
    camera = SensorCamera(model)
    state = model.state()
    poses = _transforms(1, device)
    camera_rays = np.zeros((1, 1, 1, 2, 3), dtype=np.float32)
    camera_rays[0, 0, 0, 0] = (0.1, 0.0, -0.15)
    camera_rays[0, 0, 0, 1] = (0.0, 0.0, -1.0)
    camera_rays = wp.array(camera_rays, dtype=wp.vec3f, device=device)
    depth = camera.allocate_image_depth(1, 1, 1)
    shape_image = camera.allocate_image_shape_index(1, 1, 1)
    status = wp.empty((1, 1, 1), dtype=wp.int32, device=device)
    config = SensorCamera.RenderConfig(max_alpha_rejections=1)
    for full_output in (False, True):
        with test.subTest(full_output=full_output):
            outputs = {"depth_image": depth}
            if full_output:
                outputs["shape_index_image"] = shape_image
            camera.update(
                state,
                poses,
                camera_rays,
                **outputs,
                alpha_truncated_image=status,
                render_config=config,
            )
            np.testing.assert_array_equal(status.numpy(), [[[0]]])
            np.testing.assert_allclose(depth.numpy(), [[[0.15]]], atol=1.0e-5)
            if full_output:
                test.assertEqual(int(shape_image.numpy()[0, 0, 0]), shape)


def test_alpha_same_face_retry_budget_fails_closed_for_raycast_and_camera(test: TestRayOpacity, device: str):
    # At a grazing angle the position-space advance changes x but rounds z
    # back to the rejected plane, even after four doublings.
    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_shape_mesh(-1, mesh=_layered_mesh([(-0.2, False)]))
    builder.end_world()
    model = builder.finalize(device=device)
    origin = (-0.5245203375816345, -0.18875321745872498, -0.1999979019165039)
    direction = (1.0, 0.0, -2.5971164632210275e-6)
    origins = wp.array([origin], dtype=wp.vec3f, device=device)
    directions = wp.array([direction], dtype=wp.vec3f, device=device)
    worlds = wp.array([0], dtype=wp.int32, device=device)
    dist = wp.empty(1, dtype=wp.float32, device=device)
    ids = wp.empty(1, dtype=wp.int32, device=device)
    truncated = wp.empty(1, dtype=wp.int32, device=device)
    rejections = wp.empty(1, dtype=wp.int32, device=device)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=dist,
        out_shape_id=ids,
        out_truncated=truncated,
        out_alpha_rejections=rejections,
        max_alpha_rejections=8,
    )
    np.testing.assert_array_equal(rejections.numpy(), [1])
    np.testing.assert_array_equal(truncated.numpy(), [1])
    np.testing.assert_array_equal(dist.numpy(), [-1.0])
    np.testing.assert_array_equal(ids.numpy(), [-1])

    camera = SensorCamera(model)
    camera_rays = np.zeros((1, 1, 1, 2, 3), dtype=np.float32)
    camera_rays[0, 0, 0, 0] = origin
    camera_rays[0, 0, 0, 1] = direction
    camera_rays = wp.array(camera_rays, dtype=wp.vec3f, device=device)
    depth = camera.allocate_image_depth(1, 1, 1)
    shape_image = camera.allocate_image_shape_index(1, 1, 1)
    status = wp.empty((1, 1, 1), dtype=wp.int32, device=device)
    for full_output in (False, True):
        with test.subTest(full_output=full_output):
            outputs = {"depth_image": depth}
            if full_output:
                outputs["shape_index_image"] = shape_image
            camera.update(
                model.state(),
                _transforms(1, device),
                camera_rays,
                **outputs,
                alpha_truncated_image=status,
                render_config=SensorCamera.RenderConfig(max_alpha_rejections=8),
            )
            np.testing.assert_array_equal(status.numpy(), [[[1]]])


def test_alpha_continuation_uses_world_distance_with_scaled_mesh(test: TestRayOpacity, device: str):
    rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), wp.pi / 2)
    xform = wp.transform(wp.vec3(3, -2, 4), rotation)
    for scale_z in (0.25, 4.0):
        with test.subTest(scale_z=scale_z):
            builder = newton.ModelBuilder()
            builder.begin_world()
            shape = builder.add_shape_mesh(
                -1,
                mesh=_layered_mesh([(0, False), (-0.1, False), (-0.2, False), (-0.3, True)]),
                xform=xform,
                scale=(2.0, 0.5, scale_z),
            )
            builder.end_world()
            model = builder.finalize(device=device)
            # After the 90-degree rotation, local -z is world -x. The local
            # direction magnitudes are 4 and 0.25 for the two mesh scales.
            origins = wp.array(
                [[3 - 0.15 * scale_z, -2, 3.8], [3 + 2 * scale_z, -2, 3.8]],
                dtype=wp.vec3f,
                device=device,
            )
            directions = wp.array([[-1, 0, 0], [-1, 0, 0]], dtype=wp.vec3f, device=device)
            worlds = wp.array([0, 0], dtype=wp.int32, device=device)
            dist = wp.empty(2, dtype=wp.float32, device=device)
            ids = wp.empty(2, dtype=wp.int32, device=device)
            truncated = wp.empty(2, dtype=wp.int32, device=device)
            rejections = wp.empty(2, dtype=wp.int32, device=device)
            for budget, counts, exhausted, distances, shapes in (
                (3, [1, 3], [0, 0], [0.15 * scale_z, 2.3 * scale_z], [shape, shape]),
                (1, [1, 1], [0, 1], [0.15 * scale_z, -1.0], [shape, -1]),
            ):
                with test.subTest(budget=budget):
                    newton.intersect_ray(
                        model,
                        ray_origins=origins,
                        ray_directions=directions,
                        ray_worlds=worlds,
                        out_dist=dist,
                        out_shape_id=ids,
                        out_truncated=truncated,
                        out_alpha_rejections=rejections,
                        max_alpha_rejections=budget,
                    )
                    np.testing.assert_array_equal(rejections.numpy(), counts)
                    np.testing.assert_array_equal(truncated.numpy(), exhausted)
                    np.testing.assert_allclose(dist.numpy(), distances, atol=1.0e-5, rtol=0)
                    np.testing.assert_array_equal(ids.numpy(), shapes)


def test_alpha_rejection_counts_across_shapes_and_worlds(test: TestRayOpacity, device: str):
    builder = newton.ModelBuilder()
    global_shape = builder.add_shape_mesh(-1, mesh=_layered_mesh([(-0.2, False), (-0.3, True)]))
    builder.begin_world()
    builder.add_shape_mesh(-1, mesh=_layered_mesh([(0, False)]))
    builder.add_shape_mesh(-1, mesh=_layered_mesh([(-0.1, False)]))
    builder.end_world()
    model = builder.finalize(device=device)
    origins = wp.array([[0.1, 0, 2]], dtype=wp.vec3f, device=device)
    directions = wp.array([[0, 0, -1]], dtype=wp.vec3f, device=device)
    worlds = wp.array([0], dtype=wp.int32, device=device)
    dist = wp.empty(1, dtype=wp.float32, device=device)
    ids = wp.empty(1, dtype=wp.int32, device=device)
    truncated = wp.empty(1, dtype=wp.int32, device=device)
    rejections = wp.empty(1, dtype=wp.int32, device=device)

    for budget, expected_count, expected_truncated, expected_dist, expected_shape in (
        (2, 2, 1, -1.0, -1),
        (3, 3, 0, 2.3, global_shape),
    ):
        with test.subTest(budget=budget):
            newton.intersect_ray(
                model,
                ray_origins=origins,
                ray_directions=directions,
                ray_worlds=worlds,
                out_dist=dist,
                out_shape_id=ids,
                out_truncated=truncated,
                out_alpha_rejections=rejections,
                max_alpha_rejections=budget,
            )
            np.testing.assert_array_equal(rejections.numpy(), [expected_count])
            np.testing.assert_array_equal(truncated.numpy(), [expected_truncated])
            np.testing.assert_allclose(dist.numpy(), [expected_dist], atol=1.0e-5)
            np.testing.assert_array_equal(ids.numpy(), [expected_shape])


def test_alpha_continuation_resolves_known_small_gap(test: TestRayOpacity, device: str):
    # At 100 m, the rounded advance is about 38.1 µm, below this 100 µm gap.
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
    rejections = wp.empty(2, dtype=wp.int32, device=device)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=dist,
        out_truncated=truncated,
        out_alpha_rejections=rejections,
    )
    np.testing.assert_array_equal(truncated.numpy(), [0, 0])
    np.testing.assert_array_equal(rejections.numpy(), [1, 1])
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


def test_cutout_materials_share_resources_without_aliasing(test: TestRayOpacity, device: str):
    base = _quad(0.0, cutout=True)
    mask = _rgba_mask()
    mask[:, :4, 3] = 100
    base.opacity_texture = mask
    copies = [base.copy() for _ in range(12)]
    different_uv = base.copy()
    different_uv.uvs[:, 0] += 0.5
    different_transform = base.copy()
    different_transform.texture_transform = ((1.0, 0.0, 0.5), (0.0, 1.0, 0.0))
    different_mask = base.copy()
    different_mask.opacity_texture[:, :4, 3] = 255
    different_threshold = base.copy()
    different_threshold.opacity_threshold = 0.25
    different_sidedness = base.copy()
    different_sidedness.double_sided = False
    scalar_low = base.copy()
    scalar_low.opacity_texture = None
    scalar_low.opacity = 0.49
    scalar_equal = scalar_low.copy()
    scalar_equal.opacity = 0.5
    variants = [
        different_uv,
        different_transform,
        different_mask,
        different_threshold,
        different_sidedness,
        scalar_low,
        scalar_equal,
    ]

    # Distinct source objects and texture arrays model USD imports, then clone
    # them into the 256-world configuration. Device resources scale with the
    # distinct alpha/UV inputs, not with leaves or cloned worlds.
    prototype = newton.ModelBuilder()
    for source in copies + variants:
        prototype.add_shape_mesh(-1, mesh=source)
    builder = newton.ModelBuilder()
    builder.replicate(prototype, copy_count=256)
    model = builder.finalize(device=device)
    table = model._ray_opacity
    ids = table.shapes.numpy()
    test.assertEqual(len(ids), 19 * 256)
    test.assertEqual(len(np.unique(ids)), 7)
    test.assertEqual(np.count_nonzero(ids == ids[0]), 12 * 256)
    test.assertEqual(len(table.materials), 7)
    test.assertEqual(len(table._resources), 5)  # three UV buffers and two textures

    # The same alpha data may be shared even when the source meshes differ,
    # but changing UVs, transform, mask, threshold or per-shape scalar opacity
    # must still change the actual ray result.
    semantic = newton.ModelBuilder()
    cases = [copies[0], copies[1], *variants]
    for source in cases:
        semantic.begin_world()
        semantic.add_shape_mesh(-1, mesh=source)
        semantic.add_shape_mesh(-1, mesh=_quad(-1.0))
        semantic.end_world()
    model = semantic.finalize(device=device)
    origins = wp.array([[-0.5, 0.0, 2.0]] * len(cases), dtype=wp.vec3f, device=device)
    directions = wp.array([[0.0, 0.0, -1.0]] * len(cases), dtype=wp.vec3f, device=device)
    worlds = wp.array(np.arange(len(cases)), dtype=wp.int32, device=device)
    distances = wp.empty(len(cases), dtype=wp.float32, device=device)
    truncated = wp.empty(len(cases), dtype=wp.int32, device=device)
    newton.intersect_ray(
        model,
        ray_origins=origins,
        ray_directions=directions,
        ray_worlds=worlds,
        out_dist=distances,
        out_truncated=truncated,
    )
    np.testing.assert_allclose(distances.numpy(), [3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0], atol=1.0e-5)
    np.testing.assert_array_equal(truncated.numpy(), np.zeros(len(cases), dtype=np.int32))


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
    "test_alpha_rejection_counts_per_ray_and_graph_capture",
    test_alpha_rejection_counts_per_ray_and_graph_capture,
    devices=devices,
)
add_function_test(
    TestRayOpacity,
    "test_alpha_continuation_rejects_rounded_self_hit_once",
    test_alpha_continuation_rejects_rounded_self_hit_once,
    devices=devices,
)
add_function_test(
    TestRayOpacity,
    "test_alpha_same_face_retry_budget_fails_closed_for_raycast_and_camera",
    test_alpha_same_face_retry_budget_fails_closed_for_raycast_and_camera,
    devices=devices,
)
add_function_test(
    TestRayOpacity,
    "test_alpha_continuation_uses_world_distance_with_scaled_mesh",
    test_alpha_continuation_uses_world_distance_with_scaled_mesh,
    devices=devices,
)
add_function_test(
    TestRayOpacity,
    "test_alpha_rejection_counts_across_shapes_and_worlds",
    test_alpha_rejection_counts_across_shapes_and_worlds,
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
add_function_test(
    TestRayOpacity,
    "test_cutout_materials_share_resources_without_aliasing",
    test_cutout_materials_share_resources_without_aliasing,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
