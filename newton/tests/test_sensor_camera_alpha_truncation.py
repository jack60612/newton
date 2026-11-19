# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""SensorCamera alpha-rejection budgets and per-pixel failure diagnostics."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorCamera
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestSensorCameraAlphaTruncation(unittest.TestCase):
    pass


def _quad(z: float, *, cutout: bool):
    # Both cutouts cover only the left half of the opaque receiver. This lets
    # different samples of the same pixel hit the same receiver with different
    # primary and shadow traversal outcomes.
    x0, x1 = (-1.0, 0.0) if cutout else (-1.0, 1.0)
    vertices = np.array([(x0, -1.0, z), (x1, -1.0, z), (x1, 1.0, z), (x0, 1.0, z)], dtype=np.float32)
    texture = np.full((2, 2, 4), 255, dtype=np.uint8)
    texture[:, :, 3] = 0
    return newton.Mesh(
        vertices,
        np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        uvs=np.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=np.float32),
        opacity_texture=texture if cutout else None,
        opacity_threshold=0.5 if cutout else None,
        double_sided=True,
        compute_inertia=False,
    )


def _scene(device, *, cutouts=True):
    builder = newton.ModelBuilder()
    if cutouts:
        builder.add_shape_mesh(-1, mesh=_quad(0.0, cutout=True))
        builder.add_shape_mesh(-1, mesh=_quad(0.25, cutout=True))
    builder.add_shape_mesh(-1, mesh=_quad(-1.0, cutout=False), color=(1.0, 1.0, 1.0))
    return builder.finalize(device=device)


def _rays(device, mode, *, shadow):
    samples = 1 if mode == SensorCamera.AntiAliasing.NONE else 2
    rays = np.zeros((1, 1, samples, 2, 3), dtype=np.float32)
    # The safe subsample comes first; both see the same receiver from below
    # in the shadow test, exercising MSAA's per-surface shading cache.
    xs = [-0.5] if samples == 1 else [0.5, -0.5]
    z = -2.0 if shadow else 1.0
    dz = 1.0 if shadow else -1.0
    for i, x in enumerate(xs):
        rays[0, 0, i, 0] = (x, 0.25, z)
        rays[0, 0, i, 1] = (0.0, 0.0, dz)
    return wp.array(rays, dtype=wp.vec3f, device=device)


def _pose(device):
    return wp.array([wp.transform_identity()], dtype=wp.transformf, device=device)


def _outputs(camera):
    return {
        "color_image": camera.allocate_image_color(1, 1, 1),
        "depth_image": camera.allocate_image_depth(1, 1, 1),
        "forward_depth_image": camera.allocate_image_depth_forward(1, 1, 1),
        "shape_index_image": camera.allocate_image_shape_index(1, 1, 1),
        "normal_image": camera.allocate_image_normal(1, 1, 1),
        "albedo_image": camera.allocate_image_albedo(1, 1, 1),
        "hdr_color_image": camera.allocate_image_color_hdr(1, 1, 1),
    }


def _assert_cleared(test, outputs):
    test.assertEqual(int(outputs["color_image"].numpy()[0, 0, 0]), 0x10203040)
    test.assertEqual(int(outputs["albedo_image"].numpy()[0, 0, 0]), 0x50302010)
    test.assertEqual(float(outputs["depth_image"].numpy()[0, 0, 0]), -3.0)
    test.assertEqual(float(outputs["forward_depth_image"].numpy()[0, 0, 0]), -3.0)
    test.assertEqual(int(outputs["shape_index_image"].numpy()[0, 0, 0]), 12345)
    np.testing.assert_allclose(outputs["normal_image"].numpy()[0, 0, 0], (0.25, 0.5, 0.75))
    np.testing.assert_array_equal(outputs["hdr_color_image"].numpy()[0, 0, 0], (0.0, 0.0, 0.0))


def test_primary_budget_and_all_output_clear(test: TestSensorCameraAlphaTruncation, device: str):
    model = _scene(device)
    camera = SensorCamera(model)
    state = model.state()
    transforms = _pose(device)
    clear = SensorCamera.ClearData(
        clear_color=0x10203040,
        clear_depth=-3.0,
        clear_shape_index=12345,
        clear_normal=(0.25, 0.5, 0.75),
        clear_albedo=0x50302010,
    )
    outputs = _outputs(camera)
    status = wp.empty((1, 1, 1), dtype=wp.int32, device=device)
    for mode in SensorCamera.AntiAliasing:
        config = SensorCamera.RenderConfig(anti_aliasing=mode, max_alpha_rejections=1)
        rays = _rays(device, mode, shadow=False)
        camera.update(
            state, transforms, rays, **outputs, alpha_truncated_image=status, clear_data=clear, render_config=config
        )
        test.assertIs(camera.alpha_truncated_image, status)
        test.assertEqual(int(status.numpy()[0, 0, 0]), 1)
        _assert_cleared(test, outputs)

        # Exercise the independent no-normal, depth-only traversal as well.
        camera.update(
            state,
            transforms,
            rays,
            depth_image=outputs["depth_image"],
            alpha_truncated_image=status,
            clear_data=clear,
            render_config=config,
        )
        test.assertEqual(int(status.numpy()[0, 0, 0]), 1)
        test.assertEqual(float(outputs["depth_image"].numpy()[0, 0, 0]), -3.0)

        # The same global budget is shared across both candidate meshes. Two
        # rejections are sufficient and the opaque receiver must become visible.
        config.max_alpha_rejections = 2
        camera.update(
            state,
            transforms,
            rays,
            depth_image=outputs["depth_image"],
            alpha_truncated_image=status,
            clear_data=clear,
            render_config=config,
        )
        test.assertEqual(int(status.numpy()[0, 0, 0]), 0)
        np.testing.assert_allclose(outputs["depth_image"].numpy(), 2.0, atol=1.0e-5)


def test_shadow_budget_clears_all_aa_modes(test: TestSensorCameraAlphaTruncation, device: str):
    model = _scene(device)
    camera = SensorCamera(model)
    camera.create_default_light(enable_shadows=True, direction=wp.vec3f(0.0, 0.0, -1.0))
    state = model.state()
    transforms = _pose(device)
    clear = SensorCamera.ClearData(
        clear_color=0x10203040,
        clear_depth=-3.0,
        clear_shape_index=12345,
        clear_normal=(0.25, 0.5, 0.75),
        clear_albedo=0x50302010,
    )
    outputs = _outputs(camera)
    status = wp.empty((1, 1, 1), dtype=wp.int32, device=device)
    for mode in SensorCamera.AntiAliasing:
        config = SensorCamera.RenderConfig(
            anti_aliasing=mode,
            max_alpha_rejections=1,
            enable_shadows=True,
            enable_ambient_lighting=False,
            enable_backface_culling=False,
        )
        rays = _rays(device, mode, shadow=True)
        camera.update(
            state, transforms, rays, **outputs, alpha_truncated_image=status, clear_data=clear, render_config=config
        )
        test.assertEqual(int(status.numpy()[0, 0, 0]), 1)
        _assert_cleared(test, outputs)

        config.max_alpha_rejections = 2
        camera.update(
            state, transforms, rays, **outputs, alpha_truncated_image=status, clear_data=clear, render_config=config
        )
        test.assertEqual(int(status.numpy()[0, 0, 0]), 0)
        test.assertGreater(float(outputs["hdr_color_image"].numpy()[0, 0, 0, 0]), 0.0)


def test_owned_status_preallocated_capture_and_disabled_views(test: TestSensorCameraAlphaTruncation, device: str):
    model = _scene(device)
    camera = SensorCamera(model)
    state = model.state()
    transforms = _pose(device)
    rays = _rays(device, SensorCamera.AntiAliasing.NONE, shadow=False)
    depth = camera.allocate_image_depth(1, 1, 1)
    config = SensorCamera.RenderConfig(max_alpha_rejections=1)
    test.assertIsNone(camera.alpha_truncated_image)
    camera.update(state, transforms, rays, depth_image=depth, render_config=config)
    owned = camera.alpha_truncated_image
    test.assertEqual(int(owned.numpy()[0, 0, 0]), 1)
    camera.update(state, transforms, rays, depth_image=depth, render_config=config)
    test.assertIs(camera.alpha_truncated_image, owned)

    status = wp.empty((1, 1, 1), dtype=wp.int32, device=device)

    def render():
        camera.update(state, transforms, rays, depth_image=depth, alpha_truncated_image=status, render_config=config)

    render()
    if wp.get_device(device).is_cuda:
        with wp.ScopedCapture(device=device) as capture:
            render()
        wp.capture_launch(capture.graph)
    test.assertIs(camera.alpha_truncated_image, status)
    test.assertEqual(int(status.numpy()[0, 0, 0]), 1)

    disabled = wp.array([int(SensorCamera.WorldRenderFlag.DISABLE_PRESERVE)], dtype=wp.int32, device=device)
    depth.fill_(42.0)
    status.fill_(1)
    camera.update(
        state,
        transforms,
        rays,
        depth_image=depth,
        alpha_truncated_image=status,
        world_indices=disabled,
        render_config=config,
    )
    test.assertEqual(int(status.numpy()[0, 0, 0]), 0)
    test.assertEqual(float(depth.numpy()[0, 0, 0]), 42.0)


def test_status_is_per_pixel(test: TestSensorCameraAlphaTruncation, device: str):
    model = _scene(device)
    camera = SensorCamera(model)
    rays = np.zeros((1, 2, 1, 2, 3), dtype=np.float32)
    rays[0, :, 0, 0] = [(0.5, 0.25, 1.0), (-0.5, 0.25, 1.0)]
    rays[0, :, 0, 1] = (0.0, 0.0, -1.0)
    depth = camera.allocate_image_depth(1, 2, 1)
    camera.update(
        model.state(),
        _pose(device),
        wp.array(rays, dtype=wp.vec3f, device=device),
        depth_image=depth,
        clear_data=SensorCamera.ClearData(clear_depth=-3.0),
        render_config=SensorCamera.RenderConfig(max_alpha_rejections=1),
    )
    np.testing.assert_array_equal(camera.alpha_truncated_image.numpy().reshape(-1), [0, 1])
    np.testing.assert_allclose(depth.numpy().reshape(-1), [2.0, -3.0], atol=1.0e-5)


def test_opaque_and_invalid_diagnostics(test: TestSensorCameraAlphaTruncation, device: str):
    model = _scene(device, cutouts=False)
    camera = SensorCamera(model)
    state = model.state()
    transforms = _pose(device)
    rays = _rays(device, SensorCamera.AntiAliasing.NONE, shadow=False)
    depth = camera.allocate_image_depth(1, 1, 1)
    config = SensorCamera.RenderConfig(max_alpha_rejections=0)
    camera.update(state, transforms, rays, depth_image=depth, render_config=config)
    test.assertEqual(int(camera.alpha_truncated_image.numpy()[0, 0, 0]), 0)
    np.testing.assert_allclose(depth.numpy(), 2.0, atol=1.0e-5)

    for invalid in (-1, 1.5, True, 2**31):
        with test.subTest(invalid=invalid), test.assertRaisesRegex(ValueError, "max_alpha_rejections"):
            camera.update(
                state,
                transforms,
                rays,
                depth_image=depth,
                render_config=SensorCamera.RenderConfig(max_alpha_rejections=invalid),
            )
    with test.assertRaisesRegex(ValueError, "alpha_truncated_image"):
        camera.update(
            state,
            transforms,
            rays,
            depth_image=depth,
            alpha_truncated_image=wp.empty((1, 1, 1), dtype=wp.float32, device=device),
        )
    with test.assertRaisesRegex(ValueError, "alpha_truncated_image"):
        camera.update(
            state,
            transforms,
            rays,
            depth_image=depth,
            alpha_truncated_image=wp.empty((1, 2, 1), dtype=wp.int32, device=device),
        )


devices = get_test_devices()
add_function_test(
    TestSensorCameraAlphaTruncation,
    "test_primary_budget_and_all_output_clear",
    test_primary_budget_and_all_output_clear,
    devices=devices,
)
add_function_test(
    TestSensorCameraAlphaTruncation,
    "test_shadow_budget_clears_all_aa_modes",
    test_shadow_budget_clears_all_aa_modes,
    devices=devices,
)
add_function_test(
    TestSensorCameraAlphaTruncation,
    "test_owned_status_preallocated_capture_and_disabled_views",
    test_owned_status_preallocated_capture_and_disabled_views,
    devices=devices,
)
add_function_test(
    TestSensorCameraAlphaTruncation, "test_status_is_per_pixel", test_status_is_per_pixel, devices=devices
)
add_function_test(
    TestSensorCameraAlphaTruncation,
    "test_opaque_and_invalid_diagnostics",
    test_opaque_and_invalid_diagnostics,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
