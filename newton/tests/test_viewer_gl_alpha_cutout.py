# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Headless GL coverage for authored alpha-cutout mesh materials."""

import ctypes
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.viewer.gl.opengl import MeshInstancerGL, RendererGL
from newton.tests.test_viewer_get_frame import _make_headless_viewer_gl_or_skip


class TestViewerGLAlphaCutout(unittest.TestCase):
    def setUp(self):
        self.viewer = _make_headless_viewer_gl_or_skip(self, width=128, height=128)
        renderer = self.viewer.renderer
        renderer.msaa_samples = 0
        renderer.draw_sky = False
        renderer.draw_shadows = True
        renderer.draw_edges = False
        renderer.sky_upper = (0.0, 0.0, 0.0)
        renderer.sky_lower = (0.0, 0.0, 0.0)
        renderer.ambient_sky = (1.0, 1.0, 1.0)
        renderer.ambient_ground = (1.0, 1.0, 1.0)
        renderer.diffuse_scale = 0.0
        renderer.specular_scale = 0.0
        renderer.spotlight_enabled = False
        renderer._env_path = None
        # The allocated shadow attachment remains large; only this viewport is read.
        renderer._shadow_width = 256
        renderer._shadow_height = 256
        self._camera(-3.0)

        self.vertices = np.array(
            [(-0.8, 0.0, -0.8), (0.8, 0.0, -0.8), (0.8, 0.0, 0.8), (-0.8, 0.0, 0.8)],
            dtype=np.float32,
        )
        self.indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int32)
        self.normals = np.tile((0.0, -1.0, 0.0), (4, 1)).astype(np.float32)
        self.uvs = np.array([(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)], dtype=np.float32)
        self.texture = np.full((2, 2, 4), 255, dtype=np.uint8)
        self.texture[0, 0, 3] = 0

    def tearDown(self):
        self.viewer.close()

    def _camera(self, y):
        self.viewer.set_camera(wp.vec3(0.0, y, 0.0), pitch=0.0, yaw=0.0)
        self.viewer.camera.look_at((0.0, 0.0, 0.0))

    def _source(self, **kwargs):
        return newton.Mesh(
            self.vertices,
            self.indices,
            normals=self.normals,
            uvs=kwargs.pop("uvs", self.uvs),
            compute_inertia=False,
            **kwargs,
        )

    def _log_card(self, source, opacity=1.0):
        viewer = self.viewer
        viewer.log_geo("/cutout/card", newton.GeoType.MESH, (1.0, 1.0, 1.0), 0.0, True, geo_src=source, hidden=True)
        device = viewer.device
        viewer.log_instances(
            "/cutout/instance",
            "/cutout/card",
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3, device=device),
            wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3, device=device),
            wp.array([(0.5, 0.0, 0.0, float(source.texture is not None))], dtype=wp.vec4, device=device),
            opacities=wp.array([opacity], dtype=wp.float32, device=device),
        )

    def _depth(self, fbo, width, height):
        gl = RendererGL.gl
        depth = np.empty((height, width), dtype=np.float32)
        self.viewer.renderer._make_current()
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fbo)
        gl.glReadPixels(0, 0, width, height, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, depth.ctypes.data_as(ctypes.c_void_p))
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        return depth

    def _frame(self):
        self.viewer.begin_frame(0.0)
        self.viewer.end_frame()
        color = self.viewer.get_frame().numpy().copy()
        renderer = self.viewer.renderer
        depth = self._depth(renderer._frame_fbo, 128, 128)
        shadow = self._depth(renderer._shadow_fbo, 256, 256)
        return color, depth, shadow

    def test_color_depth_and_shadow_use_the_same_cutout(self):
        self._log_card(self._source(texture=self.texture, double_sided=True))
        opaque_color, opaque_depth, opaque_shadow = self._frame()
        opaque = opaque_depth < 1.0
        self.assertGreater(np.count_nonzero(opaque), 1000)
        self.assertGreater(np.count_nonzero(opaque_shadow < 1.0), 20)

        # A color texture's alpha alone has never implied transparency.
        self._log_card(self._source(texture=self.texture, opacity_threshold=0.0, double_sided=True))
        zero_color, zero_depth, zero_shadow = self._frame()
        np.testing.assert_array_equal(zero_color, opaque_color)
        np.testing.assert_array_equal(zero_depth, opaque_depth)
        np.testing.assert_array_equal(zero_shadow, opaque_shadow)

        self._log_card(
            self._source(texture=self.texture, opacity_texture=self.texture, opacity_threshold=0.5, double_sided=True)
        )
        cut_color, cut_depth, cut_shadow = self._frame()
        kept = cut_depth < 1.0
        holes = opaque & ~kept
        self.assertGreater(np.count_nonzero(kept), 200)
        self.assertGreater(np.count_nonzero(holes), 200)
        self.assertTrue(np.all(cut_color[127 - np.where(holes)[0], np.where(holes)[1]] == 0))
        self.assertTrue(np.all(kept <= opaque))
        shadow_opaque = opaque_shadow < 1.0
        shadow_cut = cut_shadow < 1.0
        self.assertGreater(np.count_nonzero(shadow_cut), 10)
        self.assertGreater(np.count_nonzero(shadow_opaque & ~shadow_cut), 10)
        self.assertTrue(np.all(shadow_cut <= shadow_opaque))
        prototype = self.viewer.objects["/cutout/card"]
        self.assertIs(prototype.opacity_texture_id, prototype.texture_id)

    def test_bilinear_repeat_and_scalar_threshold(self):
        def visible(uv, *, threshold=0.5, opacity_texture=True, opacity=1.0):
            source = self._source(
                uvs=np.tile(uv, (4, 1)),
                texture=self.texture,
                opacity_texture=self.texture if opacity_texture else None,
                opacity_threshold=threshold,
                double_sided=True,
            )
            self._log_card(source, opacity=opacity)
            color, depth, shadow = self._frame()
            return color[64, 64].max() > 0, depth[64, 64] < 1.0, np.any(shadow < 1.0)

        self.assertEqual(visible((0.25, 0.75)), (False, False, False))
        self.assertEqual(visible((1.25, -0.25)), (False, False, False))
        self.assertEqual(visible((0.75, 0.75)), (True, True, True))
        # The boundary lies at u=0.5: base-level bilinear alpha is 0.48/0.52.
        self.assertEqual(visible((0.49, 0.75)), (False, False, False))
        self.assertEqual(visible((0.51, 0.75)), (True, True, True))
        # No connected alpha means compare the per-instance scalar, with equality kept.
        self.assertEqual(visible((0.25, 0.75), opacity_texture=False, opacity=0.5), (True, True, True))
        self.assertEqual(visible((0.25, 0.75), opacity_texture=False, opacity=0.49), (False, False, False))
        self.assertEqual(
            visible((0.25, 0.75), threshold=0.9999, opacity_texture=False, opacity=0.9995),
            (False, False, False),
        )

    def test_direct_mesh_uses_authored_scalar_opacity(self):
        for opacity, expected in ((0.49, False), (0.5, True)):
            with self.subTest(opacity=opacity):
                source = self._source(opacity=opacity, opacity_threshold=0.5, double_sided=True)
                self.viewer.log_geo("/cutout/direct", newton.GeoType.MESH, (1.0, 1.0, 1.0), 0.0, True, geo_src=source)
                _, depth, shadow = self._frame()
                self.assertEqual(depth[64, 64] < 1.0, expected)
                self.assertEqual(np.any(shadow < 1.0), expected)

    def test_edge_overlay_does_not_draw_through_cutout(self):
        renderer = self.viewer.renderer
        renderer._edge_color = (1.0, 0.0, 0.0, 1.0)
        source = self._source(uvs=np.tile((0.25, 0.75), (4, 1)), texture=self.texture, double_sided=True)
        self._log_card(source)
        renderer.draw_edges = False
        plain, _, _ = self._frame()
        renderer.draw_edges = True
        outlined, _, _ = self._frame()
        self.assertTrue(np.any(plain != outlined), "the edge overlay did not render")

        source.opacity_texture = self.texture
        source.opacity_threshold = 0.5
        self._log_card(source)
        cut_color, cut_depth, cut_shadow = self._frame()
        self.assertFalse(np.any(cut_color))
        self.assertTrue(np.all(cut_depth == 1.0))
        self.assertTrue(np.all(cut_shadow == 1.0))

    def test_hole_reveals_geometry_behind_the_card(self):
        behind = self.vertices.copy()
        behind[:, 1] = 0.5
        device = self.viewer.device
        self.viewer.log_mesh(
            "/cutout/behind",
            wp.array(behind, dtype=wp.vec3, device=device),
            wp.array(self.indices, dtype=wp.int32, device=device),
            wp.array(self.normals, dtype=wp.vec3, device=device),
            color=(0.0, 1.0, 0.0),
            backface_culling=False,
        )
        background_color, background_depth, _ = self._frame()
        self.assertLess(background_depth[64, 64], 1.0)

        source = self._source(
            uvs=np.tile((0.25, 0.75), (4, 1)),
            texture=self.texture,
            opacity_texture=self.texture,
            opacity_threshold=0.5,
            double_sided=True,
        )
        self._log_card(source)
        through_color, through_depth, _ = self._frame()
        np.testing.assert_array_equal(through_color, background_color)
        np.testing.assert_array_equal(through_depth, background_depth)

        source = self._source(
            uvs=np.tile((0.75, 0.75), (4, 1)),
            texture=self.texture,
            opacity_texture=self.texture,
            opacity_threshold=0.5,
            double_sided=True,
        )
        self._log_card(source)
        front_color, front_depth, _ = self._frame()
        self.assertLess(front_depth[64, 64], background_depth[64, 64])
        self.assertFalse(np.array_equal(front_color[64, 64], background_color[64, 64]))

    def test_model_scalar_cutout_updates_without_blending(self):
        source = self._source(opacity=0.5, opacity_threshold=0.5, double_sided=True)
        builder = newton.ModelBuilder()
        builder.add_shape_mesh(-1, mesh=source)
        model = builder.finalize(device="cpu")
        self.viewer.set_model(model)

        for opacity, expected in ((0.5, True), (0.49, False), (0.5, True)):
            model.shape_opacity.assign(np.array([opacity], dtype=np.float32))
            self.viewer.begin_frame(0.0)
            self.viewer.log_state(model.state())
            self.viewer.end_frame()
            depth = self._depth(self.viewer.renderer._frame_fbo, 128, 128)
            shadow = self._depth(self.viewer.renderer._shadow_fbo, 256, 256)
            self.assertEqual(depth[64, 64] < 1.0, expected)
            self.assertEqual(np.any(shadow < 1.0), expected)

    def test_distinct_opacity_texture_and_uv_transform(self):
        diffuse = np.full_like(self.texture, 255)
        diffuse[:, :, 3] = 0
        solid_alpha = np.full_like(self.texture, 255)
        constant_uvs = np.tile((0.25, 0.75), (4, 1))
        self._log_card(
            self._source(
                uvs=constant_uvs,
                texture=diffuse,
                opacity_texture=solid_alpha,
                opacity_threshold=0.5,
                double_sided=True,
            )
        )
        _, depth, _ = self._frame()
        self.assertLess(depth[64, 64], 1.0)

        transform = ((0.7, 0.0, 0.2), (0.0, 0.8, -0.1))
        self._log_card(
            self._source(
                opacity_texture=self.texture, opacity_threshold=0.5, texture_transform=transform, double_sided=True
            )
        )
        _, transformed_depth, transformed_shadow = self._frame()
        expected_uvs = self.uvs * (0.7, 0.8) + (0.2, -0.1)
        self._log_card(
            self._source(uvs=expected_uvs, opacity_texture=self.texture, opacity_threshold=0.5, double_sided=True)
        )
        _, expected_depth, expected_shadow = self._frame()
        self.assertGreater(np.count_nonzero(expected_depth < 1.0), 100)
        self.assertGreater(np.count_nonzero(expected_depth == 1.0), 100)
        self.assertLess(np.mean((transformed_depth < 1.0) != (expected_depth < 1.0)), 0.002)
        self.assertLess(np.mean((transformed_shadow < 1.0) != (expected_shadow < 1.0)), 0.002)

    def test_model_batches_keep_distinct_cutout_materials(self):
        builder = newton.ModelBuilder()
        for x, threshold in ((-0.85, 0.25), (0.85, 0.75)):
            source = self._source(
                uvs=np.tile((0.5, 0.75), (4, 1)),
                texture=self.texture,
                opacity_texture=self.texture,
                opacity_threshold=threshold,
                double_sided=True,
            )
            builder.add_shape_mesh(-1, xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()), mesh=source)
        model = builder.finalize(device="cpu")
        self.viewer.set_model(model)
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(model.state())
        self.viewer.end_frame()
        depth = self._depth(self.viewer.renderer._frame_fbo, 128, 128)
        self.assertEqual(np.count_nonzero(depth[64, [33, 95]] < 1.0), 1)
        thresholds = {
            obj.mesh.opacity_threshold
            for obj in self.viewer.objects.values()
            if isinstance(obj, MeshInstancerGL) and not obj.hidden
        }
        self.assertEqual(thresholds, {0.25, 0.75})

    def test_model_instances_honor_sidedness_and_mirrored_prototypes(self):
        source = self._source(
            uvs=np.tile((0.75, 0.75), (4, 1)),
            texture=self.texture,
            opacity_texture=self.texture,
            opacity_threshold=0.5,
            double_sided=True,
        )
        for scale in ((1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)):
            with self.subTest(scale=scale):
                builder = newton.ModelBuilder()
                builder.add_shape_mesh(-1, mesh=source, scale=scale)
                model = builder.finalize(device="cpu")
                self.viewer.set_model(model)
                self._camera(3.0)  # View the back of the authored single-sided fan.
                self.viewer.begin_frame(0.0)
                self.viewer.log_state(model.state())
                self.viewer.end_frame()
                depth = self._depth(self.viewer.renderer._frame_fbo, 128, 128)
                self.assertLess(depth[64, 64], 1.0)
                instances = [
                    obj for obj in self.viewer.objects.values() if isinstance(obj, MeshInstancerGL) and not obj.hidden
                ]
                self.assertTrue(instances)
                self.assertTrue(
                    any(obj.mesh.opacity_threshold == 0.5 and not obj.mesh.backface_culling for obj in instances)
                )

    def test_missing_opacity_texture_does_not_silently_become_opaque(self):
        source = self._source(opacity_texture="/does/not/exist/alpha.png", opacity_threshold=0.5, double_sided=True)
        with (
            self.assertWarnsRegex(Warning, "Failed to load texture image"),
            self.assertRaisesRegex(ValueError, "Opacity texture"),
        ):
            self._log_card(source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
