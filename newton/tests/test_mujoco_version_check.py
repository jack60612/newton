# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import types
import unittest
import warnings
from unittest import mock

import newton
from newton._src.solvers.mujoco import solver_mujoco
from newton.exceptions import NewtonWarning

_MOCK_REQUIREMENTS = (
    "mujoco~=3.8.0 ; extra == 'sim'",
    "mujoco-warp~=3.8.0,>=3.8.0.3 ; extra == 'sim'",
)
_MOCK_METADATA = "\n".join(f"Requires-Dist: {requirement}" for requirement in _MOCK_REQUIREMENTS)


def _mujoco_dependency_specs():
    return {
        package: solver_mujoco._required_specifier(package, _MOCK_REQUIREMENTS) for package in ("mujoco", "mujoco-warp")
    }


class TestMuJoCoVersionCheck(unittest.TestCase):
    def setUp(self):
        mock_dist = types.SimpleNamespace(read_text=lambda name: _MOCK_METADATA)
        patcher = mock.patch.object(solver_mujoco.importlib_metadata, "distribution", return_value=mock_dist)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_warns_when_installed_versions_do_not_satisfy_pyproject(self):
        specs = _mujoco_dependency_specs()
        versions = {
            package: "0.0.0"
            for package, specifier in specs.items()
            if specifier and not solver_mujoco._version_satisfies("0.0.0", specifier)
        }

        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
            with self.assertWarnsRegex(
                NewtonWarning,
                r"MuJoCo dependency version mismatch.*mujoco==0\.0\.0.*mujoco-warp==0\.0\.0",
            ):
                solver_mujoco._warn_if_mujoco_versions_mismatch(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(),
                )

    def test_warns_when_only_mujoco_warp_mismatches_pyproject(self):
        specs = _mujoco_dependency_specs()
        mujoco_warp_bad_version = "0.0.0"
        self.assertFalse(solver_mujoco._version_satisfies(mujoco_warp_bad_version, specs["mujoco-warp"]))

        versions = {"mujoco": _matching_version(specs["mujoco"]), "mujoco-warp": mujoco_warp_bad_version}
        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
            with self.assertWarnsRegex(NewtonWarning, f"mujoco-warp=={mujoco_warp_bad_version}"):
                solver_mujoco._warn_if_mujoco_versions_mismatch(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(),
                )

    def test_import_mujoco_warns_for_cached_mismatched_versions(self):
        specs = _mujoco_dependency_specs()
        versions = {
            package: "0.0.0"
            for package, specifier in specs.items()
            if specifier and not solver_mujoco._version_satisfies("0.0.0", specifier)
        }
        previous_mujoco = solver_mujoco.SolverMuJoCo._mujoco
        previous_mujoco_warp = solver_mujoco.SolverMuJoCo._mujoco_warp
        previous_versions_checked = solver_mujoco.SolverMuJoCo._versions_checked

        try:
            solver_mujoco.SolverMuJoCo._mujoco = types.SimpleNamespace()
            solver_mujoco.SolverMuJoCo._mujoco_warp = types.SimpleNamespace()
            solver_mujoco.SolverMuJoCo._versions_checked = False

            with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
                with self.assertWarnsRegex(NewtonWarning, "MuJoCo dependency version mismatch"):
                    solver_mujoco.SolverMuJoCo.import_mujoco()
        finally:
            solver_mujoco.SolverMuJoCo._mujoco = previous_mujoco
            solver_mujoco.SolverMuJoCo._mujoco_warp = previous_mujoco_warp
            solver_mujoco.SolverMuJoCo._versions_checked = previous_versions_checked

    def test_accepts_versions_that_satisfy_pyproject(self):
        versions = {
            package: _matching_version(specifier)
            for package, specifier in _mujoco_dependency_specs().items()
            if specifier
        }

        with mock.patch.object(solver_mujoco.importlib_metadata, "version", side_effect=versions.get):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                solver_mujoco._warn_if_mujoco_versions_mismatch(
                    types.SimpleNamespace(),
                    types.SimpleNamespace(),
                )

        messages = [str(warning.message) for warning in caught]
        self.assertFalse(any("MuJoCo dependency version mismatch" in message for message in messages))

    def test_required_specifier_returns_none(self):
        cases = {
            "empty requirements": [],
            "package not in requirements": ["warp-lang>=1.0"],
        }
        for name, requirements in cases.items():
            with self.subTest(name):
                self.assertIsNone(solver_mujoco._required_specifier("mujoco", requirements))


class TestMuJoCoDeterminismConfig(unittest.TestCase):
    def test_mujoco_warp_modules_never_take_warp_determinism(self):
        solver = object.__new__(solver_mujoco.SolverMuJoCo)
        solver._deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN
        smooth_module = types.SimpleNamespace(__name__="mujoco_warp._src.smooth")
        forward_module = types.SimpleNamespace(__name__="mujoco_warp._src.forward")

        with (
            mock.patch.object(
                solver_mujoco,
                "_mujoco_warp_modules",
                return_value=[smooth_module, forward_module],
            ),
            mock.patch.object(solver, "_set_module_options") as set_module_options,
        ):
            solver._set_module_determinism()

        mujoco_warp_options = {
            "deterministic": solver_mujoco.wp.DeterministicMode.NOT_GUARANTEED,
            "deterministic_max_records": 0,
        }
        newton_options = {
            "deterministic": solver_mujoco.wp.DeterministicMode.RUN_TO_RUN,
            "deterministic_max_records": 0,
        }
        self.assertEqual(
            set_module_options.call_args_list,
            [
                mock.call(mujoco_warp_options, module=smooth_module),
                mock.call(mujoco_warp_options, module=forward_module),
                mock.call(newton_options, module=solver_mujoco.kernels),
                mock.call(
                    newton_options,
                    module=solver_mujoco.convert_newton_contacts_to_mjwarp_kernel.module,
                ),
            ],
        )

    def test_scoped_execution_disables_warp_determinism_and_restores_globals(self):
        solver = object.__new__(solver_mujoco.SolverMuJoCo)
        solver._deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN

        original_mode = solver_mujoco.wp.config.deterministic
        original_max_records = solver_mujoco.wp.config.deterministic_max_records
        try:
            solver_mujoco.wp.config.deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN
            solver_mujoco.wp.config.deterministic_max_records = 17
            with (
                mock.patch.object(solver, "_apply_module_options") as apply_module_options,
                solver._scoped_mujoco_warp_execution(),
            ):
                apply_module_options.assert_called_once_with()
                self.assertEqual(
                    solver_mujoco.wp.config.deterministic,
                    solver_mujoco.wp.DeterministicMode.NOT_GUARANTEED,
                )
                self.assertEqual(solver_mujoco.wp.config.deterministic_max_records, 0)

            self.assertEqual(solver_mujoco.wp.config.deterministic, solver_mujoco.wp.DeterministicMode.RUN_TO_RUN)
            self.assertEqual(solver_mujoco.wp.config.deterministic_max_records, 17)
        finally:
            solver_mujoco.wp.config.deterministic = original_mode
            solver_mujoco.wp.config.deterministic_max_records = original_max_records

    def test_notify_model_changed_runs_mjwarp_without_warp_determinism(self):
        solver = object.__new__(solver_mujoco.SolverMuJoCo)
        solver.use_mujoco_cpu = False
        solver._deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN
        solver.has_connect_constraints = False
        solver.has_jnt_connect_constraints = False
        observed_options = []

        def observe_options():
            observed_options.append(
                (
                    solver_mujoco.wp.config.deterministic,
                    solver_mujoco.wp.config.deterministic_max_records,
                )
            )

        original_mode = solver_mujoco.wp.config.deterministic
        original_max_records = solver_mujoco.wp.config.deterministic_max_records
        try:
            solver_mujoco.wp.config.deterministic = solver_mujoco.wp.DeterministicMode.RUN_TO_RUN
            solver_mujoco.wp.config.deterministic_max_records = 17
            with (
                mock.patch.object(solver, "_apply_module_options") as apply_module_options,
                mock.patch.object(solver, "_update_model_properties", side_effect=observe_options),
                mock.patch.object(solver, "_invalidate_contact_fast_path"),
            ):
                solver.notify_model_changed(solver_mujoco.ModelFlags.MODEL_PROPERTIES)

            apply_module_options.assert_called_once_with()
            self.assertEqual(
                observed_options,
                [(solver_mujoco.wp.DeterministicMode.NOT_GUARANTEED, 0)],
            )
        finally:
            solver_mujoco.wp.config.deterministic = original_mode
            solver_mujoco.wp.config.deterministic_max_records = original_max_records

    def test_mujoco_warp_rejects_gpu_to_gpu(self):
        model = newton.ModelBuilder().finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "GPU_TO_GPU"):
            solver_mujoco.SolverMuJoCo(
                model,
                use_mujoco_cpu=False,
                deterministic=solver_mujoco.wp.DeterministicMode.GPU_TO_GPU,
            )


def _matching_version(specifier: str) -> str:
    for pattern in (r">=\s*([0-9][^,;]*)", r"~=\s*([0-9][^,;]*)"):
        match = solver_mujoco.re.search(pattern, specifier)
        if match:
            return match.group(1)
    raise ValueError(f"_matching_version cannot derive a satisfying version from specifier {specifier!r}")


if __name__ == "__main__":
    unittest.main()
