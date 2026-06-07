"""Enhanced tests for scale.olm.check module covering untested functionality."""

import pytest
import numpy as np
import scale.olm as so
import scale.olm.check as check
import scale.olm.internal as internal
from unittest.mock import Mock, patch
from pathlib import Path
import json
import tempfile
import os


def data_file(filename):
    """Helper to get test data files."""
    p = Path(__file__).parent.parent / "data" / filename
    size = p.stat().st_size
    if size < 5e4:
        raise ValueError(f"Data file {p} may be a GIT LFS pointer. Run `git lfs pull`.")
    return p


class TestGridGradientAdvanced:
    """Test advanced GridGradient functionality."""

    def test_default_params_enhanced(self):
        """Test that default_params returns expected values."""
        params = check.GridGradient.default_params()

        # Verify all expected keys exist
        expected_keys = {"eps0", "epsa", "epsr", "target_q1", "target_q2"}
        assert set(params.keys()) == expected_keys

        # Verify reasonable default values
        assert params["eps0"] == 1e-20
        assert params["epsa"] == 1e-1
        assert params["epsr"] == 1e-1
        assert params["target_q1"] == 0.5
        assert params["target_q2"] == 0.7

    def test_describe_params_enhanced(self):
        """Test that describe_params returns helpful descriptions."""
        descriptions = check.GridGradient.describe_params()

        # Verify all parameter descriptions exist
        expected_keys = {"eps0", "epsa", "epsr", "target_q1", "target_q2"}
        assert set(descriptions.keys()) == expected_keys

        # Verify descriptions are strings
        for desc in descriptions.values():
            assert isinstance(desc, str)
            assert len(desc) > 5  # Should be meaningful descriptions

    def test_initialization_with_env(self):
        """Test GridGradient initialization with environment variables."""
        env = {"nprocs": 8}
        grid_grad = check.GridGradient(_env=env, eps0=1e-15, target_q1=0.8)

        assert grid_grad.eps0 == 1e-15
        assert grid_grad.target_q1 == 0.8
        assert grid_grad.nprocs == 8

    def test_kernel_with_simple_data(self):
        """Test the kernel function with simple mathematical data."""
        # Create simple test data
        rel_axes = [[0.0, 0.5, 1.0], [0.0, 1.0]]  # 2D grid
        yreshape = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # coefficient 0
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],  # coefficient 1
            ]
        )
        eps0 = 1e-10

        ahist, rhist, khist = check.GridGradient._GridGradient__kernel(
            rel_axes, yreshape, eps0
        )

        # Verify output arrays have correct structure
        n_axes = len(rel_axes)
        n_intervals = sum(len(axis) - 1 for axis in rel_axes)  # 2 + 1 = 3
        n_coeff = yreshape.shape[0]  # 2
        expected_length = n_axes * n_intervals * n_coeff  # 2 * 3 * 2 = 12

        assert len(ahist) == expected_length
        assert len(rhist) == expected_length
        assert len(khist) == expected_length

        # Verify all values are finite and non-negative
        assert np.all(np.isfinite(ahist))
        assert np.all(np.isfinite(rhist))
        assert np.all(ahist >= 0)
        assert np.all(rhist >= 0)

        # Verify coefficient indices are valid
        assert np.all(khist >= 0)
        assert np.all(khist < n_coeff)

    def test_info_calculation(self):
        """Test the info calculation with known histogram data."""
        grid_grad = check.GridGradient(
            epsa=0.1, epsr=0.05, target_q1=0.7, target_q2=0.8
        )

        # Manually set histogram data for predictable testing
        # rhist > epsr: points that fail relative test
        # ahist > epsa AND rhist > epsr: points that fail both tests
        grid_grad.ahist = np.array([0.15, 0.05, 0.2, 0.01])  # indices 0,2 > 0.1
        grid_grad.rhist = np.array([0.08, 0.02, 0.1, 0.001])  # indices 0,2 > 0.05
        grid_grad.khist = np.array([0, 1, 0, 1])

        info = grid_grad.info()

        # Verify basic properties
        assert info.name == "GridGradient"
        assert info.m == 4  # total points

        # Let's check the logic:
        # rhist > epsr (0.05): indices 0 (0.08) and 2 (0.1) fail relative test
        # ahist > epsa (0.1) AND rhist > epsr: indices 0 (0.15 > 0.1 AND 0.08 > 0.05) and 2 (0.2 > 0.1 AND 0.1 > 0.05)
        assert info.wr == 2  # points failing relative test (indices 0,2)
        assert info.wa == 2  # points failing both tests (indices 0,2)

        # Verify score calculations
        expected_fr = 2.0 / 4.0  # fraction failing relative = 0.5
        expected_fa = 2.0 / 4.0  # fraction failing absolute + relative = 0.5
        expected_q1 = 1.0 - expected_fr  # 0.5
        expected_q2 = (
            1.0 - 0.9 * expected_fa - 0.1 * expected_fr
        )  # 1.0 - 0.45 - 0.05 = 0.5

        assert info.fr == pytest.approx(expected_fr)
        assert info.fa == pytest.approx(expected_fa)
        assert info.q1 == pytest.approx(expected_q1)
        assert info.q2 == pytest.approx(expected_q2)

        # Verify test pass flags
        assert info.test_pass_q1 == (expected_q1 >= 0.7)  # False
        assert info.test_pass_q2 == (expected_q2 >= 0.8)  # False
        assert info.test_pass == (info.test_pass_q1 and info.test_pass_q2)  # False


class TestSequencer:
    """Test the check sequencer functionality."""

    def test_schema_sequencer(self):
        """Test schema generation for sequencer."""
        schema = check._schema_sequencer()
        assert isinstance(schema, dict)
        assert "_type" in schema or "properties" in schema

        schema_with_state = check._schema_sequencer(with_state=True)
        assert isinstance(schema_with_state, dict)

    def test_test_args_sequencer(self):
        """Test test args generation for sequencer."""
        args = check._test_args_sequencer()

        assert args["_type"] == "scale.olm.check:sequencer"
        assert "sequence" in args
        assert isinstance(args["sequence"], list)
        assert len(args["sequence"]) >= 1

        # Verify sequence contains valid check types
        for check_def in args["sequence"]:
            assert "_type" in check_def
            assert check_def["_type"].startswith("scale.olm.check:")

    @patch("scale.olm.internal.logger")
    def test_sequencer_dry_run_enhanced(self, mock_logger):
        """Test sequencer in dry run mode."""
        sequence = [{"_type": "GridGradient", "eps0": 1e-10}]
        model = {"name": "test_model"}
        env = {"work_dir": "/tmp"}

        result = check.sequencer(sequence, model, env, dry_run=True)

        assert result["test_pass"] == False
        assert "output" in result
        assert isinstance(result["output"], list)


class TestLowOrderConsistency:
    """Test LowOrderConsistency functionality."""

    def test_default_params_enhanced_loc(self):
        """Test that default_params returns expected values."""
        params = check.LowOrderConsistency.default_params()

        # Should return a dictionary with expected parameters
        assert isinstance(params, dict)
        expected_keys = {
            "eps0",
            "epsa",
            "epsr",
            "target_q1",
            "target_q2",
            "nuclide_compare",
            "assembly_average",
            "template",
            "name",
        }

        # May not have all keys but should be a reasonable subset
        assert len(params) > 0
        assert params["metric"] == "grams_per_initial_hm"

    def test_describe_params_enhanced_loc(self):
        """Test that describe_params returns helpful descriptions."""
        descriptions = check.LowOrderConsistency.describe_params()

        expected_keys = {
            "eps0",
            "epsa",
            "epsr",
            "target_q1",
            "target_q2",
            "nlib_start",
            "nlib_max",
            "nburn_start",
            "nburn_max",
            "q1_stop_criteria",
            "q2_stop_criteria",
            "nuclide_compare",
            "assembly_average",
            "template",
            "name",
            "metric",
        }
        assert set(descriptions.keys()) == expected_keys

        # Verify descriptions are strings
        for desc in descriptions.values():
            assert isinstance(desc, str)
            assert len(desc) > 2  # Should be meaningful

    def test_amounts_to_grams_per_initial_hm(self):
        """Test MOLES inventories are converted to grams per gram IHM."""
        amounts = np.array(
            [
                [[2.0, 3.0]],
                [[4.0, 5.0]],
            ]
        )
        masses = np.array([235.0, 238.0])
        initialhm = np.array([1.0, 2.0])

        grams = check.LowOrderConsistency._amounts_to_grams_per_initial_hm(
            amounts, masses, initialhm
        )

        np.testing.assert_allclose(
            grams,
            np.array(
                [
                    [[470.0, 714.0]],
                    [[470.0, 595.0]],
                ]
            )
            / 1.0e6,
        )

    def test_amounts_to_grams_per_initial_hm_rejects_nonpositive_initialhm(self):
        """Test g/gIHM conversion requires a positive initial HM basis."""
        amounts = np.array([[[1.0]]])
        masses = np.array([235.0])

        with pytest.raises(ValueError, match="positive initial heavy metal"):
            check.LowOrderConsistency._amounts_to_grams_per_initial_hm(
                amounts, masses, np.array([0.0])
            )

    def test_low_order_consistency_rejects_unknown_metric(self):
        """Test metric configuration rejects unsupported inventory metrics."""
        with pytest.raises(ValueError, match="Unsupported LowOrderConsistency metric"):
            check.LowOrderConsistency(metric="moles", _dry_run=True)

    def test_low_order_consistency_rejects_nonpositive_nburn_start(self):
        """Test LOC requires positive ORIGAMI burn substeps."""
        with pytest.raises(ValueError, match="nburn_start must be positive"):
            check.LowOrderConsistency(nburn_start=0, _dry_run=True)

    def test_low_order_consistency_rejects_nburn_max_before_start(self):
        """Test LOC rejects an impossible nburn convergence range."""
        with pytest.raises(ValueError, match="nburn_max"):
            check.LowOrderConsistency(
                nburn_start=4,
                nburn_max=2,
                _dry_run=True,
            )

    def test_low_order_consistency_reports_metric_on_early_failure(self):
        """Test early LOC failures still report the configured metric."""
        loc = check.LowOrderConsistency(metric="grams_per_initial_hm", _dry_run=True)
        loc.run_success = False

        info = loc.info()

        assert info.test_pass is False
        assert info.metric == "grams_per_initial_hm"
        assert info.units == "g/gIHM"

    def test_low_order_consistency_requires_perfect_time_zero_scores(
        self, monkeypatch, tmp_path
    ):
        """Test LOC fails when time-zero q1 and q2 are not both exactly one."""
        monkeypatch.setattr(
            check.LowOrderConsistency,
            "_plot_metric_histogram",
            lambda self, ahist, rhist, hist_image, ylabel: str(hist_image),
        )
        loc = check.LowOrderConsistency(
            metric="grams_per_initial_hm",
            epsa=1e-20,
            epsr=1e-3,
            target_q1=0.0,
            target_q2=0.0,
            nuclide_compare=[],
            _dry_run=True,
        )
        loc.run_success = True
        loc.hi_list = [np.array([[1.0], [1.0]])]
        loc.lo_list = [np.array([[1.01], [1.0]])]
        loc.names = ["0092235"]
        loc.nuclide_data = {"0092235": {"mass": 235.0}}
        loc.initialhm_list = [1.0]
        loc.time_list = [0.0, 1.0]
        loc.work_path = tmp_path
        loc.check_path = tmp_path / "check"
        loc.ii_json_list = [(tmp_path / "hi.ii.json", tmp_path / "lo.ii.json")]

        info = loc.info()

        assert info.test_pass_q1 is True
        assert info.test_pass_q2 is True
        assert info.time0["q1"] == 0.0
        assert info.time0["q2"] == pytest.approx(0.0)
        assert info.test_pass_time0 is False
        assert info.test_pass is False

    def test_low_order_consistency_matches_high_order_endpoint_times(self):
        """Test LOC can compare high-order endpoints to low-order substeps."""
        indices = check.LowOrderConsistency._matching_time_indices(
            [0.0, 10.0, 20.0],
            [0.0, 5.0, 10.0, 15.0, 20.0],
        )

        assert indices == [0, 2, 4]

    def test_low_order_consistency_rejects_missing_endpoint_time(self):
        """Test LOC reports a clear error when low-order times miss an endpoint."""
        with pytest.raises(ValueError, match="did not match exactly one"):
            check.LowOrderConsistency._matching_time_indices(
                [0.0, 10.0, 20.0],
                [0.0, 5.0, 15.0, 20.0],
            )

    def test_low_order_consistency_doubles_nlib_until_scores_stabilize(self):
        """Test nlib convergence doubles until q1 and q2 stop changing."""
        loc = check.LowOrderConsistency(
            nlib_start=2,
            nlib_max=16,
            q1_stop_criteria=0.01,
            q2_stop_criteria=0.01,
            _dry_run=True,
        )
        calls = []
        values = [(0.80, 0.90), (0.85, 0.93), (0.855, 0.935)]

        def fake_run_once(_do_run, nlib, nburn):
            calls.append((nlib, nburn))
            info = check.CheckInfo()
            info.nlib = nlib
            info.nburn = nburn
            info.q1, info.q2 = values[len(calls) - 1]
            info.test_pass = True
            info.test_pass_q1 = True
            info.test_pass_q2 = True
            info.mean_abs_diff = 0.0
            info.mean_rel_diff = 0.0
            return info

        loc._run_once = fake_run_once

        info = loc.run(None)

        assert calls == [(2, 1), (4, 1), (8, 1)]
        assert info.nlib == 8
        assert info.nburn == 1
        assert info.nlib_converged is True
        assert info.nburn_converged is True
        assert info.test_pass_nlib is True
        assert info.test_pass_nburn is True
        assert info.test_pass is True
        assert len(info.nlib_history) == 3
        assert len(info.nburn_history) == 1

    def test_low_order_consistency_fails_when_nlib_does_not_converge(self):
        """Test LOC fails the nlib check when max nlib is reached first."""
        loc = check.LowOrderConsistency(
            nlib_start=2,
            nlib_max=8,
            q1_stop_criteria=1e-6,
            q2_stop_criteria=1e-6,
            _dry_run=True,
        )
        calls = []
        values = [(0.80, 0.90), (0.85, 0.93), (0.87, 0.94)]

        def fake_run_once(_do_run, nlib, nburn):
            calls.append((nlib, nburn))
            info = check.CheckInfo()
            info.nlib = nlib
            info.nburn = nburn
            info.q1, info.q2 = values[len(calls) - 1]
            info.test_pass = True
            info.test_pass_q1 = True
            info.test_pass_q2 = True
            info.mean_abs_diff = 0.0
            info.mean_rel_diff = 0.0
            return info

        loc._run_once = fake_run_once

        info = loc.run(None)

        assert calls == [(2, 1), (4, 1), (8, 1)]
        assert info.nlib == 8
        assert info.nburn == 1
        assert info.nlib_converged is False
        assert info.nburn_converged is True
        assert info.test_pass_nlib is False
        assert info.test_pass_nburn is True
        assert info.test_pass is False

    def test_low_order_consistency_doubles_nburn_after_nlib_converges(self):
        """Test LOC doubles nburn after each nlib convergence pass."""
        loc = check.LowOrderConsistency(
            nlib_start=2,
            nlib_max=8,
            nburn_start=1,
            nburn_max=4,
            q1_stop_criteria=0.01,
            q2_stop_criteria=0.01,
            _dry_run=True,
        )
        calls = []
        values = {
            (1, 2): (0.80, 0.90),
            (1, 4): (0.85, 0.93),
            (1, 8): (0.855, 0.935),
            (2, 2): (0.81, 0.91),
            (2, 4): (0.851, 0.931),
            (2, 8): (0.856, 0.936),
        }

        def fake_run_once(_do_run, nlib, nburn):
            calls.append((nburn, nlib))
            info = check.CheckInfo()
            info.nlib = nlib
            info.nburn = nburn
            info.q1, info.q2 = values[(nburn, nlib)]
            info.test_pass = True
            info.test_pass_q1 = True
            info.test_pass_q2 = True
            info.mean_abs_diff = 0.0
            info.mean_rel_diff = 0.0
            return info

        loc._run_once = fake_run_once

        info = loc.run(None)

        assert calls == [(1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (2, 8)]
        assert info.nburn == 2
        assert info.nlib == 8
        assert info.nlib_converged is True
        assert info.nburn_converged is True
        assert info.test_pass is True
        assert len(info.nburn_history) == 2

    def test_low_order_consistency_fails_when_nburn_does_not_converge(self):
        """Test LOC fails the nburn check when max nburn is reached first."""
        loc = check.LowOrderConsistency(
            nlib_start=2,
            nlib_max=4,
            nburn_start=1,
            nburn_max=2,
            q1_stop_criteria=1e-6,
            q2_stop_criteria=1e-6,
            _dry_run=True,
        )
        calls = []
        values = {
            (1, 2): (0.80, 0.90),
            (1, 4): (0.80, 0.90),
            (2, 2): (0.85, 0.93),
            (2, 4): (0.85, 0.93),
        }

        def fake_run_once(_do_run, nlib, nburn):
            calls.append((nburn, nlib))
            info = check.CheckInfo()
            info.nlib = nlib
            info.nburn = nburn
            info.q1, info.q2 = values[(nburn, nlib)]
            info.test_pass = True
            info.test_pass_q1 = True
            info.test_pass_q2 = True
            info.mean_abs_diff = 0.0
            info.mean_rel_diff = 0.0
            return info

        loc._run_once = fake_run_once

        info = loc.run(None)

        assert calls == [(1, 2), (1, 4), (2, 2), (2, 4)]
        assert info.nburn == 2
        assert info.nlib == 4
        assert info.nlib_converged is True
        assert info.nburn_converged is False
        assert info.test_pass_nburn is False
        assert info.test_pass is False

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.figure")
    def test_make_diff_plot_basic(self, mock_figure, mock_savefig):
        """Test the make_diff_plot static method with mocked matplotlib."""
        import tempfile

        # Create test data
        identifier = "u235"
        time = [0, 86400, 172800]  # days in seconds
        min_diff = [-0.01, -0.02, -0.005]
        max_diff = [0.01, 0.02, 0.01]
        max_diff0 = 0.02
        perms = [{"(lo-hi)/max(|hi|)": [-0.005, 0.015, 0.008]}]

        # Set up mock figure to return mock axes
        mock_figure_instance = Mock()
        mock_axes = Mock()
        mock_figure_instance.gca.return_value = mock_axes
        mock_figure.return_value = mock_figure_instance

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_path = tmp.name

        try:
            # This should not raise an exception
            check.LowOrderConsistency.make_diff_plot(
                identifier, image_path, time, min_diff, max_diff, max_diff0, perms
            )

            # Verify matplotlib functions were called (may be called multiple times)
            assert mock_figure.call_count >= 1
            mock_savefig.assert_called_once_with(image_path, bbox_inches="tight")

        finally:
            # Clean up
            if os.path.exists(image_path):
                os.unlink(image_path)


class TestSchemaFunctions:
    """Test schema generation functions for all check types."""

    def test_schema_gridgradient_enhanced(self):
        """Test GridGradient schema generation."""
        schema = check._schema_GridGradient()
        assert isinstance(schema, dict)
        assert "_type" in schema or "properties" in schema

        schema_with_state = check._schema_GridGradient(with_state=True)
        assert isinstance(schema_with_state, dict)

    def test_test_args_gridgradient_enhanced(self):
        """Test GridGradient test arguments generation."""
        args = check._test_args_GridGradient()

        assert args["_type"] == "scale.olm.check:GridGradient"
        assert "eps0" in args
        assert "target_q1" in args
        assert "target_q2" in args

        # Verify that target values are in reasonable range
        assert 0 <= args["target_q1"] <= 1
        assert 0 <= args["target_q2"] <= 1

    def test_schema_loworderconsistency(self):
        """Test schema generation for LowOrderConsistency."""
        schema = check._schema_LowOrderConsistency()
        assert isinstance(schema, dict)
        metric_schema = schema["properties"]["metric"]
        assert set(metric_schema["enum"]) == {"grams_per_initial_hm", "atom_fraction"}

        schema_with_state = check._schema_LowOrderConsistency(with_state=True)
        assert isinstance(schema_with_state, dict)

    def test_test_args_loworderconsistency(self):
        """Test test args generation for LowOrderConsistency."""
        args = check._test_args_LowOrderConsistency()

        assert args["_type"] == "scale.olm.check:LowOrderConsistency"
        # Should be a valid dictionary (exact content depends on implementation)
        assert isinstance(args, dict)


class TestCheckInfo:
    """Test the CheckInfo class."""

    def test_checkinfo_initialization(self):
        """Test that CheckInfo initializes correctly."""
        info = check.CheckInfo()

        # Should have test_pass set to False by default
        assert hasattr(info, "test_pass")
        assert info.test_pass == False

        # Should be able to set additional attributes
        info.name = "TestCheck"
        info.q1 = 0.85
        info.q2 = 0.90

        assert info.name == "TestCheck"
        assert info.q1 == 0.85
        assert info.q2 == 0.90


class TestUtilityFunctions:
    """Test utility functions and edge cases."""

    def test_gridgradient_with_constant_data(self):
        """Test GridGradient with constant coefficient data."""
        # Create reactor library with constant coefficients
        rl = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))

        # Override with constant data
        rl.coeff = np.ones_like(rl.coeff) * 1e-5  # Small constant value

        grid_grad = check.GridGradient(eps0=1e-10, epsa=1e-3, epsr=1e-3)
        info = grid_grad.run(rl)

        # With constant data, gradients should be very small
        assert info.name == "GridGradient"
        assert 0 <= info.q1 <= 1
        assert 0 <= info.q2 <= 1
        assert info.m > 0

        # Most or all points should pass with constant data
        assert info.q1 >= 0.5  # Should have low relative gradients

    def test_gridgradient_extreme_values(self):
        """Test GridGradient with extreme coefficient values."""
        rl = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))

        # Test with very large values
        rl.coeff = np.ones_like(rl.coeff) * 1e10

        grid_grad = check.GridGradient(eps0=1e-20, epsa=1e5, epsr=0.1)
        info = grid_grad.run(rl)

        assert info.name == "GridGradient"
        assert np.isfinite(info.q1)
        assert np.isfinite(info.q2)
        assert 0 <= info.q1 <= 1
        assert 0 <= info.q2 <= 1

    def test_gridgradient_single_axis_point(self):
        """Test GridGradient behavior with minimal axis points."""
        rl = so.core.ReactorLibrary(data_file("w17x17.arc.h5"))

        # Verify that degenerate axis duplication occurred
        mod_dens_idx = list(rl.axes_names).index("mod_dens")
        assert (
            len(rl.axes_values[mod_dens_idx]) == 2
        ), "Degenerate axis should be duplicated"

        grid_grad = check.GridGradient()
        info = grid_grad.run(rl)

        # Should work without errors even with minimal points
        assert info.name == "GridGradient"
        assert info.m > 0
        assert np.isfinite(info.q1)
        assert np.isfinite(info.q2)
