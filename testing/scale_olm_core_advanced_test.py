"""
Advanced tests for scale.olm.core module.

This module tests the mathematical algorithms, composition calculations,
and data processing functionality of the core module to improve coverage.
Focus on testing real functionality with minimal mocking.
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import scale.olm.core as core


_TEMPLATE_DIR = Path(core.__file__).parent / "templates"


class TestTemplateManager:
    """Test template expansion and inheritance behavior."""

    def test_expand_uses_template_manager_search_paths_for_inheritance(self, tmp_path):
        """TemplateManager.expand should find parents from its configured roots."""
        template_root = tmp_path / "templates"
        child_dir = template_root / "child"
        child_dir.mkdir(parents=True)
        (template_root / "base.jt.inp").write_text(
            "base:{% block body %}default{% endblock %}"
        )
        (child_dir / "model.jt.inp").write_text(
            '{% extends "base.jt.inp" %}{% block body %}{{ noun }}{% endblock %}'
        )

        tm = core.TemplateManager(paths=[template_root], include_env=False)

        assert tm.expand("child/model.jt.inp", {"noun": "fuel"}) == "base:fuel"

    def test_expand_file_uses_source_directory_not_cwd_parent(
        self, tmp_path, monkeypatch
    ):
        """expand_file should resolve inherited templates from the source path."""
        template_dir = tmp_path / "config"
        template_dir.mkdir()
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (tmp_path / "base.jt.inp").write_text(
            "wrong:{% block body %}default{% endblock %}"
        )
        (template_dir / "base.jt.inp").write_text(
            "right:{% block body %}default{% endblock %}"
        )
        child = template_dir / "child.jt.inp"
        child.write_text(
            '{% extends "base.jt.inp" %}{% block body %}{{ noun }}{% endblock %}'
        )
        monkeypatch.chdir(work_dir)

        assert core.TemplateManager.expand_file(child, {"noun": "fuel"}) == "right:fuel"

    def test_expand_text_renders_floats_as_scientific(self):
        """Template float output should preserve a deterministic scientific form."""
        text = "x={{ x }} y={{ y }}"
        data = {"x": 39.4934, "y": 1.0}

        rendered = core.TemplateManager.expand_text(text, data)

        assert rendered == "x=3.94934000000000e+01 y=1.00000000000000e+00"

    def test_tree_print_renders_floats_as_scientific(self):
        """Fallback generated input should use the same float rendering policy."""
        rendered = core.TemplateManager._tree_print({"x": [39.4934], "y": 1.0})

        assert "x[0]=3.94934000000000e+01" in rendered
        assert "y=1.00000000000000e+00" in rendered

    def test_origami_uox_template_requires_nlib(self):
        """LowOrderConsistency ORIGAMI templates require explicit nlib."""
        template = _TEMPLATE_DIR / "model/origami/system-uox.jt.inp"
        data = {
            "_": {
                "env": {"work_dir": "/tmp/olm-work"},
                "model": {"name": "uox-pin-quick"},
            },
            "history": {
                "initialhm": 1.0,
                "burndata": [{"power": 40.0, "burn": 25.0}],
            },
            "_arpinfo": {
                "interpvars": {
                    "enrichment": 3.0,
                    "mod_dens": 0.72,
                }
            },
            "comp": {
                "system": {
                    "uo2": {
                        "iso": {
                            "u234": 0.0254268158073155,
                            "u235": 3.0,
                            "u236": 0.0138,
                            "u238": 96.9607731841927,
                        }
                    }
                }
            },
            "convergence_control": {"nburn": 10},
        }

        with pytest.raises(ValueError, match="nlib"):
            core.TemplateManager.expand_file(template, data)

    def test_origami_uox_template_requires_nburn(self):
        """LowOrderConsistency ORIGAMI templates require explicit nburn."""
        template = _TEMPLATE_DIR / "model/origami/system-uox.jt.inp"
        data = {
            "_": {
                "env": {"work_dir": "/tmp/olm-work"},
                "model": {"name": "uox-pin-quick"},
            },
            "history": {
                "initialhm": 1.0,
                "burndata": [{"power": 40.0, "burn": 25.0}],
            },
            "_arpinfo": {
                "interpvars": {
                    "enrichment": 3.0,
                    "mod_dens": 0.72,
                }
            },
            "comp": {
                "system": {
                    "uo2": {
                        "iso": {
                            "u234": 0.0254268158073155,
                            "u235": 3.0,
                            "u236": 0.0138,
                            "u238": 96.9607731841927,
                        }
                    }
                }
            },
            "convergence_control": {"nlib": 1},
        }

        with pytest.raises(ValueError, match="nburn"):
            core.TemplateManager.expand_file(template, data)

    def test_origami_uox_gd2o3_template_smears_bwr_gd_pins(self):
        """Polaris BWR ORIGAMI input includes mass-smeared Gd2O3."""
        template = _TEMPLATE_DIR / "model/origami/system-uox-gd2o3.jt.inp"
        data = {
            "_": {
                "env": {"work_dir": "/tmp/olm-work"},
                "model": {"name": "polaris_uoxgd_quick"},
            },
            "history": {
                "initialhm": 1.0,
                "burndata": [{"power": 25.0, "burn": 20.0}],
            },
            "_arpinfo": {"interpvars": {"mod_dens": 0.75}},
            "comp": {
                "system": {
                    "uo2": {
                        "iso": {
                            "u234": 0.007744690378774326,
                            "u235": 0.9995296664900924,
                            "u236": 0.004597836465854426,
                            "u238": 98.98812780666528,
                        }
                    }
                }
            },
            "assembly_average": {
                "gd2o3_pin_wtpt": 3.0,
                "gd2o3_pin_count": 4,
                "fuel_pin_count": 49,
                "uox_fuel_density": 10.4,
                "gd2o3_fuel_density": 10.19,
            },
            "convergence_control": {
                "nburn": 10,
                "nlib": 4,
            },
        }

        rendered = core.TemplateManager.expand_file(template, data)

        assert "Gd2O3=2.40349084047488e-01" in rendered
        assert "fuel=9.97596509159525e+01" in rendered
        assert "nburn=10" in rendered
        assert "nlib=4" in rendered

    def test_polaris_uox_pin_template_renders_single_fuel_material(self):
        """Polaris UOX pin template uses one FUEL material and classifies as Polaris."""
        data = {
            "_": {
                "model": {
                    "description": "A 2D Polaris PWR UOX pin cell.",
                    "notes": ["single fuel material"],
                    "sources": {"1": "source"},
                },
            },
            "state": {
                "boron_ppm": 600.0,
                "coolant_density": 0.72,
                "enrichment": 3.0,
                "specific_power": 40.0,
            },
            "static": {
                "xslib": "broad_lwr",
                "pitch": 1.26,
                "fuelr": 0.4096,
                "gapr": 0.4180,
                "cladr": 0.4750,
            },
            "comp": {"fuel": {"density": 10.4}},
            "time": {"gwd_burnups": [0.5, 1.0]},
        }

        rendered = core.TemplateManager.expand_file(
            _TEMPLATE_DIR / "model/polaris/pin-uox.jt.inp", data
        )
        classification = core.ScaleInput.classify_text(rendered)

        assert rendered.startswith("=polaris")
        assert "geom FuelNode : ASSM 1" in rendered
        assert "comp fuel : UOX enr=3.00000000000000e+00" in rendered
        assert "mat FUEL.1 : fuel" in rendered
        assert "pinmap 1" in rendered
        assert "basis ALL=no FUEL=YES" in rendered
        assert "TrackingSet='Complete'" in rendered
        assert "ArchiveF33='ALL'" in rendered
        assert "Method='predictor'" not in rendered
        assert classification["artifact_contract"] == "Polaris"


class TestScaleInput:
    """Test rendered SCALE input classification."""

    @pytest.mark.parametrize(
        "sequence",
        [
            "t-depl",
            "t-depl-1d",
            "t5-depl",
            "t5-depl-shift",
            "t6-depl",
            "t6-depl-shift",
            "t6-depl-custom",
        ],
    )
    def test_classify_text_maps_triton_prefixes(self, sequence):
        """TRITON depletion contracts are classified from supported prefixes."""
        classification = core.ScaleInput.classify_text(
            f"=origen\n={sequence} parm=(bonami)\nend\n"
        )

        assert classification["sequences"] == ["origen", sequence]
        assert classification["artifact_contract"] == "TRITON"
        assert core.ScaleOutfile.get_product_name(sequence) == "TRITON"

    @pytest.mark.parametrize("sequence", ["polaris", "polaris_6.3", "polaris-custom"])
    def test_classify_text_maps_polaris_prefixes(self, sequence):
        """Polaris contracts are classified from supported prefixes."""
        classification = core.ScaleInput.classify_text(f"={sequence}\nend\n")

        assert classification["artifact_contract"] == "Polaris"
        assert core.ScaleOutfile.get_product_name(sequence) == "Polaris"

    def test_classify_text_rejects_mixed_depletion_contracts(self):
        """Inputs with both TRITON and Polaris depletion contracts are ambiguous."""
        with pytest.raises(ValueError, match="Ambiguous SCALE depletion"):
            core.ScaleInput.classify_text("=t-depl\n=polaris\n")

    def test_classify_text_leaves_unsupported_inputs_uncontracted(self):
        """Unsupported SCALE inputs do not invent an artifact contract."""
        classification = core.ScaleInput.classify_text("% comment\n=origen\nend\n")

        assert classification["sequences"] == ["origen"]
        assert classification["artifact_contract"] is None


class TestCompositionManager:
    """Test the CompositionManager class for nuclide data and calculations."""

    @pytest.fixture
    def sample_nuclide_data(self):
        """Sample nuclide data for testing."""
        return {
            "0001001": {
                "IZZZAAA": "0001001",
                "atomicNumber": 1,
                "element": "H",
                "isomericState": 0,
                "mass": 1.007825,
                "massNumber": 1,
            },
            "0001002": {
                "IZZZAAA": "0001002",
                "atomicNumber": 1,
                "element": "H",
                "isomericState": 0,
                "mass": 2.014102,
                "massNumber": 2,
            },
            "0092235": {
                "IZZZAAA": "0092235",
                "atomicNumber": 92,
                "element": "U",
                "isomericState": 0,
                "mass": 235.044,
                "massNumber": 235,
            },
            "0094239": {
                "IZZZAAA": "0094239",
                "atomicNumber": 94,
                "element": "Pu",
                "isomericState": 0,
                "mass": 239.052,
                "massNumber": 239,
            },
        }

    @pytest.fixture
    def composition_manager(self, sample_nuclide_data):
        """Create a CompositionManager instance for testing."""
        return core.CompositionManager(sample_nuclide_data)

    def test_composition_manager_initialization(self, composition_manager):
        """Test CompositionManager initialization and element mapping."""
        # Test element to atomic number mapping
        assert composition_manager.e_to_z["h"] == 1
        assert composition_manager.e_to_z["u"] == 92
        assert composition_manager.e_to_z["pu"] == 94

        # Test atomic number to element mapping
        assert composition_manager.z_to_e[1] == "h"
        assert composition_manager.z_to_e[92] == "u"
        assert composition_manager.z_to_e[94] == "pu"

    def test_parse_eam_to_eai(self):
        """Test parsing element-mass-isomer identifiers."""
        # Test normal nuclides
        e, a, i = core.CompositionManager.parse_eam_to_eai("u235")
        assert e == "u" and a == 235 and i == 0

        e, a, i = core.CompositionManager.parse_eam_to_eai("pu239")
        assert e == "pu" and a == 239 and i == 0

        # Test metastable states
        e, a, i = core.CompositionManager.parse_eam_to_eai("am242m")
        assert e == "am" and a == 242 and i == 1

        e, a, i = core.CompositionManager.parse_eam_to_eai("tc99m2")
        assert e == "tc" and a == 99 and i == 2

        # Test single-letter elements
        e, a, i = core.CompositionManager.parse_eam_to_eai("h1")
        assert e == "h" and a == 1 and i == 0

        # Test invalid formats
        with pytest.raises(ValueError, match="did not match regular expression"):
            core.CompositionManager.parse_eam_to_eai("invalid123")

    def test_mass_lookup(self, composition_manager):
        """Test mass lookup functionality using real data."""
        # Test direct IZZZAAA lookup
        mass = composition_manager.mass("0092235")
        assert mass == pytest.approx(235.044, abs=0.01)

        # Test with invalid ID - this will return None or default
        result = composition_manager.data.get("nonexistent", {"mass": 100.0})["mass"]
        assert result == 100.0

    def test_renormalize_wtpt(self):
        """Test weight percent renormalization with real calculations."""
        # Test basic renormalization
        wtpt0 = {"u235": 25.0, "u238": 75.0, "pu239": 5.0}
        wtpt, norm = core.CompositionManager.renormalize_wtpt(wtpt0, 100.0)

        # Should include all elements and sum to 100
        assert "u235" in wtpt and "u238" in wtpt and "pu239" in wtpt
        assert sum(wtpt.values()) == pytest.approx(100.0, abs=1e-10)

        # Test with filter
        wtpt_u, norm_u = core.CompositionManager.renormalize_wtpt(wtpt0, 100.0, "u")
        assert "u235" in wtpt_u and "u238" in wtpt_u
        assert "pu239" not in wtpt_u
        assert sum(wtpt_u.values()) == pytest.approx(100.0, abs=1e-10)

    def test_grams_per_mol(self):
        """Test molar mass calculation using harmonic mean formula."""
        # Test simple mixture
        iso_wts = {"u235": 50.0, "pu239": 50.0}
        molar_mass = core.CompositionManager.grams_per_mol(iso_wts, m_data={})

        # Should be close to average of mass numbers: (235 + 239) / 2 = 237
        assert molar_mass == pytest.approx(236.98, abs=0.1)

        # Test with real molar masses
        m_data = {"u235": 235.044, "pu239": 239.052}
        molar_mass = core.CompositionManager.grams_per_mol(iso_wts, m_data)
        expected = 1.0 / (0.5 / 235.044 + 0.5 / 239.052)
        assert molar_mass == pytest.approx(expected, abs=0.01)


class TestBurnupHistory:
    """Test the BurnupHistory class for time-burnup management."""

    def test_burnup_history_initialization(self):
        """Test BurnupHistory initialization with simple data."""
        time = [0, 10, 20, 30, 40]
        burnup = [0, 100, 250, 500, 1000]

        bh = core.BurnupHistory(time, burnup)

        # Verify basic attributes
        assert len(bh.time) == 5
        assert len(bh.burnup) == 5
        assert len(bh.interval_time) == 4
        assert len(bh.interval_burnup) == 4
        assert len(bh.interval_power) == 4

        # Verify interval calculations
        expected_dt = [10, 10, 10, 10]
        expected_dbu = [100, 150, 250, 500]
        expected_power = [10.0, 15.0, 25.0, 50.0]

        np.testing.assert_array_almost_equal(bh.interval_time, expected_dt)
        np.testing.assert_array_almost_equal(bh.interval_burnup, expected_dbu)
        np.testing.assert_array_almost_equal(bh.interval_power, expected_power)

    def test_union_times(self):
        """Test time grid union functionality."""
        a = np.array([0, 10, 20, 30])
        b = np.array([5, 15, 25, 35])

        c = core.BurnupHistory.union_times(a, b)
        expected = np.array([0, 5, 10, 15, 20, 25, 30, 35])

        np.testing.assert_array_equal(c, expected)

    def test_classify_operations_basic(self):
        """Test basic operations classification."""
        time = [0, 5, 10, 50, 55, 100, 105]
        burnup = [0, 0, 100, 500, 500, 1000, 1000]

        bh = core.BurnupHistory(time, burnup)
        result = bh.classify_operations()

        # Verify structure
        assert "options" in result
        assert "operations" in result

        # Verify operations
        operations = result["operations"]
        assert len(operations) >= 3  # At least some operations
        assert operations[0]["start"] == 0

    @patch("scale.olm.core.run_command")
    def test_obiwan_get_f71_history_s63(self, mock_obiwan):
        mock_obiwan.return_value = """

             pos         time        power         flux      fluence       energy    initialhm libpos   case   step DCGNAB
             (-)          (s)         (MW)    (n/cm2-s)      (n/cm2)        (MWd)      (MTIHM)    (-)    (-)    (-)    (-)
               1  0.00000e+00  4.00000e+01  8.11143e+14  0.00000e+00  0.00000e+00  1.00000e+00      1      1      0 DC----
               2  2.16000e+06  4.00000e+01  6.22529e+14  1.53582e+21  1.00000e+03  1.00000e+00      1      1     10 DC----
               3  2.16000e+07  4.00000e+01  4.26681e+14  8.78948e+21  1.00000e+04  1.00000e+00      2      1     10 DC----
               4  5.40000e+07  4.00000e+01  4.26566e+14  1.34274e+22  2.50000e+04  1.00000e+00      3      1     10 DC----
               5  1.08000e+08  4.00000e+01  4.31263e+14  2.30677e+22  5.00000e+04  1.00000e+00      4      1     10 DC----
               6  1.51200e+08  4.00000e+01  4.32303e+14  1.86058e+22  7.00000e+04  1.00000e+00      5      1     10 DC----
               7  1.94400e+08  4.00000e+01  4.33742e+14  1.86669e+22  9.00000e+04  1.00000e+00      6      1     10 DC----
               8  2.37600e+08  4.00000e+01  4.35733e+14  1.87415e+22  1.10000e+05  1.00000e+00      7      1     10 DC----
"""
        # TRITON: power is over the interval
        triton_hist = core.Obiwan.get_history_from_f71("obiwan.exe", "perm000.f71", 1)
        mock_obiwan.assert_called_once()
        expected_hist = {
            "burndata": [
                {"power": 40.0, "burn": 25.0},
                {"power": 40.0, "burn": 225.0},
                {"power": 40.0, "burn": 375.0},
                {"power": 40.0, "burn": 625.0},
                {"power": 40, "burn": 500.0},
                {"power": 40.0, "burn": 500.0},
                {"power": 40.0, "burn": 500.0},
            ],
            "initialhm": 1.0,
        }

        np.testing.assert_almost_equal(
            expected_hist["initialhm"], triton_hist["initialhm"]
        )
        assert len(expected_hist["burndata"]) == len(triton_hist["burndata"])
        for t_burndata, e_burndata in zip(
            triton_hist["burndata"], expected_hist["burndata"]
        ):
            np.testing.assert_almost_equal(t_burndata["power"], e_burndata["power"])
            np.testing.assert_almost_equal(t_burndata["burn"], e_burndata["burn"])

        # Polaris gives power at the end of the timestep, so modify the last
        # step accordingly and verify our history is still as-expected
        polaris_rv = (
            "\n".join(mock_obiwan.return_value.split("\n")[:-2])
            + """
               8  2.37600e+08  0.10000e-03  0.00000e+00  1.87415e+22  1.10000e+05  1.00000e+00      7      1     10 DC----
"""
        )
        mock_obiwan.return_value = polaris_rv
        polaris_hist = core.Obiwan.get_history_from_f71(
            "obiwan.exe", "perm000.f71", 1, is_polaris=True
        )
        assert mock_obiwan.call_count == 2
        np.testing.assert_almost_equal(
            expected_hist["initialhm"], polaris_hist["initialhm"]
        )

        assert len(expected_hist["burndata"]) == len(polaris_hist["burndata"])
        for p_burndata, e_burndata in zip(
            polaris_hist["burndata"], expected_hist["burndata"]
        ):
            np.testing.assert_almost_equal(p_burndata["power"], e_burndata["power"])
            np.testing.assert_almost_equal(p_burndata["burn"], e_burndata["burn"])

    @patch("scale.olm.core.run_command")
    def test_obiwan_get_f71_history_s70(self, mock_obiwan):
        mock_obiwan.return_value = """

            pos         time        power         flux      fluence       energy    initialhm       volume libpos   case   step DCGNAB
            (-)          (s)         (MW)   (n/cm^2-s)     (n/cm^2)        (MWd)      (MTIHM)       (cm^3)    (-)    (-)    (-)    (-)
              1  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  1.00000e+00  1.09091e+05      1     10      0 DC----
              2  2.16000e+06  3.99302e+01  2.77611e+14  5.99639e+20  9.98255e+02  1.00000e+00  1.09091e+05      2     10      1 DC----
              3  2.16000e+07  3.99294e+01  2.88762e+14  6.21316e+21  9.98238e+03  1.00000e+00  1.09091e+05      3     10      2 DC----
              4  5.40000e+07  3.99271e+01  3.13691e+14  1.63767e+22  2.49551e+04  1.00000e+00  1.09091e+05      4     10      3 DC----
              5  8.10000e+07  3.99215e+01  3.42857e+14  2.56339e+22  3.74305e+04  1.00000e+00  1.09091e+05      5     10      4 DC----
              6  1.08000e+08  3.99155e+01  3.70174e+14  3.56286e+22  4.99041e+04  1.00000e+00  1.09091e+05      6     10      5 DC----
              7  1.29600e+08  3.99087e+01  3.95311e+14  4.41673e+22  5.98813e+04  1.00000e+00  1.09091e+05      7     10      6 DC----
              8  1.51200e+08  3.99026e+01  4.18116e+14  5.31986e+22  6.98569e+04  1.00000e+00  1.09091e+05      8     10      7 DC----
"""

        # TRITON: power is over the interval
        triton_hist = core.Obiwan.get_history_from_f71("obiwan.exe", "perm000.f71", 10)
        mock_obiwan.assert_called_once()
        expected_hist = {
            "burndata": [
                {"power": 39.9302, "burn": 25.0},
                {"power": 39.9294, "burn": 225.0},
                {"power": 39.9271, "burn": 375.0},
                {"power": 39.9215, "burn": 312.5},
                {"power": 39.9155, "burn": 312.5},
                {"power": 39.9087, "burn": 250.0},
                {"power": 39.9026, "burn": 250.0},
            ],
            "initialhm": 1.0,
        }

        np.testing.assert_almost_equal(
            expected_hist["initialhm"], triton_hist["initialhm"]
        )

        assert len(expected_hist["burndata"]) == len(triton_hist["burndata"])
        for t_burndata, e_burndata in zip(
            triton_hist["burndata"], expected_hist["burndata"]
        ):
            np.testing.assert_almost_equal(t_burndata["power"], e_burndata["power"])
            np.testing.assert_almost_equal(t_burndata["burn"], e_burndata["burn"])

        # Note than in SCALE 7.0, TRITON (correctly) reports the initial
        # compositions power as zero; however, Polaris gives power at the end
        # of the timestep, so we shift the power history accordingly and verify
        # our history is still as-expected

        mock_obiwan.return_value = """

            pos         time        power         flux      fluence       energy    initialhm       volume libpos   case   step DCGNAB
            (-)          (s)         (MW)   (n/cm^2-s)     (n/cm^2)        (MWd)      (MTIHM)       (cm^3)    (-)    (-)    (-)    (-)
              1  0.00000e+00  3.99302e+01  2.77611e+14  0.00000e+00  0.00000e+00  1.00000e+00  1.09091e+05      1     10      0 DC----
              2  2.16000e+06  3.99294e+01  2.88762e+14  5.99639e+20  9.98255e+02  1.00000e+00  1.09091e+05      2     10      1 DC----
              3  2.16000e+07  3.99271e+01  3.13691e+14  6.21316e+21  9.98238e+03  1.00000e+00  1.09091e+05      3     10      2 DC----
              4  5.40000e+07  3.99215e+01  3.42857e+14  1.63767e+22  2.49551e+04  1.00000e+00  1.09091e+05      4     10      3 DC----
              5  8.10000e+07  3.99155e+01  3.70174e+14  2.56339e+22  3.74305e+04  1.00000e+00  1.09091e+05      5     10      4 DC----
              6  1.08000e+08  3.99087e+01  3.95311e+14  3.56286e+22  4.99041e+04  1.00000e+00  1.09091e+05      6     10      5 DC----
              7  1.29600e+08  3.99026e+01  4.18116e+14  4.41673e+22  5.98813e+04  1.00000e+00  1.09091e+05      7     10      6 DC----
              8  1.51200e+08  0.39026e-03  0.00000e+00  5.31986e+22  6.98569e+04  1.00000e+00  1.09091e+05      8     10      7 DC----
"""
        polaris_hist = core.Obiwan.get_history_from_f71(
            "obiwan.exe", "perm000.f71", 10, is_polaris=True
        )
        assert mock_obiwan.call_count == 2
        np.testing.assert_almost_equal(
            expected_hist["initialhm"], polaris_hist["initialhm"]
        )

        assert len(expected_hist["burndata"]) == len(polaris_hist["burndata"])
        for p_burndata, e_burndata in zip(
            polaris_hist["burndata"], expected_hist["burndata"]
        ):
            np.testing.assert_almost_equal(p_burndata["power"], e_burndata["power"])
            np.testing.assert_almost_equal(p_burndata["burn"], e_burndata["burn"])

    @patch("scale.olm.core.run_command")
    def test_obiwan_get_burnups_from_f71(self, mock_obiwan):
        """Calculate cumulative burnup from OBIWAN F71 info table energy data."""
        mock_obiwan.return_value = """

            pos         time        power         flux      fluence       energy    initialhm       volume libpos   case   step DCGNAB
            (-)          (s)         (MW)   (n/cm^2-s)     (n/cm^2)        (MWd)      (MTIHM)       (cm^3)    (-)    (-)    (-)    (-)
              1  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  2.00000e+00  1.09091e+05      1     10      0 DC----
              2  2.16000e+06  3.99302e+01  2.77611e+14  5.99639e+20  1.00000e+03  2.00000e+00  1.09091e+05      2     10      1 DC----
              3  2.16000e+07  3.99294e+01  2.88762e+14  6.21316e+21  1.00000e+04  2.00000e+00  1.09091e+05      3     10      2 DC----
              4  2.16000e+07  3.99294e+01  2.88762e+14  6.21316e+21  7.00000e+03  2.00000e+00  1.09091e+05      3     20      2 DC----
"""

        burnups = core.Obiwan.get_burnups_from_f71("obiwan.exe", "perm000.f71", 10)
        initialhm = core.Obiwan.get_initialhm_from_f71("obiwan.exe", "perm000.f71", 10)

        mock_obiwan.assert_any_call(
            "obiwan.exe view -format=info perm000.f71", echo=False
        )
        assert mock_obiwan.call_count == 2
        np.testing.assert_array_almost_equal(burnups, [0.0, 500.0, 5000.0])
        assert initialhm == 2.0

    @patch("scale.olm.core.run_command")
    def test_obiwan_get_burnups_from_f71_requires_initialhm(self, mock_obiwan):
        """Reject F71 info rows that cannot define MWd/MTIHM burnup."""
        mock_obiwan.return_value = """

            pos         time        power         flux      fluence       energy    initialhm libpos   case   step DCGNAB
            (-)          (s)         (MW)    (n/cm2-s)      (n/cm2)        (MWd)      (MTIHM)    (-)    (-)    (-)    (-)
              1  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00  0.00000e+00      1     10      0 DC----
"""

        with pytest.raises(ValueError, match="initialhm=0.0"):
            core.Obiwan.get_burnups_from_f71("obiwan.exe", "perm000.f71", 10)
        with pytest.raises(ValueError, match="initialhm=0.0"):
            core.Obiwan.get_initialhm_from_f71("obiwan.exe", "perm000.f71", 10)


class TestScaleOutfile:
    """Test the ScaleOutfile class for SCALE output parsing."""

    def test_parse_burnups_from_triton_output(self):
        """Test parsing burnup data from TRITON output using real file."""
        # Create realistic TRITON output file
        sample_output = """
Some header text...
Sub-Interval   Depletion   Sub-interval    Specific      Burn Length  Decay Length   Library Burnup
     No.       Interval     in interval  Power(MW/MTIHM)     (d)          (d)           (MWd/MTIHM)
----------------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------------
        0     ****Initial Bootstrap Calculation****                                      0.00000E+00
        1          1                1          40.000      25.000         0.000          5.00000e+02
        2          1                2          40.000     300.000         0.000          7.00000e+03
        3          1                3          40.000     300.000         0.000          1.90000e+04
        4          1                4          40.000     312.500         0.000          3.12500e+04
----------------------------------------------------------------------------------------------------
Some footer text...
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".out") as f:
            f.write(sample_output)
            temp_path = f.name

        try:
            burnups = core.ScaleOutfile.parse_burnups_from_triton_output(temp_path)
            rows = core.ScaleOutfile.parse_triton_library_table(temp_path)

            expected = [0.0, 500.0, 7000.0, 19000.0, 31250.0]
            assert len(burnups) == 5
            np.testing.assert_array_almost_equal(burnups, expected)
            assert rows == [
                {"power": 40.000, "burn": 25.000, "burnup": 500.0},
                {"power": 40.000, "burn": 300.000, "burnup": 7000.0},
                {"power": 40.000, "burn": 300.000, "burnup": 19000.0},
                {"power": 40.000, "burn": 312.500, "burnup": 31250.0},
            ]

        finally:
            os.unlink(temp_path)

    def test_parse_polaris_state_table_returns_requested_material_case(self, tmp_path):
        """Parse requested material-class cases from a Polaris output table."""
        output_file = tmp_path / "polaris.out"
        output_file.write_text(
            """
header
Integrated edits for each material class
| 7 | other | FUEL
| 9 | other | BASIS
"""
        )

        assert core.ScaleOutfile.parse_polaris_state_table(output_file) == 7
        assert core.ScaleOutfile.parse_polaris_state_table(output_file, "FUEL") == 7
        assert core.ScaleOutfile.parse_polaris_state_table(output_file, "BASIS") == 9

    def test_parse_polaris_state_table_defaults_to_fuel_case(self, tmp_path):
        """Default to the Polaris FUEL material-class case."""
        output_file = tmp_path / "polaris.out"
        output_file.write_text(
            """
header
Integrated edits for each material class
| 7 | other | FUEL
"""
        )

        assert core.ScaleOutfile.parse_polaris_state_table(output_file) == 7

    def test_get_runtime(self):
        """Test extracting runtime from SCALE output using real file."""
        sample_output = """
Some output text...
t-depl finished. used 35.2481 seconds.
More output text...
"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".out") as f:
            f.write(sample_output)
            temp_path = f.name

        try:
            runtime = core.ScaleOutfile.get_runtime(temp_path)
            assert runtime == pytest.approx(35.2481, abs=0.001)

        finally:
            os.unlink(temp_path)


class TestReactorLibraryUtilities:
    """Test ReactorLibrary utility functions with minimal mocking."""

    def test_duplicate_degenerate_axis_value(self):
        """Test degenerate axis value duplication (comprehensive mathematical testing)."""
        test_cases = [
            # (input, expected_delta)
            (0.0, 0.05),  # Zero case
            (0.723, 0.05),  # Typical reactor parameter
            (-1.0, 0.05),  # Negative value
            (100.0, 5.0),  # Large value (5% of 100)
            (1e-12, 0.05),  # Very small value
            (-50.0, 2.5),  # Large negative (5% of 50)
            (2.0, 0.1),  # Moderate value (5% of 2)
        ]

        for x0, expected_delta in test_cases:
            x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)
            actual_delta = x1 - x0
            assert actual_delta == pytest.approx(expected_delta, abs=1e-10)

            # Verify essential properties
            assert x1 > x0, f"x1 ({x1}) should be greater than x0 ({x0})"
            assert x1 != x0, f"x1 ({x1}) should be different from x0 ({x0})"
            assert np.isfinite(x1), f"x1 ({x1}) should be finite"

    def test_get_indices(self):
        """Test index calculation for library interpolation."""
        axes_names = np.array(["mod_dens", "enrichment", "burnup"])
        axes_values = [
            np.array([0.1, 0.5, 0.9]),  # mod_dens
            np.array([2.0, 3.5, 5.0]),  # enrichment
            np.array([0, 1000, 5000]),  # burnup
        ]

        # Test exact matches
        point_data = {"mod_dens": 0.5, "enrichment": 3.5, "burnup": 1000}
        indices = core.ReactorLibrary.get_indices(axes_names, axes_values, point_data)
        expected = (1, 1, 1)  # Middle values
        assert indices == expected

    def test_duplicate_degenerate_axis_value_advanced(self):
        """Test degenerate axis value duplication (comprehensive mathematical testing)."""
        test_cases = [
            # (input, expected_delta)
            (0.0, 0.05),  # Zero case
            (0.723, 0.05),  # Typical reactor parameter
            (-1.0, 0.05),  # Negative value
            (100.0, 5.0),  # Large value (5% of 100)
            (1e-12, 0.05),  # Very small value
            (-50.0, 2.5),  # Large negative (5% of 50)
            (2.0, 0.1),  # Moderate value (5% of 2)
        ]

        for x0, expected_delta in test_cases:
            x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)
            actual_delta = x1 - x0
            assert actual_delta == pytest.approx(expected_delta, abs=1e-10)

            # Verify essential properties
            assert x1 > x0, f"x1 ({x1}) should be greater than x0 ({x0})"
            assert x1 != x0, f"x1 ({x1}) should be different from x0 ({x0})"
            assert np.isfinite(x1), f"x1 ({x1}) should be finite"


class TestArpInfo:
    """Test ARPDATA parsing."""

    def test_init_block_classifies_mox_from_header_not_name_prefix(self):
        """A TRITON MOX library name does not have to start with mox_."""
        block = """
2 3 1 2 6
4.0
10.0
50.0
60.0
70.0
1
0.65
0.75
'triton_mox_pin_quick_e0400v5000w0650.h5'
'triton_mox_pin_quick_e1000v5000w0650.h5'
'triton_mox_pin_quick_e0400v6000w0650.h5'
'triton_mox_pin_quick_e1000v6000w0650.h5'
'triton_mox_pin_quick_e0400v7000w0650.h5'
'triton_mox_pin_quick_e1000v7000w0650.h5'
'triton_mox_pin_quick_e0400v5000w0750.h5'
'triton_mox_pin_quick_e1000v5000w0750.h5'
'triton_mox_pin_quick_e0400v6000w0750.h5'
'triton_mox_pin_quick_e1000v6000w0750.h5'
'triton_mox_pin_quick_e0400v7000w0750.h5'
'triton_mox_pin_quick_e1000v7000w0750.h5'
0.0
500.0
7000.0
19000.0
31250.0
43750.0
"""
        arpinfo = core.ArpInfo()

        arpinfo.init_block("triton_mox_pin_quick", block)

        assert arpinfo.fuel_type == "MOX"
        assert arpinfo.pu_frac_list == [4.0, 10.0]
        assert arpinfo.pu239_frac_list == [50.0, 60.0, 70.0]
        assert arpinfo.mod_dens_list == [0.65, 0.75]
        assert len(arpinfo.lib_list) == 12
        assert arpinfo.burnup_list == [
            0.0,
            500.0,
            7000.0,
            19000.0,
            31250.0,
            43750.0,
        ]


class TestNuclideInventory:
    """Test the NuclideInventory class using real data structures."""

    @pytest.fixture
    def sample_composition_manager(self):
        """Create a real composition manager for testing."""
        data = {
            "0092235": {
                "mass": 235.044,
                "atomicNumber": 92,
                "element": "U",
                "massNumber": 235,
            },
            "0092238": {
                "mass": 238.051,
                "atomicNumber": 92,
                "element": "U",
                "massNumber": 238,
            },
            "0094239": {
                "mass": 239.052,
                "atomicNumber": 94,
                "element": "Pu",
                "massNumber": 239,
            },
        }
        return core.CompositionManager(data)

    @pytest.fixture
    def sample_inventory(self, sample_composition_manager):
        """Create a real NuclideInventory for testing."""
        time = np.array([0, 100, 200, 300])  # days
        nuclide_amount = {
            "0092235": np.array([1000, 950, 900, 850]),  # moles
            "0092238": np.array([100, 105, 110, 115]),  # moles
            "0094239": np.array([0, 5, 15, 30]),  # moles
        }
        return core.NuclideInventory(sample_composition_manager, time, nuclide_amount)

    def test_get_hm_mass(self, sample_inventory):
        """Test heavy metal mass calculation."""
        hm_mass = sample_inventory.get_hm_mass(min_z=92)

        # Should be positive and have correct length
        assert len(hm_mass) == 4
        assert np.all(hm_mass > 0)

        # Mass should change over time due to transmutation
        assert not np.allclose(hm_mass, hm_mass[0])

    def test_get_amount(self, sample_inventory):
        """Test nuclide amount extraction."""
        # Test moles (default)
        u235_moles = sample_inventory.get_amount("u235", units="MOLES")
        expected = np.array([1000, 950, 900, 850])
        np.testing.assert_array_equal(u235_moles, expected)

        # Test grams
        u235_grams = sample_inventory.get_amount("u235", units="GRAMS")
        expected_grams = expected * 235.044  # moles * mass
        np.testing.assert_array_almost_equal(u235_grams, expected_grams)


class TestMathematicalAlgorithms:
    """Test mathematical algorithms with focus on correctness, not implementation."""

    def test_axis_duplication_mathematical_properties(self):
        """Test mathematical properties of axis duplication algorithm."""
        # Test over wide range of realistic reactor parameters
        test_values = [
            0.0,
            0.1,
            0.5,
            0.723,
            1.0,
            2.0,
            5.0,
            10.0,
            50.0,
            100.0,
            -0.1,
            -1.0,
            -10.0,
            1e-10,
            1e-5,
            1e5,
        ]

        for x0 in test_values:
            x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)

            # Essential mathematical properties
            assert x1 > x0, f"Failed monotonicity: {x1} <= {x0}"
            assert x1 != x0, f"Failed distinctness: {x1} == {x0}"
            assert np.isfinite(x1), f"Failed finiteness: {x1} is not finite"

            # Test numerical stability
            axis = np.array([x0, x1])
            gradient = np.gradient(axis)
            assert np.all(gradient > 0), f"Failed gradient positivity for {x0}"
            assert np.all(np.isfinite(gradient)), f"Failed gradient finiteness for {x0}"

    def test_composition_normalization_properties(self):
        """Test mathematical properties of composition normalization."""
        # Test various composition scenarios
        test_compositions = [
            {"u235": 25, "u238": 75},  # Simple uranium
            {"u235": 20, "u238": 70, "pu239": 10},  # U-Pu mixture
            {"pu239": 50, "pu241": 30, "am241": 20},  # TRU mixture
            {"u235": 1, "u238": 1, "pu239": 1},  # Equal parts
        ]

        for comp in test_compositions:
            # Test renormalization to 100%
            norm_comp, norm_factor = core.CompositionManager.renormalize_wtpt(
                comp, 100.0
            )

            # Mathematical properties
            total = sum(norm_comp.values())
            assert total == pytest.approx(
                100.0, abs=1e-10
            ), f"Failed normalization: {total}"
            assert (
                norm_factor > 0
            ), f"Normalization factor should be positive: {norm_factor}"

    def test_molar_mass_calculation_properties(self):
        """Test mathematical properties of molar mass calculations."""
        # Test harmonic mean formula: 1/m = sum(w_i / m_i)
        test_cases = [
            ({"u235": 50, "u238": 50}, {}),  # Equal mixture
            ({"pu239": 100}, {}),  # Pure isotope
            ({"u235": 25, "u238": 75}, {}),  # Enriched uranium
        ]

        for iso_wts, m_data in test_cases:
            molar_mass = core.CompositionManager.grams_per_mol(iso_wts, m_data)

            # Mathematical properties
            assert molar_mass > 0, f"Molar mass should be positive: {molar_mass}"
            assert np.isfinite(molar_mass), f"Molar mass should be finite: {molar_mass}"

            # For single isotope, should equal mass number (approximately)
            if len(iso_wts) == 1:
                isotope = list(iso_wts.keys())[0]
                # Extract mass number correctly using regex
                import re

                mass_str = re.sub("^[a-z]+", "", isotope)  # Remove element letters
                mass_str = re.sub(
                    "m[0-9]*$", "", mass_str
                )  # Remove metastable indicators
                mass_number = float(mass_str)
                assert molar_mass == pytest.approx(mass_number, rel=0.01)
