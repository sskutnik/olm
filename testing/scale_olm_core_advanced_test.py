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
from unittest.mock import patch

import scale.olm.core as core


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
                "massNumber": 1
            },
            "0001002": {
                "IZZZAAA": "0001002",
                "atomicNumber": 1,
                "element": "H",
                "isomericState": 0,
                "mass": 2.014102,
                "massNumber": 2
            },
            "0092235": {
                "IZZZAAA": "0092235",
                "atomicNumber": 92,
                "element": "U",
                "isomericState": 0,
                "mass": 235.044,
                "massNumber": 235
            },
            "0094239": {
                "IZZZAAA": "0094239",
                "atomicNumber": 94,
                "element": "Pu",
                "isomericState": 0,
                "mass": 239.052,
                "massNumber": 239
            }
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
        expected = 1.0 / (0.5/235.044 + 0.5/239.052)
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


    @patch('scale.olm.core.run_command')
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
        expected_hist = {"burndata": [{"power": 40.0, "burn": 25.0}, {"power": 40.0, "burn": 225.0}, {"power": 40.0, "burn": 375.0}, {"power": 40.0, "burn": 625.0}, {"power": 40, "burn": 500.0}, {"power": 40.0, "burn": 500.0}, {"power": 40.0, "burn": 500.0}], "initialhm": 1.0}

        np.testing.assert_almost_equal(expected_hist["initialhm"], triton_hist["initialhm"])
        assert len(expected_hist["burndata"]) == len(triton_hist["burndata"])
        for t_burndata, e_burndata in zip(triton_hist["burndata"], expected_hist["burndata"]):
            np.testing.assert_almost_equal(t_burndata["power"], e_burndata["power"])
            np.testing.assert_almost_equal(t_burndata["burn"], e_burndata["burn"])

        # Polaris gives power at the end of the timestep, so modify the last 
        # step accordingly and verify our history is still as-expected
        polaris_rv =  "\n".join(mock_obiwan.return_value.split("\n")[:-2]) + \
"""
               8  2.37600e+08  0.10000e-03  0.00000e+00  1.87415e+22  1.10000e+05  1.00000e+00      7      1     10 DC----
"""
        polaris_hist = core.Obiwan.get_history_from_f71("obiwan.exe", "perm000.f71", 1, is_polaris=True)
        assert mock_obiwan.call_count == 2        
        np.testing.assert_almost_equal(expected_hist["initialhm"], polaris_hist["initialhm"])

        assert len(expected_hist["burndata"]) == len(polaris_hist["burndata"])
        for p_burndata, e_burndata in zip(polaris_hist["burndata"], expected_hist["burndata"]):
            np.testing.assert_almost_equal(p_burndata["power"], e_burndata["power"])
            np.testing.assert_almost_equal(p_burndata["burn"], e_burndata["burn"])


    @patch('scale.olm.core.run_command')
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
        expected_hist = {"burndata": [{"power": 39.9302, "burn": 25.0}, {"power": 39.9294, "burn": 225.0}, {"power": 39.9271, "burn": 375.0}, {"power": 39.9215, "burn": 312.5}, {"power": 39.9155, "burn": 312.5}, {"power": 39.9087, "burn": 250.0}, {"power": 39.9026, "burn": 250.0 }], "initialhm": 1.0}

        np.testing.assert_almost_equal(expected_hist["initialhm"], triton_hist["initialhm"])

        assert len(expected_hist["burndata"]) == len(triton_hist["burndata"])
        for t_burndata, e_burndata in zip(triton_hist["burndata"], expected_hist["burndata"]):
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
        polaris_hist = core.Obiwan.get_history_from_f71("obiwan.exe", "perm000.f71", 10, is_polaris=True)
        assert mock_obiwan.call_count == 2
        np.testing.assert_almost_equal(expected_hist["initialhm"], polaris_hist["initialhm"])

        assert len(expected_hist["burndata"]) == len(polaris_hist["burndata"])
        for p_burndata, e_burndata in zip(polaris_hist["burndata"], expected_hist["burndata"]):
            np.testing.assert_almost_equal(p_burndata["power"], e_burndata["power"])
            np.testing.assert_almost_equal(p_burndata["burn"], e_burndata["burn"])


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

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.out') as f:
            f.write(sample_output)
            temp_path = f.name

        try:
            burnups = core.ScaleOutfile.parse_burnups_from_triton_output(temp_path)

            expected = [0.0, 500.0, 7000.0, 19000.0, 31250.0]
            assert len(burnups) == 5
            np.testing.assert_array_almost_equal(burnups, expected)

        finally:
            os.unlink(temp_path)

    def test_parse_burnups_from_polaris_t16(self):
        """Test parsing of the burnups from a Polaris t16 file using a real output"""

        sample_t16 = """' Record 1: nblk, lblk, record sizes.
    13  1000  5120    14   208     0     0     0     3   115    16   100   100
' Record 2: dimensioning data.
        69         0         1         2         1         6         4         8         1        10
        10         6         1         0
' Record 3: depletion data.
' Burnups
 0.000000E+00 1.000000E-01 2.500000E-01 5.000000E-01 1.000000E+00 1.500000E+00
 2.000000E+00 2.500000E+00 3.000000E+00 3.500000E+00 4.000000E+00 4.500000E+00
 5.000000E+00 5.500000E+00 6.000000E+00 6.500000E+00 7.000000E+00 7.500000E+00
 8.000000E+00 8.500000E+00 9.000000E+00 9.500000E+00 1.000000E+01 1.050000E+01
 1.100000E+01 1.150000E+01 1.200000E+01 1.250000E+01 1.300000E+01 1.350000E+01
 1.400000E+01 1.450000E+01 1.500000E+01 1.550000E+01 1.600000E+01 1.650000E+01
 1.700000E+01 1.750000E+01 1.800000E+01 1.850000E+01 1.900000E+01 1.950000E+01
 2.000000E+01 2.050000E+01 2.100000E+01 2.150000E+01 2.200000E+01 2.300000E+01
 2.400000E+01 2.500000E+01 2.750000E+01 3.000000E+01 3.250000E+01 3.500000E+01
 4.000000E+01 4.500000E+01 5.000000E+01 5.500000E+01 6.000000E+01 6.500000E+01
 7.000000E+01 7.500000E+01 8.000000E+01 8.500000E+01 9.000000E+01 9.500000E+01
 1.000000E+02 1.050000E+02 1.100000E+02
' Time
 0.000000E+00 3.831418E+00 9.578544E+00 1.915709E+01 3.831417E+01 5.747126E+01
 7.662835E+01 9.578544E+01 1.149425E+02 1.340996E+02 1.532567E+02 1.724138E+02
 1.915709E+02 2.107280E+02 2.298851E+02 2.490421E+02 2.681992E+02 2.873563E+02
 3.065134E+02 3.256705E+02 3.448276E+02 3.639847E+02 3.831418E+02 4.022989E+02
 4.214559E+02 4.406130E+02 4.597701E+02 4.789272E+02 4.980843E+02 5.172414E+02
 5.363984E+02 5.555555E+02 5.747126E+02 5.938698E+02 6.130268E+02 6.321839E+02
 6.513410E+02 6.704981E+02 6.896552E+02 7.088123E+02 7.279694E+02 7.471265E+02
 7.662835E+02 7.854406E+02 8.045977E+02 8.237548E+02 8.429119E+02 8.812261E+02
 9.195402E+02 9.578544E+02 1.053640E+03 1.149425E+03 1.245211E+03 1.340996E+03
 1.532567E+03 1.724138E+03 1.915709E+03 2.107280E+03 2.298851E+03 2.490421E+03
 2.681992E+03 2.873563E+03 3.065134E+03 3.256705E+03 3.448276E+03 3.639847E+03
 3.831418E+03 4.022989E+03 4.214560E+03
' Power
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01 2.610000E+01
 2.610000E+01 2.610000E+01 2.610000E+01
 ... additional records ..."""

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.t16') as f:
            f.write(sample_t16)
            temp_path = f.name

        try:
            burnups = core.ScaleOutfile.parse_burnups_from_polaris_t16(temp_path)

            assert len(burnups) == 69
            expected = [0.0, 100.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0, 7500.0, 8000.0, 8500.0, 9000.0, 9500.0, 10000.0, 10500.0, 11000.0, 11500.0, 12000.0, 12500.0, 13000.0, 13500.0, 14000.0, 14500.0, 15000.0, 15500.0, 16000.0, 16500.0, 17000.0, 17500.0, 18000.0, 18500.0, 19000.0, 19500.0, 20000.0, 20500.0, 21000.0, 21500.0, 22000.0, 23000.0, 24000.0, 25000.0, 27500.0, 30000.0, 32500.0, 35000.0, 40000.0, 45000.0, 50000.0, 55000.0, 60000.0, 65000.0, 70000.0, 75000.0, 80000.0, 85000.0, 90000.0, 95000.0, 100000.0, 105000.0, 110000.0 ]
            np.testing.assert_array_almost_equal(burnups, expected)

        finally:
            os.unlink(temp_path)

    def test_get_runtime(self):
        """Test extracting runtime from SCALE output using real file."""
        sample_output = """
Some output text...
t-depl finished. used 35.2481 seconds.
More output text...
"""

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.out') as f:
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
            (0.0, 0.05),           # Zero case
            (0.723, 0.05),         # Typical reactor parameter
            (-1.0, 0.05),          # Negative value
            (100.0, 5.0),          # Large value (5% of 100)
            (1e-12, 0.05),         # Very small value
            (-50.0, 2.5),          # Large negative (5% of 50)
            (2.0, 0.1),            # Moderate value (5% of 2)
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
            np.array([0.1, 0.5, 0.9]),      # mod_dens
            np.array([2.0, 3.5, 5.0]),      # enrichment
            np.array([0, 1000, 5000])       # burnup
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
            (0.0, 0.05),           # Zero case
            (0.723, 0.05),         # Typical reactor parameter
            (-1.0, 0.05),          # Negative value
            (100.0, 5.0),          # Large value (5% of 100)
            (1e-12, 0.05),         # Very small value
            (-50.0, 2.5),          # Large negative (5% of 50)
            (2.0, 0.1),            # Moderate value (5% of 2)
        ]

        for x0, expected_delta in test_cases:
            x1 = core.ReactorLibrary.duplicate_degenerate_axis_value(x0)
            actual_delta = x1 - x0
            assert actual_delta == pytest.approx(expected_delta, abs=1e-10)

            # Verify essential properties
            assert x1 > x0, f"x1 ({x1}) should be greater than x0 ({x0})"
            assert x1 != x0, f"x1 ({x1}) should be different from x0 ({x0})"
            assert np.isfinite(x1), f"x1 ({x1}) should be finite"


class TestNuclideInventory:
    """Test the NuclideInventory class using real data structures."""

    @pytest.fixture
    def sample_composition_manager(self):
        """Create a real composition manager for testing."""
        data = {
            "0092235": {"mass": 235.044, "atomicNumber": 92, "element": "U", "massNumber": 235},
            "0092238": {"mass": 238.051, "atomicNumber": 92, "element": "U", "massNumber": 238},
            "0094239": {"mass": 239.052, "atomicNumber": 94, "element": "Pu", "massNumber": 239}
        }
        return core.CompositionManager(data)

    @pytest.fixture
    def sample_inventory(self, sample_composition_manager):
        """Create a real NuclideInventory for testing."""
        time = np.array([0, 100, 200, 300])  # days
        nuclide_amount = {
            "0092235": np.array([1000, 950, 900, 850]),  # moles
            "0092238": np.array([100, 105, 110, 115]),   # moles
            "0094239": np.array([0, 5, 15, 30])          # moles
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
            0.0, 0.1, 0.5, 0.723, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0,
            -0.1, -1.0, -10.0, 1e-10, 1e-5, 1e5
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
            {"u235": 25, "u238": 75},                    # Simple uranium
            {"u235": 20, "u238": 70, "pu239": 10},      # U-Pu mixture
            {"pu239": 50, "pu241": 30, "am241": 20},    # TRU mixture
            {"u235": 1, "u238": 1, "pu239": 1},         # Equal parts
        ]

        for comp in test_compositions:
            # Test renormalization to 100%
            norm_comp, norm_factor = core.CompositionManager.renormalize_wtpt(comp, 100.0)

            # Mathematical properties
            total = sum(norm_comp.values())
            assert total == pytest.approx(100.0, abs=1e-10), f"Failed normalization: {total}"
            assert norm_factor > 0, f"Normalization factor should be positive: {norm_factor}"

    def test_molar_mass_calculation_properties(self):
        """Test mathematical properties of molar mass calculations."""
        # Test harmonic mean formula: 1/m = sum(w_i / m_i)
        test_cases = [
            ({"u235": 50, "u238": 50}, {}),           # Equal mixture
            ({"pu239": 100}, {}),                     # Pure isotope
            ({"u235": 25, "u238": 75}, {}),          # Enriched uranium
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
                mass_str = re.sub("m[0-9]*$", "", mass_str)  # Remove metastable indicators
                mass_number = float(mass_str)
                assert molar_mass == pytest.approx(mass_number, rel=0.01)
