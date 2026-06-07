import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "tools" / "post_example_validation_comment.py"
)
SPEC = importlib.util.spec_from_file_location("post_example_validation_comment", SCRIPT)
post_example_validation_comment = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = post_example_validation_comment
SPEC.loader.exec_module(post_example_validation_comment)


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _write_polaris_example(
    example_dir,
    *,
    artifact_contract="Polaris",
    scale_version="6.3.3",
    model_name="polaris_uoxgd_quick",
    model_description=None,
    fuel_type="UOX",
    static=None,
):
    work_dir = example_dir / "_work"
    perm_dir = work_dir / "perms" / "abc123"
    arplib_dir = work_dir / "arplibs"

    if model_description is None:
        model_description = (
            "Generic BWR 7x7 Polaris lattice with Gd2O3 pins"
            if model_name == "polaris_uoxgd_quick"
            else ""
        )

    config = {
        "model": {"name": model_name, "description": model_description},
        "assemble": {"fuel_type": fuel_type},
    }
    if static is not None:
        config["generate"] = {"static": static}
    if artifact_contract == "Polaris" and model_name == "polaris_uoxgd_quick":
        config["model"]["notes"] = [
            "Includes four Gd2O3-bearing fuel pins and no Cr2O3 dopant."
        ]
        config.setdefault("generate", {})["static"] = {
            "gd2o3_pin_wtpt": 3.0,
            "gd2o3_pin_count": 4,
        }
    _write_json(example_dir / "config.olm.json", config)
    _write_json(
        work_dir / "generate.olm.json",
        {
            "perms": [
                {
                    "input_file": "perms/abc123/model.inp",
                    "_scale": {
                        "sequences": ["polaris"],
                        "artifact_contract": artifact_contract,
                    },
                }
            ]
        },
    )
    _write_json(
        work_dir / "run.olm.json",
        {
            "version": scale_version,
            "command_line": "cd _work/perms && make -j 1",
            "runs": [{"success": True}],
        },
    )
    _write_json(
        work_dir / "assemble.olm.json",
        {
            "archive_file": f"arpdata.txt:{model_name}",
            "points": [
                {
                    "files": {
                        "origin": {"f71": "perms/abc123/model.f71"},
                        "ii_json": "arplibs/model.ii.json",
                    },
                    "_": {
                        "perm": {
                            "_scale": {
                                "sequences": ["polaris"],
                                "artifact_contract": artifact_contract,
                            }
                        }
                    },
                    "_arpinfo": {"burnup_list": [0.0, 500.0, 998.154]},
                }
            ],
        },
    )
    _write_json(
        arplib_dir / "model.ii.json",
        {"responses": {"system": {"amount": []}}},
    )
    _write_json(
        work_dir / "check.olm.json",
        {
            "test_pass": True,
            "sequence": [
                {
                    "test_pass": True,
                    "name": "LowOrderConsistency",
                    "metric": "grams_per_initial_hm",
                    "units": "g/gIHM",
                    "q1": 0.9913565505804312,
                    "q2": 0.9825393864013268,
                    "target_q1": 0.7,
                    "target_q2": 0.95,
                    "wr": 2085,
                    "wa": 341,
                    "m": 241200,
                    "mean_rel_diff": 0.000342195279555865,
                    "hist_image": str(work_dir / "check" / "loc" / "hist.png"),
                    "atom_fraction": {
                        "test_pass": True,
                        "units": "atom fraction",
                        "q1": 0.7903565505804312,
                        "q2": 0.9725393864013268,
                        "target_q1": 0.7,
                        "target_q2": 0.95,
                        "wr": 50566,
                        "wa": 1741,
                        "m": 241200,
                        "mean_rel_diff": 0.006342195279555865,
                        "hist_image": str(
                            work_dir / "check" / "loc" / "atom_fraction" / "hist.png"
                        ),
                    },
                }
            ],
        },
    )
    perm_dir.mkdir(parents=True, exist_ok=True)
    (perm_dir / "model.out").write_text(
        "\n".join(
            [
                "header",
                "Integrated edits for each material class",
                "|  26 |  85 |               FUEL | 0.000e+00 | 0.000e+00 |",
                "|  29 |  85 |              BASIS | 0.000e+00 | 0.000e+00 |",
            ]
        )
    )
    (perm_dir / "model.f71").write_text("")


def test_collect_example_result_reads_completed_polaris_artifacts(tmp_path):
    example_dir = tmp_path / "polaris_uoxgd_quick"
    _write_polaris_example(example_dir)

    result = post_example_validation_comment.collect_example_result(example_dir)

    assert result.passed is True
    assert result.example == "polaris_uoxgd_quick"
    assert result.model_name == "polaris_uoxgd_quick"
    assert result.case_label == "BWR UOX+Gd2O3 assembly"
    assert result.fuel_type == "UOX"
    assert result.scale_version == "6.3.3"
    assert result.successful_runs == 1
    assert result.total_runs == 1
    assert result.point_count == 1
    assert result.contracts == ("Polaris",)
    assert result.burnup_lists == ((0.0, 500.0, 998.154),)
    assert result.system_json_normalized is True
    assert len(result.low_order_consistency) == 1
    loc = result.low_order_consistency[0]
    assert loc.passed is True
    assert loc.metric == "grams_per_initial_hm"
    assert loc.units == "g/gIHM"
    assert loc.q1 == 0.9913565505804312
    assert loc.q2 == 0.9825393864013268
    assert loc.target_q1 == 0.7
    assert loc.target_q2 == 0.95
    assert loc.relative_failures == 2085
    assert loc.absolute_relative_failures == 341
    assert loc.total_comparisons == 241200
    assert loc.mean_relative_difference == 0.000342195279555865
    assert loc.histogram_path == "check/loc/hist.png"
    assert loc.atom_fraction is not None
    assert loc.atom_fraction.q1 == 0.7903565505804312
    assert loc.atom_fraction.histogram_path == "check/loc/atom_fraction/hist.png"
    assert loc.grams_per_initial_hm is None


def test_collect_example_result_labels_dopant_pin_and_assembly_cases(tmp_path):
    assembly_dir = tmp_path / "polaris_bwr_gd"
    gd_pin_dir = tmp_path / "polaris_uox_gd"
    gd_cr_pin_dir = tmp_path / "polaris_uox_gd_cr"
    _write_polaris_example(assembly_dir)
    _write_polaris_example(
        gd_pin_dir,
        model_name="polaris_uoxgd_pin_quick",
        model_description="A 2D Polaris BWR UOX+Gd2O3 pin cell.",
        static={"gd2o3_wtpt": 3.0},
    )
    _write_polaris_example(
        gd_cr_pin_dir,
        model_name="polaris_uoxgdcr_pin_quick",
        model_description="A 2D Polaris BWR UOX+Gd2O3+Cr2O3 pin cell.",
        static={"gd2o3_wtpt": 3.0, "cr2o3_ppm": 3000.0},
    )

    assembly = post_example_validation_comment.collect_example_result(assembly_dir)
    gd_pin = post_example_validation_comment.collect_example_result(gd_pin_dir)
    gd_cr_pin = post_example_validation_comment.collect_example_result(gd_cr_pin_dir)

    assert assembly.case_label == "BWR UOX+Gd2O3 assembly"
    assert gd_pin.case_label == "BWR UOX+Gd2O3 pin cell"
    assert gd_cr_pin.case_label == "BWR UOX+Gd2O3+Cr2O3 pin cell"


def test_collect_example_result_reads_atom_fraction_primary_metric(tmp_path):
    example_dir = tmp_path / "polaris_uoxgd_quick"
    _write_polaris_example(example_dir)
    check_path = example_dir / "_work" / "check.olm.json"
    check_data = json.loads(check_path.read_text())
    loc = check_data["sequence"][0]
    grams = {
        "test_pass": loc["test_pass"],
        "units": loc["units"],
        "q1": loc["q1"],
        "q2": loc["q2"],
        "target_q1": loc["target_q1"],
        "target_q2": loc["target_q2"],
        "wr": loc["wr"],
        "wa": loc["wa"],
        "m": loc["m"],
        "mean_rel_diff": loc["mean_rel_diff"],
        "hist_image": loc["hist_image"],
    }
    atom = loc.pop("atom_fraction")
    loc.update(atom)
    loc["metric"] = "atom_fraction"
    loc["grams_per_initial_hm"] = grams
    _write_json(check_path, check_data)

    result = post_example_validation_comment.collect_example_result(example_dir)

    loc = result.low_order_consistency[0]
    assert loc.metric == "atom_fraction"
    assert loc.atom_fraction is None
    assert loc.grams_per_initial_hm is not None
    assert loc.grams_per_initial_hm.q1 == 0.9913565505804312
    assert loc.grams_per_initial_hm.histogram_path == "check/loc/hist.png"

    comment = post_example_validation_comment.build_comment(
        [result],
        commit="abc123",
        generated_at="2026-06-06T00:00:00+00:00",
    )

    assert "`atom fraction`" not in comment
    assert "| SCALE version | code | case | example | q1 | q2 | Pass |" in comment
    assert (
        "| `6.3.*` | `Polaris` | `BWR UOX+Gd2O3 assembly` | "
        "`examples/polaris_uoxgd_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert "0.790357" not in comment
    assert "`check/loc/hist.png`" not in comment
    assert "`check/loc/atom_fraction/hist.png`" not in comment


def test_build_comment_includes_standard_marker_and_validation_table(tmp_path):
    example_dir = tmp_path / "polaris_uoxgd_quick"
    _write_polaris_example(example_dir)
    result = post_example_validation_comment.collect_example_result(example_dir)

    comment = post_example_validation_comment.build_comment(
        [result],
        commit="abc123",
        generated_at="2026-06-06T00:00:00+00:00",
    )

    assert "<!-- olm-example-validation -->" in comment
    assert "### OLM Low-Order Consistency Results" in comment
    assert "`abc123`" in comment
    assert "`yes`" in comment
    assert "`PASS`" not in comment
    assert "#### SCALE 6.3" not in comment
    assert "##### UOX" not in comment
    assert "`Polaris`" in comment
    assert "Fuel cases" not in comment
    assert "Low-order consistency (`g/gIHM`)" in comment
    assert "| SCALE version | code | case | example | q1 | q2 | Pass |" in comment
    assert (
        "| `6.3.*` | `Polaris` | `BWR UOX+Gd2O3 assembly` | "
        "`examples/polaris_uoxgd_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert "2085" not in comment
    assert "341" not in comment
    assert "`atom fraction`" not in comment
    assert "0.790357" not in comment
    assert "`check/loc/hist.png`" not in comment
    assert "`check/loc/atom_fraction/hist.png`" not in comment
    assert "`[0, 500, 998.154]`" not in comment
    assert "`system only`" not in comment
    assert "cd _work/perms && make -j 1" not in comment


def test_build_grouped_comment_includes_result_set_headings(tmp_path):
    scale63_dir = tmp_path / "scale63_polaris_uoxgd_quick"
    scale63_pin_dir = tmp_path / "scale63_polaris_uox_pin_quick"
    scale70_dir = tmp_path / "scale70_polaris_uoxgd_quick"
    scale70_pin_dir = tmp_path / "scale70_polaris_uox_pin_quick"
    triton_dir = tmp_path / "scale63_triton_uox_pin_quick"
    scale70_gd_pin_dir = tmp_path / "scale70_polaris_uoxgd_pin_quick"
    scale70_gd_cr_pin_dir = tmp_path / "scale70_polaris_uoxgdcr_pin_quick"
    scale70_mox_pin_dir = tmp_path / "scale70_polaris_mox_pin_quick"
    scale70_triton_mox_dir = tmp_path / "scale70_triton_mox_pin_quick"
    _write_polaris_example(scale63_dir)
    _write_polaris_example(scale63_pin_dir, model_name="polaris_uox_pin_quick")
    _write_polaris_example(scale70_dir, scale_version="7.0.b12")
    _write_polaris_example(
        scale70_pin_dir, scale_version="7.0.b12", model_name="polaris_uox_pin_quick"
    )
    _write_polaris_example(
        triton_dir, artifact_contract="TRITON", model_name="triton_uox_pin_quick"
    )
    _write_polaris_example(
        scale70_gd_pin_dir,
        scale_version="7.0.b12",
        model_name="polaris_uoxgd_pin_quick",
        model_description="A 2D Polaris BWR UOX+Gd2O3 pin cell.",
        static={"gd2o3_wtpt": 3.0},
    )
    _write_polaris_example(
        scale70_gd_cr_pin_dir,
        scale_version="7.0.b12",
        model_name="polaris_uoxgdcr_pin_quick",
        model_description="A 2D Polaris BWR UOX+Gd2O3+Cr2O3 pin cell.",
        static={"gd2o3_wtpt": 3.0, "cr2o3_ppm": 3000.0},
    )
    _write_polaris_example(
        scale70_mox_pin_dir,
        scale_version="7.0.b12",
        model_name="polaris_mox_pin_quick",
        fuel_type="MOX",
    )
    _write_polaris_example(
        scale70_triton_mox_dir,
        artifact_contract="TRITON",
        scale_version="7.0.b12",
        model_name="triton_mox_pin_quick",
        fuel_type="MOX",
    )
    scale63 = post_example_validation_comment.collect_example_result(scale63_dir)
    scale63_pin = post_example_validation_comment.collect_example_result(
        scale63_pin_dir
    )
    scale70 = post_example_validation_comment.collect_example_result(scale70_dir)
    scale70_pin = post_example_validation_comment.collect_example_result(
        scale70_pin_dir
    )
    triton = post_example_validation_comment.collect_example_result(triton_dir)
    scale70_gd_pin = post_example_validation_comment.collect_example_result(
        scale70_gd_pin_dir
    )
    scale70_gd_cr_pin = post_example_validation_comment.collect_example_result(
        scale70_gd_cr_pin_dir
    )
    scale70_mox_pin = post_example_validation_comment.collect_example_result(
        scale70_mox_pin_dir
    )
    scale70_triton_mox = post_example_validation_comment.collect_example_result(
        scale70_triton_mox_dir
    )

    comment = post_example_validation_comment.build_grouped_comment(
        [
            post_example_validation_comment.ExampleValidationGroup(
                "SCALE 6.3", (triton, scale63_pin, scale63)
            ),
            post_example_validation_comment.ExampleValidationGroup(
                "SCALE 7.0",
                (
                    scale70_pin,
                    scale70,
                    scale70_gd_pin,
                    scale70_gd_cr_pin,
                    scale70_mox_pin,
                    scale70_triton_mox,
                ),
            ),
        ],
        commit="abc123",
        generated_at="2026-06-06T00:00:00+00:00",
    )

    assert "| SCALE version | code | case | example | q1 | q2 | Pass |" in comment
    assert (
        "| `6.3.*` | `Polaris` | `UOX PWR` | "
        "`examples/polaris_uox_pin_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert (
        "| `6.3.*` | `TRITON` | `UOX PWR` | "
        "`examples/triton_uox_pin_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert (
        "| `6.3.*` | `Polaris` | `BWR UOX+Gd2O3 assembly` | "
        "`examples/polaris_uoxgd_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert (
        "| `7.0.*` | `Polaris` | `UOX PWR` | "
        "`examples/polaris_uox_pin_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert (
        "| `7.0.*` | `Polaris` | `BWR UOX+Gd2O3 assembly` | "
        "`examples/polaris_uoxgd_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert (
        "| `7.0.*` | `Polaris` | `BWR UOX+Gd2O3 pin cell` | "
        "`examples/polaris_uoxgd_pin_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert (
        "| `7.0.*` | `Polaris` | `BWR UOX+Gd2O3+Cr2O3 pin cell` | "
        "`examples/polaris_uoxgdcr_pin_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert (
        "| `7.0.*` | `Polaris` | `MOX PWR` | "
        "`examples/polaris_mox_pin_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert (
        "| `7.0.*` | `TRITON` | `MOX PWR` | "
        "`examples/triton_mox_pin_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in comment
    )
    assert "`scale63_polaris_uoxgd_quick`" not in comment
    assert "`scale70_polaris_uoxgd_quick`" not in comment
    assert "<summary>Burnup Grids</summary>" not in comment
    assert "<summary>Commands</summary>" not in comment


def test_main_accepts_grouped_examples(tmp_path, capsys):
    scale63_dir = tmp_path / "scale63_polaris_uoxgd_quick"
    scale70_dir = tmp_path / "scale70_polaris_uoxgd_quick"
    _write_polaris_example(scale63_dir)
    _write_polaris_example(scale70_dir, scale_version="7.0.b12")

    exit_code = post_example_validation_comment.main(
        [
            "--group",
            f"SCALE 6.3={scale63_dir}",
            "--group",
            f"SCALE 7.0={scale70_dir}",
            "--commit",
            "abc123",
            "--generated-at",
            "2026-06-06T00:00:00+00:00",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "| `6.3.*` | `Polaris` | `BWR UOX+Gd2O3 assembly` | "
        "`examples/polaris_uoxgd_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in output
    )
    assert (
        "| `7.0.*` | `Polaris` | `BWR UOX+Gd2O3 assembly` | "
        "`examples/polaris_uoxgd_quick` | "
        "`0.991357` | `0.982539` | `yes` |" in output
    )


def test_collect_example_result_reads_early_fail_low_order_consistency(tmp_path):
    example_dir = tmp_path / "polaris_uoxgd_quick"
    _write_polaris_example(example_dir)
    _write_json(
        example_dir / "_work" / "check.olm.json",
        {
            "test_pass": False,
            "sequence": [
                {
                    "test_pass": False,
                    "name": "LowOrderConsistency",
                    "eps0": 1e-12,
                    "epsa": 1e-6,
                    "epsr": 1e-3,
                    "target_q1": 0.7,
                    "target_q2": 0.95,
                }
            ],
        },
    )

    result = post_example_validation_comment.collect_example_result(example_dir)

    assert result.passed is False
    loc = result.low_order_consistency[0]
    assert loc.passed is False
    assert loc.q1 is None
    assert loc.q2 is None
    assert loc.target_q1 == 0.7
    assert loc.target_q2 == 0.95


def test_collect_example_result_fails_on_missing_artifact_contract(tmp_path):
    example_dir = tmp_path / "polaris_uoxgd_quick"
    _write_polaris_example(example_dir)
    generate_json = example_dir / "_work" / "generate.olm.json"
    data = json.loads(generate_json.read_text())
    del data["perms"][0]["_scale"]["artifact_contract"]
    _write_json(generate_json, data)

    with pytest.raises(ValueError, match="artifact_contract"):
        post_example_validation_comment.collect_example_result(example_dir)


def test_existing_validation_comment_id_finds_marker(monkeypatch):
    def fake_run_gh(args):
        assert args == ["api", "repos/wawiesel/olm/issues/15/comments?per_page=100"]
        return json.dumps(
            [
                {"id": 1, "body": "ordinary comment"},
                {
                    "id": 2,
                    "body": "text <!-- olm-example-validation --> text",
                },
            ]
        )

    monkeypatch.setattr(post_example_validation_comment, "_run_gh", fake_run_gh)

    comment_id = post_example_validation_comment._existing_validation_comment_id(
        "wawiesel/olm", 15, "<!-- olm-example-validation -->"
    )

    assert comment_id == 2
