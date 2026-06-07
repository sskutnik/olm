#!/usr/bin/env python3
"""Build and post standardized OLM example validation comments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as _datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable

DEFAULT_MARKER = "<!-- olm-example-validation -->"


@dataclass(frozen=True)
class LowOrderConsistencyMetricResult:
    metric: str
    units: str | None
    passed: bool
    q1: float | None
    q2: float | None
    target_q1: float | None
    target_q2: float | None
    relative_failures: int | None
    absolute_relative_failures: int | None
    total_comparisons: int | None
    mean_relative_difference: float | None
    histogram_path: str | None


@dataclass(frozen=True)
class LowOrderConsistencyResult(LowOrderConsistencyMetricResult):
    atom_fraction: LowOrderConsistencyMetricResult | None
    grams_per_initial_hm: LowOrderConsistencyMetricResult | None


@dataclass(frozen=True)
class ExampleValidationResult:
    example: str
    model_name: str
    case_label: str
    fuel_type: str
    scale_version: str
    command_line: str
    successful_runs: int
    total_runs: int
    point_count: int
    contracts: tuple[str, ...]
    burnup_lists: tuple[tuple[float, ...], ...]
    system_json_normalized: bool
    low_order_consistency: tuple[LowOrderConsistencyResult, ...]

    @property
    def passed(self) -> bool:
        artifacts_passed = (
            self.total_runs > 0
            and self.successful_runs == self.total_runs
            and self.point_count > 0
            and self.system_json_normalized
        )
        if not artifacts_passed:
            return False
        if self.low_order_consistency:
            return all(result.passed for result in self.low_order_consistency)
        return True


@dataclass(frozen=True)
class ExampleValidationGroup:
    label: str
    results: tuple[ExampleValidationResult, ...]


def _read_json(path: Path):
    if not path.exists():
        raise ValueError(f"Required validation artifact does not exist: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _require_mapping_key(mapping, key: str, source: Path):
    if key not in mapping:
        raise ValueError(f"Required key={key!r} missing from {source}")
    return mapping[key]


def _unique_preserve_order(values: Iterable):
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return tuple(unique)


def _comment_header(marker: str, commit: str, generated_at: str) -> list[str]:
    return [
        marker,
        "### OLM Low-Order Consistency Results",
        "",
        f"Commit: `{commit}`",
        f"Generated: `{generated_at}`",
        "",
        "These are local SCALE validation results. GitHub-hosted CI does not run "
        "export-controlled SCALE calculations.",
        "",
    ]


def _optional_number(mapping: dict, key: str):
    value = mapping.get(key)
    if value is None:
        return None
    return value


def _relative_artifact_path(work_dir: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(work_dir))
    except ValueError:
        return str(path)


def _collect_low_order_consistency(
    work_dir: Path, check_path: Path
) -> tuple[LowOrderConsistencyResult, ...]:
    if not check_path.exists():
        return tuple()
    check = _read_json(check_path)
    sequence = _require_mapping_key(check, "sequence", check_path)
    results = []
    for entry in sequence:
        if entry.get("name") != "LowOrderConsistency":
            continue
        primary_metric = str(entry.get("metric", "atom_fraction"))
        atom_fraction_result = None
        grams_per_initial_hm_result = None
        atom_fraction = entry.get("atom_fraction")
        if isinstance(atom_fraction, dict):
            atom_fraction_result = _collect_low_order_consistency_metric(
                work_dir, atom_fraction, "atom_fraction"
            )
        grams_per_initial_hm = entry.get("grams_per_initial_hm")
        if isinstance(grams_per_initial_hm, dict):
            grams_per_initial_hm_result = _collect_low_order_consistency_metric(
                work_dir, grams_per_initial_hm, "grams_per_initial_hm"
            )
        primary = _collect_low_order_consistency_metric(
            work_dir,
            entry,
            primary_metric,
        )
        results.append(
            LowOrderConsistencyResult(
                **primary.__dict__,
                atom_fraction=(
                    None if primary_metric == "atom_fraction" else atom_fraction_result
                ),
                grams_per_initial_hm=(
                    None
                    if primary_metric == "grams_per_initial_hm"
                    else grams_per_initial_hm_result
                ),
            )
        )
    return tuple(results)


def _collect_low_order_consistency_metric(
    work_dir: Path, entry: dict, metric: str
) -> LowOrderConsistencyMetricResult:
    return LowOrderConsistencyMetricResult(
        metric=metric,
        units=entry.get("units"),
        passed=entry.get("test_pass") is True,
        q1=_optional_number(entry, "q1"),
        q2=_optional_number(entry, "q2"),
        target_q1=_optional_number(entry, "target_q1"),
        target_q2=_optional_number(entry, "target_q2"),
        relative_failures=_optional_number(entry, "wr"),
        absolute_relative_failures=_optional_number(entry, "wa"),
        total_comparisons=_optional_number(entry, "m"),
        mean_relative_difference=_optional_number(entry, "mean_rel_diff"),
        histogram_path=_relative_artifact_path(work_dir, entry.get("hist_image")),
    )


def collect_example_result(example_dir: Path) -> ExampleValidationResult:
    example_dir = example_dir.resolve()
    work_dir = example_dir / "_work"
    config_path = example_dir / "config.olm.json"
    generate_path = work_dir / "generate.olm.json"
    run_path = work_dir / "run.olm.json"
    assemble_path = work_dir / "assemble.olm.json"
    check_path = work_dir / "check.olm.json"

    config = _read_json(config_path)
    generate = _read_json(generate_path)
    run = _read_json(run_path)
    assemble = _read_json(assemble_path)

    model = _require_mapping_key(config, "model", config_path)
    model_name = _require_mapping_key(model, "name", config_path)
    runs = _require_mapping_key(run, "runs", run_path)
    perms = _require_mapping_key(generate, "perms", generate_path)
    points = _require_mapping_key(assemble, "points", assemble_path)

    contracts = []
    for perm in perms:
        scale = _require_mapping_key(perm, "_scale", generate_path)
        contracts.append(
            _require_mapping_key(scale, "artifact_contract", generate_path)
        )
    contracts = _unique_preserve_order(contracts)
    burnup_lists = _unique_preserve_order(
        tuple(float(value) for value in point["_arpinfo"]["burnup_list"])
        for point in points
    )

    normalized_flags = []
    for point in points:
        ii_json = work_dir / point["files"]["ii_json"]
        ii = _read_json(ii_json)
        responses = _require_mapping_key(ii, "responses", ii_json)
        normalized_flags.append(set(responses.keys()) == {"system"})

    return ExampleValidationResult(
        example=example_dir.name,
        model_name=model_name,
        case_label=_summary_case_label(config),
        fuel_type=str(
            _require_mapping_key(
                _require_mapping_key(config, "assemble", config_path),
                "fuel_type",
                config_path,
            )
        ),
        scale_version=_require_mapping_key(run, "version", run_path),
        command_line=_require_mapping_key(run, "command_line", run_path),
        successful_runs=sum(1 for run_info in runs if run_info["success"] is True),
        total_runs=len(runs),
        point_count=len(points),
        contracts=contracts,
        burnup_lists=burnup_lists,
        system_json_normalized=all(normalized_flags) and len(normalized_flags) > 0,
        low_order_consistency=_collect_low_order_consistency(work_dir, check_path),
    )


def build_comment(
    results: list[ExampleValidationResult],
    *,
    marker: str = DEFAULT_MARKER,
    commit: str,
    generated_at: str,
) -> str:
    if not results:
        raise ValueError("At least one example validation result is required")

    lines = _comment_header(marker, commit, generated_at)
    _append_validation_dashboard(lines, results)
    return "\n".join(lines) + "\n"


def build_grouped_comment(
    groups: list[ExampleValidationGroup],
    *,
    marker: str = DEFAULT_MARKER,
    commit: str,
    generated_at: str,
) -> str:
    if not groups:
        raise ValueError("At least one example validation group is required")

    lines = _comment_header(marker, commit, generated_at)
    results = []
    for group in groups:
        if not group.results:
            raise ValueError(
                f"At least one example validation result is required for group={group.label!r}"
            )
        results.extend(group.results)
    _append_validation_dashboard(lines, results)
    return "\n".join(lines) + "\n"


def _append_validation_dashboard(
    lines: list[str], results: list[ExampleValidationResult]
):
    lines.extend(
        [
            "Low-order consistency (`g/gIHM`):",
            "",
            "| SCALE version | code | case | example | q1 | q2 | Pass |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in sorted(results, key=_result_sort_key):
        metric = _first_low_order_consistency_comment_metric(result)
        passed = metric is not None and metric.passed
        q1 = _format_score(metric.q1) if metric is not None else "n/a"
        q2 = _format_score(metric.q2) if metric is not None else "n/a"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{_scale_label(result.scale_version)}.*`",
                    f"`{_result_product(result)}`",
                    f"`{result.case_label}`",
                    f"`examples/{result.model_name}`",
                    f"`{q1}`",
                    f"`{q2}`",
                    "`yes`" if passed else "`no`",
                ]
            )
            + " |"
        )


def _format_score(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _scale_label(version: str) -> str:
    parts = version.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return ".".join(parts[:2])
    return version


def _scale_sort_key(label: str):
    parts = label.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]), label)
    return (999, 999, label)


def _result_sort_key(result: ExampleValidationResult):
    product = _result_product(result)
    product_order = {"Polaris": 0, "TRITON": 1}
    case_order = {
        "UOX PWR": 0,
        "UOX BWR": 1,
        "BWR UOX+Gd2O3 assembly": 2,
        "BWR UOX+Gd2O3 pin cell": 3,
        "BWR UOX+Gd2O3+Cr2O3 pin cell": 4,
        "BWR UOX+Gd2O3+Cr2O3 assembly": 5,
        "MOX PWR": 6,
        "PWR UOX+Gd2O3 pin cell": 7,
        "PWR UOX+Gd2O3+Cr2O3 pin cell": 8,
    }
    return (
        _scale_sort_key(_scale_label(result.scale_version)),
        case_order.get(result.case_label, 99),
        product_order.get(product, 99),
        result.case_label,
        product,
    )


def _result_product(result: ExampleValidationResult) -> str:
    return "/".join(result.contracts)


def _first_low_order_consistency_comment_metric(
    result: ExampleValidationResult,
) -> LowOrderConsistencyMetricResult | None:
    if not result.low_order_consistency:
        return None
    return _low_order_consistency_comment_metric(result.low_order_consistency[0])


def _low_order_consistency_comment_metric(
    loc: LowOrderConsistencyResult,
) -> LowOrderConsistencyMetricResult | None:
    if loc.metric == "grams_per_initial_hm":
        return loc
    return loc.grams_per_initial_hm


def _summary_case_label(config: dict) -> str:
    model = _require_mapping_key(config, "model", Path("config.olm.json"))
    model_name = str(_require_mapping_key(model, "name", Path("config.olm.json")))
    description = str(model.get("description", ""))
    assemble = _require_mapping_key(config, "assemble", Path("config.olm.json"))
    fuel_type = str(
        _require_mapping_key(assemble, "fuel_type", Path("config.olm.json"))
    )
    text = f"{model_name} {description}".lower()
    has_gd = _contains_key_fragment(config, "gd2o3")
    has_cr = _contains_key_fragment(config, "cr2o3")
    is_pin_cell = (
        "_pin_" in model_name.lower()
        or model_name.lower().endswith("_pin_quick")
        or " pin cell" in description.lower()
        or " single-pin" in description.lower()
    )
    if "bwr" in text and has_gd and has_cr and is_pin_cell:
        return "BWR UOX+Gd2O3+Cr2O3 pin cell"
    if "bwr" in text and has_gd and is_pin_cell:
        return "BWR UOX+Gd2O3 pin cell"
    if "bwr" in text and has_gd and has_cr:
        return "BWR UOX+Gd2O3+Cr2O3 assembly"
    if "bwr" in text and has_gd:
        return "BWR UOX+Gd2O3 assembly"
    if "bwr" in text:
        return "UOX BWR"
    if has_gd and has_cr and is_pin_cell:
        return "PWR UOX+Gd2O3+Cr2O3 pin cell"
    if has_gd and has_cr:
        return "PWR UOX+Gd2O3+Cr2O3 assembly"
    if has_gd and is_pin_cell:
        return "PWR UOX+Gd2O3 pin cell"
    if has_gd:
        return "PWR UOX+Gd2O3 assembly"
    if fuel_type == "MOX":
        return "MOX PWR"
    if fuel_type == "UOX":
        return "UOX PWR"
    return fuel_type


def _contains_key_fragment(value, fragment: str) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if fragment in str(key).lower():
                return True
            if _contains_key_fragment(item, fragment):
                return True
    elif isinstance(value, list):
        return any(_contains_key_fragment(item, fragment) for item in value)
    return False


def _run_gh(args: list[str], input_path: Path | None = None) -> str:
    command = ["gh", *args]
    if input_path is not None:
        command.extend(["--input", str(input_path)])
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            "GitHub CLI command failed: "
            + " ".join(command)
            + "\n"
            + completed.stderr.strip()
        )
    return completed.stdout


def _existing_validation_comment_id(
    repo: str, pr_number: int, marker: str
) -> int | None:
    text = _run_gh(["api", f"repos/{repo}/issues/{pr_number}/comments?per_page=100"])
    comments = json.loads(text)
    for comment in comments:
        if marker in comment["body"]:
            return int(comment["id"])
    return None


def post_comment(repo: str, pr_number: int, body: str, marker: str) -> int:
    comment_id = _existing_validation_comment_id(repo, pr_number, marker)
    payload = {"body": body}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=True) as f:
        json.dump(payload, f)
        f.flush()
        if comment_id is None:
            response = _run_gh(
                [
                    "api",
                    "--method",
                    "POST",
                    f"repos/{repo}/issues/{pr_number}/comments",
                ],
                input_path=Path(f.name),
            )
        else:
            response = _run_gh(
                [
                    "api",
                    "--method",
                    "PATCH",
                    f"repos/{repo}/issues/comments/{comment_id}",
                ],
                input_path=Path(f.name),
            )
    return int(json.loads(response)["id"])


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("Could not determine git commit: " + completed.stderr.strip())
    return completed.stdout.strip()


def _utc_now() -> str:
    return (
        _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _parse_group_arg(value: str) -> tuple[str, tuple[Path, ...]]:
    label, separator, example_text = value.partition("=")
    if not separator or not label.strip() or not example_text.strip():
        raise argparse.ArgumentTypeError(
            "--group must be formatted as 'label=example_dir[,example_dir...]'"
        )
    examples = tuple(
        Path(item.strip()) for item in example_text.split(",") if item.strip()
    )
    if not examples:
        raise argparse.ArgumentTypeError(
            "--group must include at least one example directory"
        )
    return label.strip(), examples


def parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(
        description=(
            "Build a standardized PR comment from completed OLM example validation "
            "artifacts and optionally post it to GitHub."
        )
    )
    parser.add_argument(
        "--example",
        action="append",
        type=Path,
        default=[],
        help="Example directory containing config.olm.json and _work/*.olm.json.",
    )
    parser.add_argument(
        "--group",
        action="append",
        type=_parse_group_arg,
        default=[],
        help="Result-set label and example directories as label=dir[,dir...].",
    )
    parser.add_argument("--repo", help="GitHub repository in owner/name form.")
    parser.add_argument("--pr", type=int, help="Pull request number.")
    parser.add_argument(
        "--post",
        action="store_true",
        help="Upsert the standardized comment on the pull request using gh.",
    )
    parser.add_argument(
        "--commit", default=None, help="Commit SHA to show in the comment."
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Timestamp to show in the comment. Defaults to current UTC time.",
    )
    parser.add_argument(
        "--marker",
        default=DEFAULT_MARKER,
        help="Hidden marker used to find and update the existing PR comment.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    commit = args.commit or _git_commit()
    generated_at = args.generated_at or _utc_now()
    if args.group:
        if args.example:
            raise ValueError("--example cannot be combined with --group")
        groups = [
            ExampleValidationGroup(
                label,
                tuple(collect_example_result(example) for example in examples),
            )
            for label, examples in args.group
        ]
        comment = build_grouped_comment(
            groups,
            marker=args.marker,
            commit=commit,
            generated_at=generated_at,
        )
    else:
        if not args.example:
            raise ValueError("At least one --example or --group is required")
        results = [collect_example_result(example) for example in args.example]
        comment = build_comment(
            results,
            marker=args.marker,
            commit=commit,
            generated_at=generated_at,
        )

    if not args.post:
        print(comment, end="")
        return 0

    if not args.repo or args.pr is None:
        raise ValueError("--repo and --pr are required with --post")
    comment_id = post_comment(args.repo, args.pr, comment, args.marker)
    print(f"Posted OLM example validation comment id={comment_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
