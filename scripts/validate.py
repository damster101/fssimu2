#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "rich",
#     "scipy",
# ]
# ///

import argparse
import shutil
import subprocess
import sys
import tempfile
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
from rich import print
from scipy.stats import kendalltau, pearsonr, spearmanr


class Row(TypedDict):
    image: str
    level: float
    tag: str
    binary: str
    score: float


class Stats(TypedDict, total=False):
    n: int
    mean: float
    stddev: float
    stderr: float
    min: float
    max: float


class TagComparison(TypedDict):
    ref: Stats
    other: Stats
    diff: Stats


def run_cmd(cmd: Sequence[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.decode().strip(), p.stderr.decode().strip()


def find_binary(name_or_path: str) -> Optional[str]:
    if Path(name_or_path).exists():
        return str(Path(name_or_path))
    found = shutil.which(name_or_path)
    return found


def parse_score(output: str, fallback_output: str = "") -> float:
    """Parse the last line of output into a float score. Try output first, then fallback."""
    lines = output.strip().splitlines()
    if lines:
        try:
            return float(lines[-1].strip())
        except ValueError:
            pass

    if fallback_output:
        fallback_lines = fallback_output.strip().splitlines()
        if fallback_lines:
            try:
                return float(fallback_lines[-1].strip())
            except ValueError:
                pass

    raise ValueError("empty output or unparseable score")


def compute_correlations(
    ref_scores: List[float], other_scores: List[float]
) -> Dict[str, Tuple[float, float]]:
    """Compute correlation coefficients between reference and other scores.
    Returns dict with 'pcc', 'srcc', 'krcc' keys, values are (correlation, p-value) tuples.
    """
    if len(ref_scores) < 2 or len(other_scores) < 2:
        return {
            "pcc": (float("nan"), float("nan")),
            "srcc": (float("nan"), float("nan")),
            "krcc": (float("nan"), float("nan")),
        }

    ref_array = np.array(ref_scores)
    other_array = np.array(other_scores)

    pcc_corr, pcc_p = pearsonr(ref_array, other_array)
    srcc_corr, srcc_p = spearmanr(ref_array, other_array)
    krcc_corr, krcc_p = kendalltau(ref_array, other_array)

    return {
        "pcc": (pcc_corr, pcc_p),  # Pearson Correlation Coefficient
        "srcc": (srcc_corr, srcc_p),  # Spearman Rank Correlation Coefficient
        "krcc": (krcc_corr, krcc_p),  # Kendall Rank Correlation Coefficient
    }


def stat_summary(vals: List[float]) -> Stats:
    n = len(vals)
    if n == 0:
        return {}
    mu = mean(vals)
    sd = stdev(vals) if n > 1 else 0.0
    se = sd / sqrt(n) if n > 0 else 0.0
    return {
        "n": n,
        "mean": mu,
        "stddev": sd,
        "stderr": se,
        "min": min(vals),
        "max": max(vals),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Encode images to JXL at several distances and compare SSIMULACRA2 outputs."
    )
    p.add_argument("images", nargs="+", help="input image files")
    p.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 4.0],
        help="cjxl -d levels (default: 1.0 2.0 4.0)",
    )
    p.add_argument("--cjxl", default="cjxl", help="path to cjxl")
    p.add_argument("--djxl", default="djxl", help="path to djxl")
    p.add_argument(
        "--ref-binary",
        default="ssimulacra2",
        help="reference ssimulacra2 binary (default: ssimulacra2)",
    )
    p.add_argument(
        "--custom",
        help="additional ssimulacra2-compatible binary to run against (optional)",
    )
    p.add_argument("--csv", help="write per-sample results to CSV")
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="do not delete temporary files (for debugging)",
    )
    p.add_argument("--debug", action="store_true", help="enable debug output")
    args = p.parse_args()

    cjxl = find_binary(args.cjxl)
    djxl = find_binary(args.djxl)
    ref_bin = find_binary(args.ref_binary)
    custom_bin = find_binary(args.custom) if args.custom else None

    if not cjxl:
        print("cjxl not found:", args.cjxl, file=sys.stderr)
        sys.exit(2)
    if not djxl:
        print("djxl not found:", args.djxl, file=sys.stderr)
        sys.exit(2)
    if not ref_bin:
        print("reference ssimulacra2 not found:", args.ref_binary, file=sys.stderr)
        sys.exit(2)
    if args.custom and not custom_bin:
        print("custom binary not found:", args.custom, file=sys.stderr)
        sys.exit(2)

    binaries = [("ref", ref_bin)]
    if custom_bin:
        binaries.append(("custom", custom_bin))

    rows: List[Row] = []

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for img in args.images:
            inp = Path(img)
            if not inp.exists():
                print("skipping missing file:", img, file=sys.stderr)
                continue

            for level in args.levels:
                jxl_path = td / f"{inp.stem}_d{level:.3g}.jxl"
                rc, out, err = run_cmd(
                    [cjxl, "-d", str(level), str(inp), str(jxl_path)]
                )
                if rc != 0:
                    print(
                        f"cjxl failed for {img} level {level}: {err}", file=sys.stderr
                    )
                    continue
                png_path = td / f"{inp.stem}_d{level:.3g}.png"
                rc, out, err = run_cmd([djxl, str(jxl_path), str(png_path)])
                if rc != 0 or not png_path.exists():
                    print(f"djxl failed for {jxl_path}: {err}", file=sys.stderr)
                    continue

                for tag, binpath in binaries:
                    cmd = [binpath, str(inp), str(png_path)]
                    if args.debug:
                        print(
                            f"DEBUG: Running command: {' '.join(cmd)}", file=sys.stderr
                        )
                    rc, out, err = run_cmd(cmd)
                    if args.debug:
                        print(
                            f"DEBUG: {tag} binary rc={rc}, stdout='{out}', stderr='{err}'",
                            file=sys.stderr,
                        )
                    if rc != 0:
                        print(
                            f"{tag} binary failed for {img} level {level}: {err}",
                            file=sys.stderr,
                        )
                        continue
                    try:
                        score = parse_score(out, err)
                    except Exception as e:
                        print(
                            f"failed to parse score from {tag} output for {img} level {level}: stdout='{out}', stderr='{err}' (error: {e})",
                            file=sys.stderr,
                        )
                        continue
                    rows.append(
                        {
                            "image": str(inp),
                            "level": level,
                            "tag": tag,
                            "binary": binpath,
                            "score": score,
                        }
                    )

                if not args.keep_temp:
                    try:
                        jxl_path.unlink(missing_ok=True)
                        png_path.unlink(missing_ok=True)
                    except Exception:
                        pass

        pairs: Dict[Tuple[str, float], Dict[str, float]] = {}
        for r in rows:
            key = (r["image"], r["level"])
            pairs.setdefault(key, {})[r["tag"]] = r["score"]

        comparisons: Dict[str, List[Tuple[float, float]]] = {}
        for (img, lvl), d in pairs.items():
            if "ref" not in d:
                continue
            ref_score = d["ref"]
            for tag in d:
                if tag == "ref":
                    continue
                comparisons.setdefault(tag, []).append((ref_score, d[tag]))

        print("SAMPLES:", sum(1 for k in pairs.keys() if "ref" in pairs[k]))
        print("LEVELS:", " ".join(str(x) for x in args.levels))
        print()

        summary: Dict[str, TagComparison] = {}
        for tag in comparisons:
            ref_list = [a for a, b in comparisons[tag]]
            other_list = [b for a, b in comparisons[tag]]
            diffs = [b - a for a, b in comparisons[tag]]
            s_ref = stat_summary(ref_list)
            s_other = stat_summary(other_list)
            s_diff = stat_summary(diffs)
            summary[tag] = {"ref": s_ref, "other": s_other, "diff": s_diff}

            mean_diff = s_diff.get("mean", 0)
            ref_mean = s_ref.get("mean", 0)
            percent_error = (
                abs(mean_diff) / abs(ref_mean) * 100 if ref_mean != 0 else float("nan")
            )
            max_error = max([abs(d) for d in diffs]) if diffs else float("nan")

            correlations = compute_correlations(ref_list, other_list)

            print(f"== {tag} vs ref ==")
            print(f" pairs: {s_diff.get('n', 0)}")
            print(
                f" ref mean: {s_ref.get('mean', float('nan')):.6g}  std: {s_ref.get('stddev', 0):.6g}"
            )
            print(
                f" {tag} mean: {s_other.get('mean', float('nan')):.6g}  std: {s_other.get('stddev', 0):.6g}"
            )
            print(f" mean diff (other - ref): {s_diff.get('mean', 0):.6g}")
            print(f" diff stddev: {s_diff.get('stddev', 0):.6g}")
            print(f" diff stderr: {s_diff.get('stderr', 0):.6g}")
            print(f" percentage error (mean diff / ref mean): {percent_error:.3f}%")
            print(f" max absolute error: {max_error:.6g}")
            print("\n== correlation ==")
            print(f" PCC (Pearson): {correlations['pcc'][0]:.6f}")
            print(f" SRCC (Spearman): {correlations['srcc'][0]:.6f}")
            print(f" KRCC (Kendall): {correlations['krcc'][0]:.6f}")
            print()

        if args.csv:
            import csv

            fieldnames = ["image", "level", "ref_score", "tag", "tag_score", "diff"]
            with open(args.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames)
                w.writeheader()
                for (img, lvl), d in pairs.items():
                    if "ref" not in d:
                        continue
                    ref_score = d["ref"]
                    for tag in d:
                        if tag == "ref":
                            continue
                        w.writerow(
                            {
                                "image": img,
                                "level": lvl,
                                "ref_score": ref_score,
                                "tag": tag,
                                "tag_score": d[tag],
                                "diff": d[tag] - ref_score,
                            }
                        )
            print("wrote CSV:", args.csv)


if __name__ == "__main__":
    main()
