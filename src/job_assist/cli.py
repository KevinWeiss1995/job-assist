"""CLI entry point for job-assist."""

from __future__ import annotations

import argparse
import sys

from job_assist import __version__
from job_assist.detector import ClusterInfo, detect_cluster
from job_assist.generator import generate_script, preview_script, write_script
from job_assist.prompts import gather_parameters

try:
    import questionary
except ImportError:
    questionary = None  # type: ignore[assignment]


def _print_cluster_info(cluster: ClusterInfo) -> None:
    """Dump detected cluster info (for --detect-only)."""
    print(f"Cluster:        {cluster.cluster_name or 'unknown'}")
    print(f"SLURM version:  {cluster.slurm_version or 'unknown'}")
    print(f"Hostname:       {cluster.hostname or 'unknown'}")
    print(f"Email domain:   {cluster.email_domain or 'unknown'}")
    print(f"TRES enabled:   {cluster.tres_enabled}")
    print(f"GPU available:  {cluster.gpu_available}")
    if cluster.gpu_types:
        print(f"GPU types:      {', '.join(cluster.gpu_types)}")
    print()

    if cluster.gpu_partitions:
        print("GPU Partitions:")
        for p in cluster.gpu_partitions:
            default = " (default)" if p.is_default else ""
            gpu_types = f"  [{', '.join(p.gpu_types)}]" if p.gpu_types else ""
            print(
                f"  {p.name:<20}{default:<12}"
                f"nodes={p.total_nodes:<6}"
                f"max_time={p.max_time or 'unlimited':<16}"
                f"mem={p.max_mem_per_node or '?'}MB"
                f"{gpu_types}"
            )
        print()

    if cluster.cpu_partitions:
        print("CPU Partitions:")
        for p in cluster.cpu_partitions:
            default = " (default)" if p.is_default else ""
            print(
                f"  {p.name:<20}{default:<12}"
                f"nodes={p.total_nodes:<6}"
                f"max_time={p.max_time or 'unlimited':<16}"
                f"mem={p.max_mem_per_node or '?'}MB"
            )
        print()

    if cluster.accounts:
        print("Your accounts:")
        for a in cluster.accounts:
            default = " (default)" if a.is_default else ""
            qos = f"  qos=[{', '.join(a.qos_list)}]" if a.qos_list else ""
            print(f"  {a.name}{default}{qos}")
    print()

    if cluster.features:
        print(f"Node features:  {', '.join(sorted(cluster.features))}")

    if cluster.detection_errors:
        print()
        print("Detection notes:")
        for err in cluster.detection_errors:
            print(f"  ⚠ {err}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="job-assist",
        description="Interactive SLURM sbatch script generator",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Print detected cluster info and exit",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: <job_name>.sbatch)",
    )
    parser.add_argument(
        "--no-detect",
        action="store_true",
        help="Skip cluster auto-detection (manual entry for all fields)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Cluster detection ──
    if args.no_detect:
        cluster = ClusterInfo()
    else:
        try:
            print("Detecting cluster configuration...", end=" ", flush=True)
            cluster = detect_cluster()
            if cluster.has_slurm:
                print(f"found {cluster.cluster_name or 'SLURM cluster'}.")
            else:
                print("SLURM not detected, using manual mode.")
        except Exception as e:
            print(f"\nWarning: detection failed ({e}), using manual mode.")
            cluster = ClusterInfo()

    if args.detect_only:
        _print_cluster_info(cluster)
        return 0

    # ── Interactive prompts ──
    try:
        params = gather_parameters(cluster)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130

    # ── Generate script ──
    script = generate_script(params)
    preview_script(script)

    # ── Save ──
    out_path = args.output or f"{params.job_name}.sbatch"

    if questionary:
        save = questionary.confirm(
            f"Save to {out_path}?", default=True
        ).ask()
        if save is None:
            print("Aborted.")
            return 130
    else:
        resp = input(f"\nSave to {out_path}? [Y/n] ").strip().lower()
        save = resp in ("", "y", "yes")

    if save:
        final_path = write_script(script, out_path)
        print(f"\n✓ Script saved to {final_path}")
        print(f"  Submit with:  sbatch {out_path}")
    else:
        print("\nScript not saved. You can copy it from the preview above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
