"""Interactive prompts for gathering job submission parameters.

Uses questionary for rich terminal prompts (fuzzy select, validation, etc.)
and adapts available choices based on the detected cluster configuration.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import questionary
from questionary import Style

from job_assist.detector import ClusterInfo, Partition

STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "fg:white bold"),
    ("answer", "fg:green bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:green"),
    ("separator", "fg:ansiyellow"),
    ("instruction", "fg:ansibrightblack"),
])

COMMON_SHELLS = ["#!/bin/bash", "#!/bin/sh", "#!/usr/bin/env bash", "#!/usr/bin/env zsh"]

MAIL_TYPE_CHOICES = ["NONE", "BEGIN", "END", "FAIL", "ALL", "REQUEUE", "TIME_LIMIT_90"]


@dataclass
class JobParameters:
    """All parameters needed to generate an sbatch script."""

    job_name: str = "my_job"
    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    nodes: int = 1
    ntasks: int = 1
    ntasks_per_node: Optional[int] = None
    cpus_per_task: int = 1
    mem: Optional[str] = None
    mem_per_cpu: Optional[str] = None
    time: str = "01:00:00"
    gres: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    constraint: Optional[str] = None
    output: str = "%x_%j.out"
    error: str = "%x_%j.err"
    mail_type: Optional[str] = None
    mail_user: Optional[str] = None
    workdir: Optional[str] = None
    exclusive: bool = False
    array: Optional[str] = None
    dependency: Optional[str] = None
    extra_sbatch_lines: List[str] = field(default_factory=list)
    modules: List[str] = field(default_factory=list)
    conda_env: Optional[str] = None
    commands: List[str] = field(default_factory=list)
    shell: str = "#!/bin/bash"


def _validate_time(val: str) -> bool | str:
    """Validate wall time format: MM, HH:MM:SS, or D-HH:MM:SS."""
    if re.match(r"^\d+$", val):
        return True
    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", val):
        return True
    if re.match(r"^\d+-\d{1,2}:\d{2}:\d{2}$", val):
        return True
    return "Use format MM, HH:MM:SS, or D-HH:MM:SS"


def _validate_mem(val: str) -> bool | str:
    """Validate memory specification like 4G, 500M, 1T."""
    if not val:
        return True
    if re.match(r"^\d+[KMGT]?$", val.upper()):
        return True
    return "Use format like 4G, 500M, 1024 (MB implied), or 1T"


def _validate_positive_int(val: str) -> bool | str:
    try:
        if int(val) > 0:
            return True
    except ValueError:
        pass
    return "Must be a positive integer"


def _partition_choices(cluster: ClusterInfo) -> List[questionary.Choice]:
    """Build partition selection choices with resource details."""
    choices = []
    for p in cluster.partitions:
        if p.state != "UP":
            continue
        details = []
        if p.max_time:
            details.append(f"max {p.max_time}")
        if p.cpus_per_node:
            details.append(f"{p.cpus_per_node} cpus/node")
        if p.max_mem_per_node:
            mem_gb = p.max_mem_per_node / 1024
            details.append(f"{mem_gb:.0f}G/node")
        if p.gres:
            gpu_strs = [str(g) for g in p.gres if g.name == "gpu"]
            if gpu_strs:
                details.append(f"GPU: {', '.join(gpu_strs)}")
        suffix = f"  ({', '.join(details)})" if details else ""
        default_tag = " [default]" if p.is_default else ""
        label = f"{p.name}{default_tag}{suffix}"
        choices.append(questionary.Choice(title=label, value=p.name))
    return choices


def _account_choices(cluster: ClusterInfo) -> List[questionary.Choice]:
    choices = []
    for a in cluster.accounts:
        default_tag = " [default]" if a.is_default else ""
        choices.append(questionary.Choice(title=f"{a.name}{default_tag}", value=a.name))
    return choices


def _qos_choices(cluster: ClusterInfo, account_name: Optional[str]) -> List[str]:
    """Get QOS names available for a given account, or all if unknown."""
    if account_name:
        for acct in cluster.accounts:
            if acct.name == account_name and acct.qos_list:
                return acct.qos_list
    return [q.name for q in cluster.qos_list if q.name != "normal"] or []


def _get_partition(cluster: ClusterInfo, name: str) -> Optional[Partition]:
    for p in cluster.partitions:
        if p.name == name:
            return p
    return None


def gather_parameters(cluster: ClusterInfo) -> JobParameters:
    """Run interactive prompts and return a filled JobParameters."""
    params = JobParameters()

    print()
    questionary.print("═══ job-assist: SLURM Job Script Builder ═══", style="bold fg:cyan")
    if cluster.cluster_name:
        questionary.print(f"  Cluster: {cluster.cluster_name}", style="fg:ansibrightblack")
    if cluster.slurm_version:
        questionary.print(f"  SLURM:   {cluster.slurm_version}", style="fg:ansibrightblack")
    print()

    # --- Job basics ---
    questionary.print("── Job Basics ──", style="bold fg:yellow")

    params.job_name = questionary.text(
        "Job name:",
        default="my_job",
        style=STYLE,
    ).ask()
    if params.job_name is None:
        raise KeyboardInterrupt

    # --- Account ---
    if cluster.accounts:
        default_acct = cluster.default_account or cluster.accounts[0].name
        params.account = questionary.select(
            "Account:",
            choices=_account_choices(cluster),
            default=default_acct,
            style=STYLE,
        ).ask()
        if params.account is None:
            raise KeyboardInterrupt

    # --- Partition ---
    if cluster.partitions:
        default_part = cluster.default_partition
        params.partition = questionary.select(
            "Partition:",
            choices=_partition_choices(cluster),
            default=default_part,
            style=STYLE,
        ).ask()
        if params.partition is None:
            raise KeyboardInterrupt
    else:
        params.partition = questionary.text(
            "Partition (leave blank for default):",
            style=STYLE,
        ).ask()

    selected_partition = _get_partition(cluster, params.partition) if params.partition else None

    # --- QOS ---
    available_qos = _qos_choices(cluster, params.account)
    if available_qos:
        use_qos = questionary.confirm("Specify a QOS?", default=False, style=STYLE).ask()
        if use_qos is None:
            raise KeyboardInterrupt
        if use_qos:
            params.qos = questionary.select(
                "QOS:",
                choices=available_qos,
                style=STYLE,
            ).ask()
            if params.qos is None:
                raise KeyboardInterrupt

    # --- Resources ---
    print()
    questionary.print("── Resources ──", style="bold fg:yellow")

    params.nodes = int(
        questionary.text(
            "Number of nodes:",
            default="1",
            validate=_validate_positive_int,
            style=STYLE,
        ).ask()
    )

    task_mode = questionary.select(
        "Task layout:",
        choices=[
            questionary.Choice("Total tasks across all nodes", value="total"),
            questionary.Choice("Tasks per node", value="per_node"),
        ],
        style=STYLE,
    ).ask()
    if task_mode is None:
        raise KeyboardInterrupt

    if task_mode == "total":
        params.ntasks = int(
            questionary.text(
                "Total number of tasks:",
                default=str(params.nodes),
                validate=_validate_positive_int,
                style=STYLE,
            ).ask()
        )
    else:
        params.ntasks_per_node = int(
            questionary.text(
                "Tasks per node:",
                default="1",
                validate=_validate_positive_int,
                style=STYLE,
            ).ask()
        )
        params.ntasks = params.ntasks_per_node * params.nodes

    params.cpus_per_task = int(
        questionary.text(
            "CPUs per task:",
            default="1",
            validate=_validate_positive_int,
            style=STYLE,
        ).ask()
    )

    # --- Memory ---
    mem_mode = questionary.select(
        "Memory specification:",
        choices=[
            questionary.Choice("Memory per node (e.g. 16G)", value="per_node"),
            questionary.Choice("Memory per CPU (e.g. 4G)", value="per_cpu"),
        ],
        style=STYLE,
    ).ask()
    if mem_mode is None:
        raise KeyboardInterrupt

    default_mem = "4G"
    if mem_mode == "per_node":
        params.mem = questionary.text(
            "Memory per node:",
            default=default_mem,
            validate=_validate_mem,
            style=STYLE,
        ).ask()
    else:
        params.mem_per_cpu = questionary.text(
            "Memory per CPU:",
            default=default_mem,
            validate=_validate_mem,
            style=STYLE,
        ).ask()

    # --- Wall time ---
    default_time = "01:00:00"
    if selected_partition and selected_partition.default_time:
        default_time = selected_partition.default_time

    params.time = questionary.text(
        "Wall time (HH:MM:SS or D-HH:MM:SS):",
        default=default_time,
        validate=_validate_time,
        style=STYLE,
    ).ask()
    if params.time is None:
        raise KeyboardInterrupt

    # --- GPU ---
    has_gpu = False
    if selected_partition and selected_partition.gres:
        has_gpu = any(g.name == "gpu" for g in selected_partition.gres)
    elif cluster.gpu_available:
        has_gpu = True

    if has_gpu:
        print()
        questionary.print("── GPU Resources ──", style="bold fg:yellow")

        use_gpu = questionary.confirm("Request GPU resources?", default=True, style=STYLE).ask()
        if use_gpu is None:
            raise KeyboardInterrupt

        if use_gpu:
            if cluster.gpu_types:
                params.gpu_type = questionary.select(
                    "GPU type:",
                    choices=["any"] + cluster.gpu_types,
                    style=STYLE,
                ).ask()
                if params.gpu_type == "any":
                    params.gpu_type = None

            params.gpu_count = int(
                questionary.text(
                    "Number of GPUs:",
                    default="1",
                    validate=_validate_positive_int,
                    style=STYLE,
                ).ask()
            )

            gres_parts = ["gpu"]
            if params.gpu_type:
                gres_parts.append(params.gpu_type)
            gres_parts.append(str(params.gpu_count))
            params.gres = ":".join(gres_parts)

    # --- Constraints / Features ---
    if cluster.features:
        print()
        use_constraint = questionary.confirm(
            "Add node feature constraints?",
            default=False,
            style=STYLE,
        ).ask()
        if use_constraint:
            selected_features = questionary.checkbox(
                "Select features (node constraints):",
                choices=sorted(cluster.features),
                style=STYLE,
            ).ask()
            if selected_features:
                params.constraint = "&".join(selected_features)

    # --- Output ---
    print()
    questionary.print("── Output & Notifications ──", style="bold fg:yellow")

    params.output = questionary.text(
        "Stdout file pattern:",
        default="%x_%j.out",
        style=STYLE,
    ).ask()

    params.error = questionary.text(
        "Stderr file pattern:",
        default="%x_%j.err",
        style=STYLE,
    ).ask()

    params.mail_type = questionary.select(
        "Email notifications:",
        choices=MAIL_TYPE_CHOICES,
        default="NONE",
        style=STYLE,
    ).ask()

    if params.mail_type and params.mail_type != "NONE":
        default_email = os.environ.get("USER", "") + "@" if os.environ.get("USER") else ""
        params.mail_user = questionary.text(
            "Email address:",
            default=default_email,
            style=STYLE,
        ).ask()

    # --- Job array ---
    use_array = questionary.confirm(
        "Is this a job array?",
        default=False,
        style=STYLE,
    ).ask()
    if use_array:
        params.array = questionary.text(
            "Array specification (e.g. 0-99, 1-10%5):",
            style=STYLE,
        ).ask()

    # --- Environment ---
    print()
    questionary.print("── Environment Setup ──", style="bold fg:yellow")

    use_modules = questionary.confirm(
        "Load environment modules?",
        default=False,
        style=STYLE,
    ).ask()
    if use_modules:
        mod_input = questionary.text(
            "Modules to load (space-separated, e.g. gcc/12.2 cuda/12.1):",
            style=STYLE,
        ).ask()
        if mod_input:
            params.modules = mod_input.split()

    use_conda = questionary.confirm(
        "Activate a conda/mamba environment?",
        default=False,
        style=STYLE,
    ).ask()
    if use_conda:
        params.conda_env = questionary.text(
            "Conda environment name:",
            style=STYLE,
        ).ask()

    # --- Commands ---
    print()
    questionary.print("── Commands ──", style="bold fg:yellow")
    questionary.print(
        "  Enter the commands to run (one per prompt). Leave blank to finish.",
        style="fg:ansibrightblack",
    )

    while True:
        cmd = questionary.text(
            "Command:" if not params.commands else "Next command (blank to finish):",
            style=STYLE,
        ).ask()
        if cmd is None:
            raise KeyboardInterrupt
        if not cmd.strip():
            break
        params.commands.append(cmd.strip())

    if not params.commands:
        params.commands = ["echo 'Hello from SLURM job'"]

    # --- Advanced ---
    print()
    advanced = questionary.confirm(
        "Configure advanced options?",
        default=False,
        style=STYLE,
    ).ask()

    if advanced:
        params.exclusive = questionary.confirm(
            "Exclusive node access?",
            default=False,
            style=STYLE,
        ).ask()

        dep_input = questionary.text(
            "Job dependency (e.g. afterok:12345, leave blank for none):",
            default="",
            style=STYLE,
        ).ask()
        if dep_input:
            params.dependency = dep_input

        params.workdir = questionary.text(
            "Working directory (blank for submission dir):",
            default="",
            style=STYLE,
        ).ask() or None

        extra = questionary.text(
            "Extra #SBATCH lines (semicolon-separated, blank for none):",
            default="",
            style=STYLE,
        ).ask()
        if extra:
            params.extra_sbatch_lines = [
                line.strip() for line in extra.split(";") if line.strip()
            ]

    return params
