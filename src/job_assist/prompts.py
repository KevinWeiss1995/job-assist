"""Interactive prompts for gathering job submission parameters.

TRES-forward design: users specify the resources they need first (GPUs, CPUs,
memory, time). The partition is auto-suggested based on those resources and
only shown as a confirmation step. This reflects the modern SLURM resource-based
scheduling model where partitions are a routing detail, not the primary choice.
"""

from __future__ import annotations

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

MAIL_TYPE_CHOICES = ["ALL", "NONE", "BEGIN", "END", "FAIL", "REQUEUE", "TIME_LIMIT_90"]


@dataclass
class JobParameters:
    """All parameters needed to generate an sbatch script."""

    job_name: str = "my_job"
    job_type: str = "cpu"  # "cpu" or "gpu"

    partition: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None

    # CPU job resources
    nodes: int = 1
    ntasks: int = 1
    ntasks_per_node: Optional[int] = None
    cpus_per_task: int = 4
    mem: Optional[str] = None
    mem_per_cpu: Optional[str] = None

    # GPU job resources
    gpu_type: Optional[str] = None
    gpu_count: int = 1
    cpus_per_gpu: int = 4
    mem_per_gpu: Optional[str] = None
    gres: Optional[str] = None

    # Time
    time: str = "04:00:00"

    # Output
    output: str = "%x_%j.out"
    error: str = "%x_%j.err"

    # Notifications
    mail_type: Optional[str] = None
    mail_user: Optional[str] = None

    # Advanced
    constraint: Optional[str] = None
    workdir: Optional[str] = None
    exclusive: bool = False
    array: Optional[str] = None
    dependency: Optional[str] = None
    extra_sbatch_lines: List[str] = field(default_factory=list)

    # Environment
    modules: List[str] = field(default_factory=list)
    conda_env: Optional[str] = None
    commands: List[str] = field(default_factory=list)
    shell: str = "#!/bin/bash"


# ── Validators ──

def _validate_time(val: str) -> bool | str:
    if re.match(r"^\d+$", val):
        return True
    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", val):
        return True
    if re.match(r"^\d+-\d{1,2}:\d{2}:\d{2}$", val):
        return True
    return "Use format MM, HH:MM:SS, or D-HH:MM:SS"


def _validate_mem(val: str) -> bool | str:
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


# ── Partition auto-selection ──

def _suggest_partition(
    cluster: ClusterInfo,
    is_gpu: bool,
    gpu_type: Optional[str] = None,
) -> Optional[str]:
    """Pick the best partition for the requested resources.

    For GPU jobs: prefer the broadest GPU partition (most nodes/types).
    For CPU jobs: prefer the default partition, or the largest CPU partition.
    """
    if is_gpu:
        candidates = cluster.gpu_partitions
        if gpu_type:
            with_type = [
                p for p in candidates if gpu_type in p.gpu_types
            ]
            if with_type:
                candidates = with_type
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.total_nodes, reverse=True)
        return candidates[0].name
    else:
        candidates = cluster.cpu_partitions
        if not candidates:
            return cluster.default_partition
        for p in candidates:
            if p.is_default:
                return p.name
        candidates.sort(key=lambda p: p.total_nodes, reverse=True)
        return candidates[0].name


def _partition_choices(
    partitions: List[Partition],
    show_gpu_info: bool = False,
) -> List[questionary.Choice]:
    choices = []
    for p in partitions:
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
        if show_gpu_info and p.gpu_types:
            details.append(f"GPU: {', '.join(p.gpu_types)}")
        elif show_gpu_info and p.has_gpu:
            details.append("GPU: yes")
        if p.total_nodes:
            details.append(f"{p.total_nodes} nodes")
        suffix = f"  ({', '.join(details)})" if details else ""
        label = f"{p.name}{suffix}"
        choices.append(questionary.Choice(title=label, value=p.name))
    return choices


def _get_partition(cluster: ClusterInfo, name: str) -> Optional[Partition]:
    for p in cluster.partitions:
        if p.name == name:
            return p
    return None


def _ask_or_abort(question: questionary.Question) -> str:
    result = question.ask()
    if result is None:
        raise KeyboardInterrupt
    return result


# ── Main prompt flow ──

def gather_parameters(cluster: ClusterInfo) -> JobParameters:
    """TRES-forward prompt flow: resources first, partition auto-suggested."""
    params = JobParameters()

    # ── Header ──
    print()
    questionary.print(
        "═══ job-assist: SLURM Job Script Builder ═══", style="bold fg:cyan"
    )
    if cluster.cluster_name:
        questionary.print(f"  Cluster: {cluster.cluster_name}", style="fg:ansibrightblack")
    if cluster.slurm_version:
        questionary.print(f"  SLURM:   {cluster.slurm_version}", style="fg:ansibrightblack")
    if cluster.gpu_types:
        questionary.print(
            f"  GPUs:    {', '.join(cluster.gpu_types)}", style="fg:ansibrightblack"
        )
    print()

    # ── Job type ──
    questionary.print("── Job Configuration ──", style="bold fg:yellow")

    params.job_name = _ask_or_abort(questionary.text(
        "Job name:",
        default="my_job",
        style=STYLE,
    ))

    if cluster.gpu_available:
        params.job_type = _ask_or_abort(questionary.select(
            "Job type:",
            choices=[
                questionary.Choice("GPU job", value="gpu"),
                questionary.Choice("CPU job", value="cpu"),
            ],
            style=STYLE,
        ))
    else:
        params.job_type = "cpu"

    is_gpu = params.job_type == "gpu"

    # ── Resources FIRST (TRES-forward) ──
    if is_gpu:
        print()
        questionary.print("── Resources ──", style="bold fg:yellow")

        if cluster.gpu_types:
            params.gpu_type = _ask_or_abort(questionary.select(
                "GPU type:",
                choices=cluster.gpu_types,
                style=STYLE,
            ))
        else:
            typed = _ask_or_abort(questionary.text(
                "GPU type (e.g. v100, a100, l40s — leave blank for any):",
                style=STYLE,
            ))
            params.gpu_type = typed if typed.strip() else None

        params.gpu_count = int(_ask_or_abort(questionary.text(
            "Number of GPUs:",
            default="1",
            validate=_validate_positive_int,
            style=STYLE,
        )))

        gres_parts = ["gpu"]
        if params.gpu_type:
            gres_parts.append(params.gpu_type)
        gres_parts.append(str(params.gpu_count))
        params.gres = ":".join(gres_parts)

        params.cpus_per_gpu = int(_ask_or_abort(questionary.text(
            "CPUs per GPU:",
            default="4",
            validate=_validate_positive_int,
            style=STYLE,
        )))

        params.mem_per_gpu = _ask_or_abort(questionary.text(
            "Memory per GPU (e.g. 16G):",
            default="16G",
            validate=_validate_mem,
            style=STYLE,
        ))
    else:
        print()
        questionary.print("── Resources ──", style="bold fg:yellow")

        params.cpus_per_task = int(_ask_or_abort(questionary.text(
            "CPUs (cores):",
            default="4",
            validate=_validate_positive_int,
            style=STYLE,
        )))

        mem_mode = _ask_or_abort(questionary.select(
            "Memory specification:",
            choices=[
                questionary.Choice("Total memory for the job (--mem)", value="total"),
                questionary.Choice("Memory per CPU (--mem-per-cpu)", value="per_cpu"),
            ],
            style=STYLE,
        ))

        if mem_mode == "total":
            params.mem = _ask_or_abort(questionary.text(
                "Total memory (e.g. 8G):",
                default="8G",
                validate=_validate_mem,
                style=STYLE,
            ))
        else:
            params.mem_per_cpu = _ask_or_abort(questionary.text(
                "Memory per CPU (e.g. 4G):",
                default="4G",
                validate=_validate_mem,
                style=STYLE,
            ))

        multi_node = _ask_or_abort(
            questionary.confirm("Multi-node job?", default=False, style=STYLE)
        )
        if multi_node:
            params.nodes = int(_ask_or_abort(questionary.text(
                "Number of nodes:",
                default="2",
                validate=_validate_positive_int,
                style=STYLE,
            )))
            task_mode = _ask_or_abort(questionary.select(
                "Task layout:",
                choices=[
                    questionary.Choice("Total tasks across all nodes", value="total"),
                    questionary.Choice("Tasks per node", value="per_node"),
                ],
                style=STYLE,
            ))
            if task_mode == "total":
                params.ntasks = int(_ask_or_abort(questionary.text(
                    "Total number of tasks:",
                    default=str(params.nodes),
                    validate=_validate_positive_int,
                    style=STYLE,
                )))
            else:
                params.ntasks_per_node = int(_ask_or_abort(questionary.text(
                    "Tasks per node:",
                    default="1",
                    validate=_validate_positive_int,
                    style=STYLE,
                )))
                params.ntasks = params.ntasks_per_node * params.nodes

    # ── Wall time ──
    params.time = _ask_or_abort(questionary.text(
        "Wall time (HH:MM:SS or D-HH:MM:SS):",
        default="04:00:00",
        validate=_validate_time,
        style=STYLE,
    ))

    # ── Partition (auto-suggested, overridable) ──
    print()
    suggested = _suggest_partition(cluster, is_gpu, params.gpu_type)

    if suggested:
        selected_partition = _get_partition(cluster, suggested)
        time_note = ""
        if selected_partition and selected_partition.max_time:
            time_note = f", max time {selected_partition.max_time}"

        questionary.print(
            f"  Suggested partition: {suggested}{time_note}",
            style="fg:ansibrightblack",
        )
        change_part = _ask_or_abort(
            questionary.confirm(
                f"Use partition '{suggested}'?", default=True, style=STYLE
            )
        )
        if change_part:
            params.partition = suggested
        else:
            relevant = cluster.gpu_partitions if is_gpu else cluster.cpu_partitions
            if not relevant:
                relevant = [p for p in cluster.partitions if p.state == "UP"]
            part_choices = _partition_choices(relevant, show_gpu_info=is_gpu)
            if part_choices:
                params.partition = _ask_or_abort(questionary.select(
                    "Partition:",
                    choices=part_choices,
                    style=STYLE,
                ))
            else:
                params.partition = _ask_or_abort(questionary.text(
                    "Partition name:",
                    style=STYLE,
                ))
    elif cluster.partitions:
        relevant = cluster.gpu_partitions if is_gpu else cluster.cpu_partitions
        if not relevant:
            relevant = [p for p in cluster.partitions if p.state == "UP"]
        part_choices = _partition_choices(relevant, show_gpu_info=is_gpu)
        if part_choices:
            params.partition = _ask_or_abort(questionary.select(
                "Partition:",
                choices=part_choices,
                style=STYLE,
            ))
    else:
        params.partition = _ask_or_abort(questionary.text(
            "Partition (leave blank for cluster default):",
            style=STYLE,
        )) or None

    # ── Output & notifications ──
    print()
    questionary.print("── Output & Notifications ──", style="bold fg:yellow")

    params.output = _ask_or_abort(questionary.text(
        "Stdout file pattern:",
        default="%x_%j.out",
        style=STYLE,
    ))

    params.error = _ask_or_abort(questionary.text(
        "Stderr file pattern:",
        default="%x_%j.err",
        style=STYLE,
    ))

    params.mail_type = _ask_or_abort(questionary.select(
        "Email notifications:",
        choices=MAIL_TYPE_CHOICES,
        default="ALL",
        style=STYLE,
    ))

    if params.mail_type and params.mail_type != "NONE":
        params.mail_user = _ask_or_abort(questionary.text(
            "Email address:",
            style=STYLE,
        ))

    # ── Job array ──
    use_array = _ask_or_abort(
        questionary.confirm("Is this a job array?", default=False, style=STYLE)
    )
    if use_array:
        params.array = _ask_or_abort(questionary.text(
            "Array spec (e.g. 0-99, 1-10%5):",
            style=STYLE,
        ))

    # ── Environment ──
    print()
    questionary.print("── Environment Setup ──", style="bold fg:yellow")

    use_modules = _ask_or_abort(
        questionary.confirm("Load environment modules?", default=True, style=STYLE)
    )
    if use_modules:
        mod_input = _ask_or_abort(questionary.text(
            "Modules to load (space-separated, e.g. python3 cuda/12.1):",
            style=STYLE,
        ))
        if mod_input:
            params.modules = mod_input.split()

    use_conda = _ask_or_abort(
        questionary.confirm(
            "Activate a conda/mamba environment?", default=False, style=STYLE
        )
    )
    if use_conda:
        params.conda_env = _ask_or_abort(questionary.text(
            "Conda environment name:",
            style=STYLE,
        ))

    # ── Commands ──
    print()
    questionary.print("── Commands ──", style="bold fg:yellow")
    questionary.print(
        "  Enter the commands your job should run. Leave blank to finish.",
        style="fg:ansibrightblack",
    )

    while True:
        prompt = "Command:" if not params.commands else "Next command (blank to finish):"
        cmd = _ask_or_abort(questionary.text(prompt, style=STYLE))
        if not cmd.strip():
            break
        params.commands.append(cmd.strip())

    if not params.commands:
        params.commands = ["echo 'Hello from SLURM job'"]

    # ── Advanced ──
    print()
    advanced = _ask_or_abort(
        questionary.confirm("Configure advanced options?", default=False, style=STYLE)
    )

    if advanced:
        if cluster.features:
            use_constraint = _ask_or_abort(
                questionary.confirm(
                    "Add node feature constraints?", default=False, style=STYLE
                )
            )
            if use_constraint:
                selected_features = _ask_or_abort(questionary.checkbox(
                    "Select features:",
                    choices=sorted(cluster.features),
                    style=STYLE,
                ))
                if selected_features:
                    params.constraint = "&".join(selected_features)

        params.exclusive = _ask_or_abort(
            questionary.confirm("Exclusive node access?", default=False, style=STYLE)
        )

        dep_input = _ask_or_abort(questionary.text(
            "Job dependency (e.g. afterok:12345, blank for none):",
            default="",
            style=STYLE,
        ))
        if dep_input:
            params.dependency = dep_input

        params.workdir = _ask_or_abort(questionary.text(
            "Working directory (blank for submission dir):",
            default="",
            style=STYLE,
        )) or None

        extra = _ask_or_abort(questionary.text(
            "Extra #SBATCH lines (semicolon-separated, blank for none):",
            default="",
            style=STYLE,
        ))
        if extra:
            params.extra_sbatch_lines = [
                line.strip() for line in extra.split(";") if line.strip()
            ]

    return params
