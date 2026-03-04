"""Interactive prompts for gathering job submission parameters.

Uses questionary for rich terminal prompts (fuzzy select, validation, etc.)
and adapts available choices based on detected cluster configuration.
GPU and CPU jobs follow distinct resource models per SLURM best practices.
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


# ── Choice builders ──

def _partition_choices(
    partitions: List[Partition],
    show_gpu_info: bool = False,
) -> List[questionary.Choice]:
    """Build partition choices with resource details."""
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
        default_tag = " *" if p.is_default else ""
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


def _gpu_type_choices(
    cluster: ClusterInfo,
    partition: Optional[Partition],
) -> List[str]:
    """Get GPU types, preferring partition-specific types if available."""
    if partition and partition.gpu_types:
        return partition.gpu_types
    return cluster.gpu_types


def _default_email(cluster: ClusterInfo) -> str:
    if cluster.email_domain:
        return f"{cluster.username}@{cluster.email_domain}"
    return ""


def _ask_or_abort(question: questionary.Question) -> str:
    """Run a questionary prompt, raising KeyboardInterrupt on Ctrl-C / None."""
    result = question.ask()
    if result is None:
        raise KeyboardInterrupt
    return result


# ── Main prompt flow ──

def gather_parameters(cluster: ClusterInfo) -> JobParameters:
    """Run the full interactive prompt flow and return a filled JobParameters."""
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

    # ── Account ──
    if cluster.accounts:
        print()
        questionary.print("── Account & Partition ──", style="bold fg:yellow")
        if len(cluster.accounts) == 1:
            params.account = cluster.accounts[0].name
            questionary.print(
                f"  SLURM account: {params.account}", style="fg:ansibrightblack"
            )
        else:
            default_acct = cluster.default_account or cluster.accounts[0].name
            params.account = _ask_or_abort(questionary.select(
                "SLURM account (project/allocation):",
                choices=_account_choices(cluster),
                default=default_acct,
                style=STYLE,
            ))

    # ── Partition ──
    if cluster.partitions:
        if not cluster.accounts:
            print()
            questionary.print("── Account & Partition ──", style="bold fg:yellow")

        if is_gpu:
            relevant = cluster.gpu_partitions
            label = "Partition (GPU partitions):"
        else:
            relevant = cluster.cpu_partitions
            label = "Partition (CPU partitions):"

        if not relevant:
            relevant = [p for p in cluster.partitions if p.state == "UP"]
            label = "Partition:"

        part_choices = _partition_choices(relevant, show_gpu_info=is_gpu)

        default_part = None
        relevant_names = {p.name for p in relevant}
        if cluster.default_partition and cluster.default_partition in relevant_names:
            default_part = cluster.default_partition

        if part_choices:
            params.partition = _ask_or_abort(questionary.select(
                label,
                choices=part_choices,
                default=default_part,
                style=STYLE,
            ))
    else:
        params.partition = _ask_or_abort(questionary.text(
            "Partition (leave blank for cluster default):",
            style=STYLE,
        )) or None

    selected_partition = (
        _get_partition(cluster, params.partition) if params.partition else None
    )

    # ── QOS ──
    available_qos = _qos_choices(cluster, params.account)
    if available_qos:
        use_qos = _ask_or_abort(
            questionary.confirm("Specify a QOS?", default=False, style=STYLE)
        )
        if use_qos:
            params.qos = _ask_or_abort(questionary.select(
                "QOS:",
                choices=available_qos,
                style=STYLE,
            ))

    # ── Resources (GPU path) ──
    if is_gpu:
        print()
        questionary.print("── GPU Resources ──", style="bold fg:yellow")

        available_types = _gpu_type_choices(cluster, selected_partition)
        if available_types:
            params.gpu_type = _ask_or_abort(questionary.select(
                "GPU type:",
                choices=available_types,
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

    # ── Resources (CPU path) ──
    else:
        print()
        questionary.print("── CPU Resources ──", style="bold fg:yellow")

        params.cpus_per_task = int(_ask_or_abort(questionary.text(
            "CPUs per task (cores):",
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
    print()
    questionary.print("── Time Limit ──", style="bold fg:yellow")

    default_time = "04:00:00"
    if selected_partition and selected_partition.default_time:
        default_time = selected_partition.default_time

    time_hint = ""
    if selected_partition and selected_partition.max_time:
        time_hint = f" (partition max: {selected_partition.max_time})"

    params.time = _ask_or_abort(questionary.text(
        f"Wall time{time_hint}:",
        default=default_time,
        validate=_validate_time,
        style=STYLE,
    ))

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
            default=_default_email(cluster),
            style=STYLE,
        ))

    # ── Constraints / Features ──
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
