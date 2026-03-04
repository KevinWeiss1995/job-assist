"""Auto-detect SLURM cluster configuration.

Queries scontrol, sinfo, and sacctmgr to build a complete picture of what the
cluster offers: partitions, GRES/TRES, accounts, QOS, node features, and limits.
All detection is best-effort — missing commands or permissions are handled gracefully.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class GresResource:
    """A single GRES type available on the cluster (e.g. gpu:a100:4)."""

    name: str  # e.g. "gpu"
    gres_type: Optional[str] = None  # e.g. "a100"
    count: int = 0

    def __str__(self) -> str:
        parts = [self.name]
        if self.gres_type:
            parts.append(self.gres_type)
        if self.count:
            parts.append(str(self.count))
        return ":".join(parts)


@dataclass
class Partition:
    """A SLURM partition and its resource limits."""

    name: str
    is_default: bool = False
    state: str = "UP"
    max_time: Optional[str] = None  # timelimit string e.g. "7-00:00:00"
    default_time: Optional[str] = None
    max_nodes: Optional[int] = None
    max_cpus_per_node: Optional[int] = None
    max_mem_per_node: Optional[int] = None  # MB
    max_mem_per_cpu: Optional[int] = None  # MB
    total_nodes: int = 0
    avail_nodes: int = 0
    cpus_per_node: Optional[int] = None
    gres: List[GresResource] = field(default_factory=list)
    features: Set[str] = field(default_factory=set)
    allow_accounts: Optional[List[str]] = None  # None means all
    deny_accounts: Optional[List[str]] = None
    allow_qos: Optional[List[str]] = None
    deny_qos: Optional[List[str]] = None
    qos: Optional[str] = None  # partition QOS


@dataclass
class AccountInfo:
    """A SLURM account the user has access to."""

    name: str
    is_default: bool = False
    qos_list: List[str] = field(default_factory=list)
    default_qos: Optional[str] = None
    max_tres: Optional[str] = None
    grp_tres: Optional[str] = None


@dataclass
class QosInfo:
    """A SLURM Quality of Service definition."""

    name: str
    max_wall: Optional[str] = None
    max_tres_per_user: Optional[str] = None
    max_tres_per_job: Optional[str] = None
    priority: int = 0
    preempt_mode: Optional[str] = None


@dataclass
class ClusterInfo:
    """Complete detected cluster configuration."""

    cluster_name: Optional[str] = None
    slurm_version: Optional[str] = None
    partitions: List[Partition] = field(default_factory=list)
    default_partition: Optional[str] = None
    accounts: List[AccountInfo] = field(default_factory=list)
    default_account: Optional[str] = None
    qos_list: List[QosInfo] = field(default_factory=list)
    gres_types: Set[str] = field(default_factory=set)
    tres_enabled: bool = False
    gpu_available: bool = False
    gpu_types: List[str] = field(default_factory=list)
    features: Set[str] = field(default_factory=set)
    has_slurm: bool = False
    detection_errors: List[str] = field(default_factory=list)

    @property
    def partition_names(self) -> List[str]:
        return [p.name for p in self.partitions if p.state == "UP"]


def _run(cmd: List[str], timeout: int = 15) -> Tuple[Optional[str], Optional[str]]:
    """Run a command and return (stdout, stderr). Returns (None, error) on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "SLURM_TIME_FORMAT": "standard"},
        )
        if result.returncode != 0:
            return None, result.stderr.strip()
        return result.stdout.strip(), None
    except FileNotFoundError:
        return None, f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return None, f"Command timed out: {' '.join(cmd)}"
    except Exception as e:
        return None, str(e)


def _parse_key_value_block(text: str) -> Dict[str, str]:
    """Parse SLURM key=value output (scontrol style, space-separated)."""
    result: Dict[str, str] = {}
    for token in re.split(r"\s+", text):
        if "=" in token:
            key, _, value = token.partition("=")
            result[key.strip()] = value.strip()
    return result


def _parse_scontrol_blocks(text: str) -> List[Dict[str, str]]:
    """Parse multi-record scontrol output into a list of dicts."""
    blocks = []
    for block in re.split(r"\n\n+", text.strip()):
        if block.strip():
            merged = " ".join(block.split("\n"))
            blocks.append(_parse_key_value_block(merged))
    return blocks


def _parse_gres_string(gres_str: str) -> List[GresResource]:
    """Parse a GRES string like 'gpu:a100:4,gpu:v100:2' into GresResource objects."""
    resources = []
    if not gres_str or gres_str in ("(null)", "N/A"):
        return resources
    for entry in gres_str.split(","):
        parts = entry.strip().split(":")
        if not parts or not parts[0]:
            continue
        name = parts[0]
        gres_type = None
        count = 0
        if len(parts) == 3:
            gres_type = parts[1]
            try:
                count = int(parts[2])
            except ValueError:
                count = 0
        elif len(parts) == 2:
            try:
                count = int(parts[1])
            except ValueError:
                gres_type = parts[1]
        resources.append(GresResource(name=name, gres_type=gres_type, count=count))
    return resources


def _parse_mem_string(mem_str: str) -> Optional[int]:
    """Parse memory strings like '256000', '256G', '512000M' into MB."""
    if not mem_str or mem_str in ("(null)", "UNLIMITED", "0"):
        return None
    mem_str = mem_str.strip().upper()
    match = re.match(r"^(\d+)\s*([KMGT]?)B?$", mem_str)
    if not match:
        try:
            return int(mem_str)
        except ValueError:
            return None
    value = int(match.group(1))
    suffix = match.group(2)
    multipliers = {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    return int(value * multipliers.get(suffix, 1))


def _detect_slurm_present() -> bool:
    """Check if SLURM commands are available."""
    return shutil.which("sinfo") is not None


def _detect_cluster_config(info: ClusterInfo) -> None:
    """Detect basic cluster configuration via scontrol show config."""
    out, err = _run(["scontrol", "show", "config"])
    if err or not out:
        info.detection_errors.append(f"scontrol show config: {err}")
        return

    for line in out.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if key == "ClusterName":
            info.cluster_name = value
        elif key == "SLURM_VERSION":
            info.slurm_version = value
        elif key == "AccountingStorageTRES":
            info.tres_enabled = bool(value and value != "(null)")
            if "gres/gpu" in value:
                info.gpu_available = True


def _detect_partitions(info: ClusterInfo) -> None:
    """Detect partitions and their properties via scontrol show partition."""
    out, err = _run(["scontrol", "show", "partition", "-o"])
    if err or not out:
        info.detection_errors.append(f"scontrol show partition: {err}")
        return

    for line in out.strip().splitlines():
        kv = _parse_key_value_block(line)
        if not kv.get("PartitionName"):
            continue

        p = Partition(name=kv["PartitionName"])
        p.is_default = kv.get("Default", "NO") == "YES"
        p.state = kv.get("State", "UNKNOWN")
        p.max_time = kv.get("MaxTime") if kv.get("MaxTime") != "UNLIMITED" else None
        raw_default_time = kv.get("DefaultTime")
        p.default_time = raw_default_time if raw_default_time not in ("NONE", None) else None
        p.total_nodes = int(kv.get("TotalNodes", 0))

        max_mem = kv.get("MaxMemPerNode")
        if max_mem and max_mem != "UNLIMITED":
            p.max_mem_per_node = _parse_mem_string(max_mem)

        max_mem_cpu = kv.get("MaxMemPerCPU")
        if max_mem_cpu and max_mem_cpu != "UNLIMITED":
            p.max_mem_per_cpu = _parse_mem_string(max_mem_cpu)

        allow_acct = kv.get("AllowAccounts")
        if allow_acct and allow_acct != "ALL":
            p.allow_accounts = [a.strip() for a in allow_acct.split(",") if a.strip()]

        deny_acct = kv.get("DenyAccounts")
        if deny_acct and deny_acct != "(null)":
            p.deny_accounts = [a.strip() for a in deny_acct.split(",") if a.strip()]

        allow_qos = kv.get("AllowQos")
        if allow_qos and allow_qos != "ALL":
            p.allow_qos = [q.strip() for q in allow_qos.split(",") if q.strip()]

        deny_qos = kv.get("DenyQos")
        if deny_qos and deny_qos != "(null)":
            p.deny_qos = [q.strip() for q in deny_qos.split(",") if q.strip()]

        p.qos = kv.get("QoS") if kv.get("QoS") not in ("N/A", "(null)", None) else None

        if p.is_default:
            info.default_partition = p.name

        info.partitions.append(p)


def _detect_partition_resources(info: ClusterInfo) -> None:
    """Augment partitions with per-node resource info from sinfo."""
    out, err = _run([
        "sinfo",
        "-o", "%R|%G|%f|%m|%c|%D|%a",
        "--noheader",
    ])
    if err or not out:
        info.detection_errors.append(f"sinfo resources: {err}")
        return

    partition_map = {p.name: p for p in info.partitions}
    seen = set()

    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 7:
            continue
        pname = parts[0].strip().rstrip("*")
        if pname in seen:
            continue
        seen.add(pname)

        p = partition_map.get(pname)
        if not p:
            continue

        gres_str = parts[1].strip()
        p.gres = _parse_gres_string(gres_str)
        for g in p.gres:
            if g.name == "gpu":
                info.gpu_available = True
                info.gres_types.add("gpu")
                if g.gres_type:
                    if g.gres_type not in info.gpu_types:
                        info.gpu_types.append(g.gres_type)

        features_str = parts[2].strip()
        if features_str and features_str != "(null)":
            p.features = {f.strip() for f in features_str.split(",") if f.strip()}
            info.features.update(p.features)

        try:
            p.max_mem_per_node = p.max_mem_per_node or int(parts[3].strip().rstrip("+"))
        except (ValueError, IndexError):
            pass

        try:
            p.cpus_per_node = int(parts[4].strip().rstrip("+"))
        except (ValueError, IndexError):
            pass


def _detect_accounts(info: ClusterInfo) -> None:
    """Detect user's available accounts and associated QOS via sacctmgr."""
    user = os.environ.get("USER", os.environ.get("LOGNAME", ""))
    if not user:
        info.detection_errors.append("Cannot determine username for account lookup")
        return

    out, err = _run([
        "sacctmgr", "show", "user", user, "withassoc",
        "format=Account,DefaultAccount,QOS,DefaultQOS,MaxTRES,GrpTRES",
        "-n", "-P",
    ])
    if err or not out:
        info.detection_errors.append(f"sacctmgr user assoc: {err}")
        return

    seen_accounts: Set[str] = set()
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue

        acct_name = parts[0].strip()
        if not acct_name or acct_name in seen_accounts:
            continue
        seen_accounts.add(acct_name)

        acct = AccountInfo(name=acct_name)
        acct.is_default = parts[1].strip() == acct_name if len(parts) > 1 else False
        qos_str = parts[2].strip() if len(parts) > 2 else ""
        acct.qos_list = [q.strip() for q in qos_str.split(",") if q.strip()]
        acct.default_qos = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
        acct.max_tres = parts[4].strip() if len(parts) > 4 and parts[4].strip() else None
        acct.grp_tres = parts[5].strip() if len(parts) > 5 and parts[5].strip() else None

        if acct.is_default:
            info.default_account = acct_name

        info.accounts.append(acct)


def _detect_qos(info: ClusterInfo) -> None:
    """Detect available QOS definitions via sacctmgr."""
    out, err = _run([
        "sacctmgr", "show", "qos",
        "format=Name,MaxWall,MaxTRESPerUser,MaxTRESPerJob,Priority,PreemptMode",
        "-n", "-P",
    ])
    if err or not out:
        info.detection_errors.append(f"sacctmgr qos: {err}")
        return

    for line in out.strip().splitlines():
        parts = line.split("|")
        if not parts or not parts[0].strip():
            continue

        qos = QosInfo(name=parts[0].strip())
        if len(parts) > 1 and parts[1].strip():
            qos.max_wall = parts[1].strip()
        if len(parts) > 2 and parts[2].strip():
            qos.max_tres_per_user = parts[2].strip()
        if len(parts) > 3 and parts[3].strip():
            qos.max_tres_per_job = parts[3].strip()
        if len(parts) > 4 and parts[4].strip():
            try:
                qos.priority = int(parts[4].strip())
            except ValueError:
                pass
        if len(parts) > 5 and parts[5].strip():
            qos.preempt_mode = parts[5].strip()

        info.qos_list.append(qos)


def _detect_gpu_types(info: ClusterInfo) -> None:
    """Detect specific GPU types available via sinfo GRES."""
    out, err = _run(["sinfo", "-o", "%G", "--noheader"])
    if err or not out:
        return

    gpu_types: Set[str] = set()
    for line in out.strip().splitlines():
        for gres in _parse_gres_string(line.strip()):
            if gres.name == "gpu" and gres.gres_type:
                gpu_types.add(gres.gres_type)

    info.gpu_types = sorted(gpu_types)


def detect_cluster() -> ClusterInfo:
    """Run full cluster detection and return a ClusterInfo object.

    Safe to call on non-SLURM systems — will return a mostly-empty ClusterInfo
    with has_slurm=False.
    """
    info = ClusterInfo()
    info.has_slurm = _detect_slurm_present()

    if not info.has_slurm:
        info.detection_errors.append("SLURM commands not found in PATH")
        return info

    _detect_cluster_config(info)
    _detect_partitions(info)
    _detect_partition_resources(info)
    _detect_accounts(info)
    _detect_qos(info)
    _detect_gpu_types(info)

    return info
